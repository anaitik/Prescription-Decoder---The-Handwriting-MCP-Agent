import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

from server import (
    _fetch_patient_profile,
    _load_config,
    check_allergies,
    check_drug_interactions,
    decode_prescription,
    send_sms,
    translate_to_hindi,
)

STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
STATE_WORKING = "TASK_STATE_WORKING"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_FAILED = "TASK_STATE_FAILED"

LOGGER = logging.getLogger("prescription_agent")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_message(
    role: str,
    parts: List[Dict[str, Any]],
    message_id: Optional[str] = None,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "messageId": message_id or str(uuid.uuid4()),
        "role": role,
        "parts": parts,
        "kind": "message",
    }
    if context_id:
        msg["contextId"] = context_id
    if task_id:
        msg["taskId"] = task_id
    if metadata:
        msg["metadata"] = metadata
    return msg


def _task_object(
    task_id: str,
    context_id: str,
    state: str,
    message: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {"state": state, "timestamp": _utc_now()}
    if message:
        status["message"] = message

    task: Dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": status,
        "kind": "task",
    }
    if artifacts is not None:
        task["artifacts"] = artifacts
    if history is not None:
        task["history"] = history
    if metadata:
        task["metadata"] = metadata
    return task


class _DummyRequest:
    def __init__(self, headers: Dict[str, str]) -> None:
        self.headers = headers


class _DummyRequestContext:
    def __init__(self, headers: Dict[str, str], meta: Dict[str, Any]) -> None:
        self.request = _DummyRequest(headers)
        self.meta = meta


class _DummyContext:
    def __init__(self, headers: Dict[str, str], meta: Dict[str, Any]) -> None:
        self.request_context = _DummyRequestContext(headers, meta)


def _build_sharp_context(
    patient_id: str,
    fhir_access_token: str,
    fhir_server_url: Optional[str],
) -> _DummyContext:
    headers = {
        "x-patient-id": patient_id,
        "x-fhir-access-token": fhir_access_token,
    }
    if fhir_server_url:
        headers["x-fhir-server-url"] = fhir_server_url
    meta = {
        "patient_id": patient_id,
        "fhir_access_token": fhir_access_token,
        "fhir_server_url": fhir_server_url,
    }
    return _DummyContext(headers=headers, meta=meta)


class PrescriptionCompleterAgent:
    """A2A-compliant agent that orchestrates prescription decoding and safety checks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def send_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        message = request.get("message") or {}
        config = request.get("configuration") or {}
        return_immediately = bool(config.get("returnImmediately"))
        include_history = config.get("historyLength")

        task_id = str(uuid.uuid4())
        context_id = message.get("contextId") or str(uuid.uuid4())
        history = [message] if include_history else None

        task = _task_object(task_id, context_id, STATE_SUBMITTED, history=history)

        with self._lock:
            self._tasks[task_id] = {"task": task, "history": history, "artifacts": []}

        if return_immediately:
            threading.Thread(
                target=self._run_task,
                args=(task_id, request),
                daemon=True,
            ).start()
            return {"task": task}

        completed_task = self._run_task(task_id, request, return_task=True)
        return {"task": completed_task}

    def send_message_stream(
        self, request: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        message = request.get("message") or {}
        task_id = str(uuid.uuid4())
        context_id = message.get("contextId") or str(uuid.uuid4())

        task = _task_object(task_id, context_id, STATE_SUBMITTED)
        with self._lock:
            self._tasks[task_id] = {"task": task, "history": [message], "artifacts": []}

        yield {"task": task}

        status_update = {
            "taskId": task_id,
            "contextId": context_id,
            "status": {"state": STATE_WORKING, "timestamp": _utc_now()},
        }
        yield {"statusUpdate": status_update}

        try:
            report, artifacts = self._process_request(task_id, request)
        except Exception as exc:
            error_message = _new_message(
                role="ROLE_AGENT",
                parts=[{"text": f"Task failed: {exc}"}],
                context_id=context_id,
                task_id=task_id,
            )
            failed_update = {
                "taskId": task_id,
                "contextId": context_id,
                "status": {
                    "state": STATE_FAILED,
                    "timestamp": _utc_now(),
                    "message": error_message,
                },
            }
            yield {"statusUpdate": failed_update}
            return

        for artifact in artifacts:
            yield {
                "artifactUpdate": {
                    "taskId": task_id,
                    "contextId": context_id,
                    "artifact": artifact,
                    "append": False,
                    "lastChunk": True,
                }
            }

        completed_update = {
            "taskId": task_id,
            "contextId": context_id,
            "status": {"state": STATE_COMPLETED, "timestamp": _utc_now()},
        }
        yield {"statusUpdate": completed_update}

    def get_task(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            record = self._tasks.get(task_id)
        if not record:
            raise KeyError(f"Task {task_id} not found.")
        return {"task": record["task"]}

    def _run_task(
        self,
        task_id: str,
        request: Dict[str, Any],
        return_task: bool = False,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._tasks[task_id]
        task = record["task"]
        task["status"] = {"state": STATE_WORKING, "timestamp": _utc_now()}

        try:
            report, artifacts = self._process_request(task_id, request)
            task["status"] = {"state": STATE_COMPLETED, "timestamp": _utc_now()}
            task["artifacts"] = artifacts
        except Exception as exc:
            error_message = _new_message(
                role="ROLE_AGENT",
                parts=[{"text": f"Task failed: {exc}"}],
                context_id=task.get("contextId"),
                task_id=task_id,
            )
            task["status"] = {
                "state": STATE_FAILED,
                "timestamp": _utc_now(),
                "message": error_message,
            }
        with self._lock:
            record["task"] = task
        if return_task:
            return task
        return None

    def _process_request(
        self, task_id: str, request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        message = request.get("message") or {}
        parts = message.get("parts") or []
        metadata = message.get("metadata") or request.get("metadata") or {}

        image_data = self._extract_image_data(parts)
        if not image_data:
            raise ValueError("No image data provided in message parts.")

        patient_id = (
            metadata.get("patient_id")
            or metadata.get("patientId")
            or metadata.get("x-patient-id")
        )
        fhir_access_token = metadata.get("fhir_access_token") or metadata.get(
            "x-fhir-access-token"
        )
        fhir_server_url = metadata.get("fhir_server_url") or metadata.get(
            "x-fhir-server-url"
        )

        if not patient_id or not fhir_access_token:
            raise ValueError(
                "Missing patient context. Provide patient_id and fhir_access_token in metadata."
            )

        ctx = _build_sharp_context(patient_id, fhir_access_token, fhir_server_url)

        meds = decode_prescription(image_data=image_data, patient_id=patient_id, ctx=ctx)

        interaction_reports: List[Dict[str, Any]] = []
        for med in meds:
            report = check_drug_interactions(
                medications=[med],
                patient_id=patient_id,
                ctx=ctx,
            )
            interaction_reports.extend(report.get("interactions") or [])

        allergy_report = check_allergies(
            new_medications=meds,
            patient_id=patient_id,
            ctx=ctx,
        )

        config = _load_config()
        try:
            profile = _fetch_patient_profile(
                patient_id=patient_id,
                fhir_token=fhir_access_token,
                fhir_server_url=fhir_server_url,
            )
        except Exception:
            # Avoid failing the full workflow if patient preferences are unavailable.
            LOGGER.warning("Patient profile lookup failed; using defaults.")
            profile = {}

        preferred_language = (
            profile.get("preferred_language")
            or config.get("default_language")
            or "hi"
        )
        phone_number = profile.get("phone_number")

        summary = self._summarize_results(meds, interaction_reports, allergy_report)
        patient_friendly = self._build_patient_friendly_text(
            meds, interaction_reports, allergy_report, config.get("prescription_view_url")
        )

        translated_text = patient_friendly
        translation_result = None
        sms_result = None
        try:
            translation_result = translate_to_hindi(
                text=patient_friendly, target_language=preferred_language
            )
            translated_text = translation_result.get("translated_text") or translated_text
        except Exception:
            translation_result = {"language": preferred_language, "translated_text": patient_friendly}

        if phone_number:
            try:
                sms_result = send_sms(phone_number=phone_number, message=translated_text)
            except Exception as exc:
                sms_result = {"status": "failed", "error": str(exc), "to": phone_number}

        report_payload = {
            "patient_id": patient_id,
            "prescription": meds,
            "interaction_warnings": interaction_reports,
            "allergy_alerts": allergy_report.get("warnings") or [],
            "summary": summary,
            "patient_message": translated_text,
            "language": preferred_language,
            "sms_status": sms_result,
        }

        artifacts = [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "prescription_report.json",
                "description": "FHIR prescription and safety checks",
                "parts": [
                    {"data": report_payload, "mediaType": "application/json"},
                    {"text": summary, "mediaType": "text/plain"},
                    {"text": translated_text, "mediaType": "text/plain"},
                ],
            }
        ]

        return report_payload, artifacts

    @staticmethod
    def _extract_image_data(parts: List[Dict[str, Any]]) -> Optional[str]:
        for part in parts:
            if "raw" in part and part.get("mediaType", "").startswith("image/"):
                return part["raw"]
            if "data" in part and isinstance(part["data"], dict):
                data = part["data"]
                if "image_data" in data:
                    return data["image_data"]
        return None

    @staticmethod
    def _summarize_results(
        meds: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        allergies: Dict[str, Any],
    ) -> str:
        med_count = len(meds)
        interaction_count = len(interactions)
        allergy_count = len(allergies.get("warnings") or [])

        summary_lines = [
            f"We identified {med_count} medication(s) on your prescription.",
        ]
        if interaction_count:
            summary_lines.append(
                f"We found {interaction_count} potential interaction(s)."
            )
        else:
            summary_lines.append("We did not detect any known drug interactions.")

        if allergy_count:
            summary_lines.append(
                f"We found {allergy_count} potential allergy-related alert(s)."
            )
        else:
            summary_lines.append("We did not detect any allergy conflicts.")

        summary_lines.append(
            "Please review these results with your healthcare provider before starting any medication."
        )
        return " ".join(summary_lines)

    @staticmethod
    def _build_patient_friendly_text(
        meds: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        allergies: Dict[str, Any],
        prescription_link_base: Optional[str],
    ) -> str:
        lines: List[str] = ["Your prescription summary:"]
        if meds:
            lines.append("Medications:")
            for med in meds:
                med_cc = med.get("medicationCodeableConcept") or med.get("medication") or {}
                med_name = (med_cc.get("text") if isinstance(med_cc, dict) else None) or "Medication"
                dosage_instr = ""
                dosage_instructions = med.get("dosageInstruction") or []
                if dosage_instructions:
                    dosage_instr = dosage_instructions[0].get("text") or ""
                lines.append(f"- {med_name} {dosage_instr}".strip())
        if interactions:
            lines.append("Important interaction warnings:")
            for interaction in interactions:
                lines.append(
                    f"- {interaction.get('medications')}: {interaction.get('description')} "
                    f"({interaction.get('severity')})"
                )
        if allergies.get("warnings"):
            lines.append("Allergy alerts:")
            for warning in allergies["warnings"]:
                lines.append(
                    f"- {warning.get('medication')} vs {warning.get('allergy')}: "
                    f"{warning.get('note')} ({warning.get('criticality')})"
                )

        if prescription_link_base:
            lines.append(f"View your full prescription here: {prescription_link_base}")

        lines.append("If you have questions, contact your healthcare provider.")
        return "\n".join(lines)


if __name__ == "__main__":
    agent = PrescriptionCompleterAgent()
    example_request = {
        "message": {
            "role": "ROLE_USER",
            "parts": [
                {
                    "raw": "iVBORw0KGgoAAAANSUhEUgAAAAUA",
                    "mediaType": "image/png",
                    "filename": "prescription.png",
                }
            ],
            "messageId": str(uuid.uuid4()),
            "metadata": {
                "patient_id": "example-patient-1",
                "fhir_access_token": "demo-token",
                "fhir_server_url": "https://fhir.example.com",
            },
        },
        "configuration": {"returnImmediately": True},
    }
    print(agent.send_message(example_request))
