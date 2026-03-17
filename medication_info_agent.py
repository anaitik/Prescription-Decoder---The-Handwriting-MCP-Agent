import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

from server import explain_medications

STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
STATE_WORKING = "TASK_STATE_WORKING"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_FAILED = "TASK_STATE_FAILED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_message(
    role: str,
    parts: List[Dict[str, Any]],
    message_id: Optional[str] = None,
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
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
    return msg


def _task_object(
    task_id: str,
    context_id: str,
    state: str,
    message: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
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
    return task


class MedicationInfoAgent:
    """A2A-compliant agent that explains medications in plain language."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def send_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        message = request.get("message") or {}
        config = request.get("configuration") or {}
        return_immediately = bool(config.get("returnImmediately"))

        task_id = str(uuid.uuid4())
        context_id = message.get("contextId") or str(uuid.uuid4())
        task = _task_object(task_id, context_id, STATE_SUBMITTED)

        with self._lock:
            self._tasks[task_id] = {"task": task}

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
            self._tasks[task_id] = {"task": task}

        yield {"task": task}
        yield {
            "statusUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "status": {"state": STATE_WORKING, "timestamp": _utc_now()},
            }
        }

        try:
            report, artifacts = self._process_request(task_id, request)
        except Exception as exc:
            error_message = _new_message(
                role="ROLE_AGENT",
                parts=[{"text": f"Task failed: {exc}"}],
                context_id=context_id,
                task_id=task_id,
            )
            yield {
                "statusUpdate": {
                    "taskId": task_id,
                    "contextId": context_id,
                    "status": {
                        "state": STATE_FAILED,
                        "timestamp": _utc_now(),
                        "message": error_message,
                    },
                }
            }
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

        yield {
            "statusUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "status": {"state": STATE_COMPLETED, "timestamp": _utc_now()},
            }
        }

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
        metadata = message.get("metadata") or {}

        meds = self._extract_medications(parts, metadata)
        if not meds:
            raise ValueError("No medications provided for explanation.")

        explanations = explain_medications(medications=meds)
        report = {"medications": meds, "explanations": explanations}

        artifacts = [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "medication_explanations.json",
                "description": "Medication explanations and common indications",
                "parts": [{"data": report, "mediaType": "application/json"}],
            }
        ]

        return report, artifacts

    @staticmethod
    def _extract_medications(
        parts: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> List[Any]:
        # Accept medications from message parts or metadata.
        for part in parts:
            if "data" in part and isinstance(part["data"], dict):
                data = part["data"]
                if "medications" in data:
                    return data["medications"] or []
        return metadata.get("medications") or []
