import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import streamlit as st
import pandas as pd

from a2a_agent import PrescriptionCompleterAgent

# Basic logging for healthcare demos; use structured logging in production.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("streamlit_app")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _audit_event(audit: List[Dict[str, Any]], event: str, details: Dict[str, Any]) -> None:
    # Minimal audit trail. Replace with a compliant audit sink in production.
    audit.append({"timestamp": _utc_now(), "event": event, "details": details})


def _build_request(
    image_b64: str,
    media_type: str,
    patient_id: str,
    fhir_access_token: str,
    fhir_server_url: str,
) -> Dict[str, Any]:
    return {
        "message": {
            "role": "ROLE_USER",
            "parts": [{"raw": image_b64, "mediaType": media_type}],
            "metadata": {
                "patient_id": patient_id,
                "fhir_access_token": fhir_access_token,
                "fhir_server_url": fhir_server_url,
            },
        }
    }


def _extract_report(task: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = task.get("artifacts") or []
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if part.get("mediaType") == "application/json":
                return part.get("data") or {}
    return {}


def main() -> None:
    st.set_page_config(page_title="Prescription Completer", layout="wide")
    st.title("PrescriptionCompleterAgent Demo")
    st.markdown(
        "Upload a prescription image and provide patient context to generate FHIR resources, "
        "interaction checks, allergy alerts, and patient-friendly messaging."
    )

    audit_trail: List[Dict[str, Any]] = []

    with st.sidebar:
        st.header("Patient Context")
        patient_id = st.text_input("Patient ID", value="patient-123")
        fhir_access_token = st.text_input("FHIR Access Token", value="demo-token", type="password")
        fhir_server_url = st.text_input("FHIR Server URL", value="https://fhir.example.com")
        st.caption("These values are required for SHARP context propagation.")

    uploaded_file = st.file_uploader(
        "Upload prescription image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if st.button("Run Workflow", type="primary"):
        if not uploaded_file:
            st.error("Please upload a prescription image.")
            return
        if not patient_id or not fhir_access_token:
            st.error("Please provide Patient ID and FHIR Access Token.")
            return

        try:
            image_bytes = uploaded_file.read()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            media_type = uploaded_file.type or "image/png"

            _audit_event(audit_trail, "workflow_submitted", {"patient_id": patient_id})

            agent = PrescriptionCompleterAgent()
            request = _build_request(
                image_b64=image_b64,
                media_type=media_type,
                patient_id=patient_id,
                fhir_access_token=fhir_access_token,
                fhir_server_url=fhir_server_url,
            )

            with st.spinner("Processing prescription..."):
                response = agent.send_message(request)

            task = response.get("task") or {}
            status = task.get("status") or {}
            if status.get("state") == "TASK_STATE_FAILED":
                message = status.get("message") or {}
                parts = message.get("parts") or []
                error_text = None
                if parts:
                    error_text = parts[0].get("text")
                st.error(error_text or "Workflow failed.")
                _audit_event(audit_trail, "workflow_failed", {"task_state": status})
                st.subheader("Audit Trail")
                st.json(audit_trail)
                return

            report = _extract_report(task)
            _audit_event(audit_trail, "workflow_completed", {"task_state": task.get("status")})

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Clinician View")
                if not report:
                    st.json({"message": "No report data returned."})
                else:
                    st.markdown("**Summary**")
                    st.write(report.get("summary") or "No summary.")

                    meds = report.get("prescription") or []
                    med_rows = []
                    for med in meds:
                        med_cc = med.get("medicationCodeableConcept") or med.get("medication") or {}
                        if isinstance(med_cc, dict) and "concept" in med_cc:
                            med_cc = med_cc.get("concept") or {}
                        med_name = (
                            med_cc.get("text")
                            if isinstance(med_cc, dict)
                            else "Medication"
                        )
                        dosage = ""
                        dosage_instructions = med.get("dosageInstruction") or []
                        if dosage_instructions:
                            dosage = dosage_instructions[0].get("text") or ""
                        med_rows.append(
                            {
                                "Medication": med_name or "Medication",
                                "Instructions": dosage,
                                "Status": med.get("status"),
                                "Intent": med.get("intent"),
                            }
                        )
                    if med_rows:
                        st.markdown("**Medications**")
                        st.dataframe(pd.DataFrame(med_rows), use_container_width=True)

                    interactions = report.get("interaction_warnings") or []
                    if interactions:
                        st.markdown("**Interaction Warnings**")
                        st.dataframe(pd.DataFrame(interactions), use_container_width=True)
                    else:
                        st.markdown("**Interaction Warnings**")
                        st.write("None")

                    allergies = report.get("allergy_alerts") or []
                    if allergies:
                        st.markdown("**Allergy Alerts**")
                        st.dataframe(pd.DataFrame(allergies), use_container_width=True)
                    else:
                        st.markdown("**Allergy Alerts**")
                        st.write("None")

                    with st.expander("Raw JSON"):
                        st.json(report)
            with col2:
                st.subheader("Patient View")
                st.text_area(
                    "Patient Message",
                    value=report.get("patient_message") or "No patient message returned.",
                    height=300,
                )

            st.subheader("Audit Trail")
            st.json(audit_trail)
        except Exception as exc:
            LOGGER.exception("Workflow failed")
            _audit_event(audit_trail, "workflow_failed", {"error": str(exc)})
            st.error(f"Workflow failed: {exc}")


if __name__ == "__main__":
    main()
