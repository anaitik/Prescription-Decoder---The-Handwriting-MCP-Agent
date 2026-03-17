import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from a2a_agent import PrescriptionCompleterAgent

# Basic logging configuration suitable for healthcare demos.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("demo")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _audit_event(audit: List[Dict[str, Any]], event: str, details: Dict[str, Any]) -> None:
    # Minimal audit trail for demo purposes. Replace with a secure audit sink in production.
    audit.append({"timestamp": _utc_now(), "event": event, "details": details})


def _sample_image_base64() -> str:
    # Tiny PNG header as a placeholder image payload.
    dummy_png_header = b"\x89PNG\r\n\x1a\n"
    return base64.b64encode(dummy_png_header).decode("ascii")


def _build_request() -> Dict[str, Any]:
    # A2A request with message parts and SHARP context in metadata.
    return {
        "message": {
            "role": "ROLE_USER",
            "parts": [
                {"raw": _sample_image_base64(), "mediaType": "image/png"},
            ],
            "metadata": {
                "patient_id": "patient-123",
                "fhir_access_token": "demo-token",
                "fhir_server_url": "https://fhir.example.com",
            },
        }
    }


def _mock_response() -> Dict[str, Any]:
    # Mocked clinician and patient views to keep the demo runnable without external services.
    return {
        "patient_id": "patient-123",
        "prescription": [
            {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {"text": "Amoxicillin"},
                "dosageInstruction": [{"text": "250 mg twice daily for 5 days"}],
            }
        ],
        "interaction_warnings": [
            {
                "medications": "aspirin, warfarin",
                "severity": "high",
                "description": "Increased risk of bleeding.",
                "recommendation": "Avoid combination if possible.",
            }
        ],
        "allergy_alerts": [
            {
                "medication": "Amoxicillin",
                "allergy": "Penicillin",
                "criticality": "high",
                "reaction": "Rash",
                "note": "Possible cross-reactivity",
            }
        ],
        "summary": "We identified 1 medication(s) on your prescription. We found 1 potential interaction(s). We found 1 potential allergy-related alert(s). Please review these results with your healthcare provider before starting any medication.",
        "patient_message": "आपकी दवाओं का सारांश: एमोक्सिसिलिन 250 mg दिन में दो बार 5 दिनों के लिए।",
        "language": "hi",
        "sms_status": {"status": "mocked"},
    }


def main() -> int:
    audit_trail: List[Dict[str, Any]] = []
    agent = PrescriptionCompleterAgent()
    request = _build_request()

    LOGGER.info("Starting prescription workflow demo.")
    _audit_event(audit_trail, "workflow_submitted", {"patient_id": "patient-123"})

    use_real_services = bool(os.getenv("OPENAI_API_KEY")) and bool(
        os.getenv("TWILIO_ACCOUNT_SID")
    )

    if not use_real_services:
        LOGGER.warning("Missing API keys. Running demo with mocked outputs.")
        _audit_event(audit_trail, "workflow_mocked", {"reason": "missing_api_keys"})
        response = {"task": {"status": {"state": "TASK_STATE_COMPLETED"}}}
        report = _mock_response()
    else:
        try:
            response = agent.send_message(request)
            _audit_event(audit_trail, "workflow_completed", {"mode": "sync"})
            task = response.get("task") or {}
            artifacts = task.get("artifacts") or []
            report = {}
            if artifacts:
                for part in artifacts[0].get("parts", []):
                    if part.get("mediaType") == "application/json":
                        report = part.get("data") or {}
        except Exception as exc:
            LOGGER.error("Workflow failed: %s", exc)
            _audit_event(audit_trail, "workflow_failed", {"error": str(exc)})
            return 1

    LOGGER.info("Clinician view:\n%s", json.dumps(report, indent=2))
    LOGGER.info("Patient view:\n%s", report.get("patient_message"))

    LOGGER.info("Audit trail:\n%s", json.dumps(audit_trail, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
