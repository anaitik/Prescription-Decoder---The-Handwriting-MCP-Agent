# Prescription Decoder MCP + A2A Agent

## Overview
This repository provides a healthcare-focused MCP server and an A2A-compliant agent for decoding prescription images into FHIR R4 `MedicationRequest` resources, running safety checks (drug interactions and allergies), and communicating patient-friendly summaries via translation and SMS. It is designed for staged development with mock FHIR data and mock interaction databases, while preserving the interfaces needed for production integrations.

Key capabilities:
1. MCP tools for prescription decoding, interaction checks, allergy checks, translation, and SMS delivery.
2. A2A agent orchestration with task lifecycle support (submitted, working, completed, failed).
3. FHIR-aware output and SHARP context handling for patient identity and access tokens.

## Architecture Diagram (Text)
```
User/App
  |
  |  (A2A sendMessage)
  v
PrescriptionCompleterAgent
  |-- decode_prescription (MCP tool, OpenAI vision)
  |-- check_drug_interactions (MCP tool, mock DB)
  |-- check_allergies (MCP tool, mock FHIR)
  |-- translate_to_hindi (MCP tool, OpenAI text)
  |-- send_sms (MCP tool, Twilio)
  v
FHIR MedicationRequest + Safety Report + Patient Summary
```

## Setup Instructions
1. Create a virtual environment and install dependencies.
2. Install required SDKs.
3. Set environment variables.
4. Review `config.json` for model and SMS settings.
5. Run the MCP server or the A2A demo.

Suggested install:
```bash
pip install mcp openai fhir.resources twilio requests
```

Run the demo:
```bash
python demo.py
```

Run the Streamlit frontend:
```bash
pip install streamlit
streamlit run streamlit_app.py
```

## Environment Variables Needed
Required:
- `OPENAI_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`

Optional:
- `MCP_CONFIG_PATH` (path to `config.json`)

SHARP context headers expected on requests:
- `X-Patient-ID`
- `X-FHIR-Access-Token`
- `X-FHIR-Server-URL` (optional)

FHIR patient preferences:
- The agent reads patient language and phone from the FHIR `Patient` resource (`communication` and `telecom`).
- Set `use_mock_fhir` to `false` in `config.json` to enable real FHIR calls.

## Example API Calls

MCP tool call (conceptual JSON payload):
```json
{
  "tool": "decode_prescription",
  "input": {
    "image_data": "<base64-image>",
    "patient_id": "patient-123"
  }
}
```

A2A synchronous request:
```json
{
  "message": {
    "role": "ROLE_USER",
    "parts": [
      { "raw": "<base64-image>", "mediaType": "image/png" }
    ],
    "metadata": {
      "patient_id": "patient-123",
      "fhir_access_token": "token-abc",
      "fhir_server_url": "https://fhir.example.com"
    }
  }
}
```

A2A async request (return immediately):
```json
{
  "message": { "...": "..." },
  "configuration": { "returnImmediately": true }
}
```

## Best Practices (Healthcare)
Logging, error handling, and audit trails are critical. The demo script includes:
- Structured logging for each major step.
- Clear error handling for missing data or external failures.
- A simple audit trail for actions and outcomes.

In production:
- Store audit trails in a compliant system of record.
- Enforce least-privilege access to FHIR and SMS APIs.
- Use secure secrets management.
- Validate FHIR payloads and log patient-impacting decisions.
