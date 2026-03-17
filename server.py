import base64
import binascii
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP

try:
    from fhir.resources.annotation import Annotation
    from fhir.resources.codeableconcept import CodeableConcept
    from fhir.resources.coding import Coding
    from fhir.resources.dosage import Dosage
    from fhir.resources.medicationrequest import MedicationRequest
    from fhir.resources.reference import Reference
    from fhir.resources.timing import Timing, TimingRepeat
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "fhir.resources is not installed. Install it with: pip install fhir.resources"
    ) from exc

CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.json"))
PROMPT = (
    "Extract all medications, dosages, and instructions from this doctor's prescription. "
    "Format the output as a JSON array with fields: medication_name, dosage, frequency, "
    "duration, and instructions. If something is unclear, make your best guess and note "
    "it with a 'confidence: low' field."
)
FHIR_ID_PATTERN = re.compile(r"^[A-Za-z0-9\-\.]{1,64}$")

mcp = FastMCP("Prescription Decoder", json_response=True)


def _normalize_base64(image_data: str) -> str:
    data = image_data.strip()
    if data.startswith("data:"):
        parts = data.split(",", 1)
        if len(parts) == 2:
            data = parts[1]
    return data


def _detect_mime(decoded: bytes) -> str:
    if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if decoded.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if decoded.startswith(b"GIF87a") or decoded.startswith(b"GIF89a"):
        return "image/gif"
    if decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _to_data_url(original: str, decoded: bytes) -> str:
    raw = original.strip()
    if raw.startswith("data:"):
        return raw
    mime = _detect_mime(decoded)
    encoded = base64.b64encode(decoded).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _normalize_patient_id(value: str) -> str:
    candidate = value.strip()
    if candidate.lower().startswith("patient/"):
        candidate = candidate.split("/", 1)[1]
    if not FHIR_ID_PATTERN.fullmatch(candidate):
        raise ValueError(
            "Invalid patient_id format. Expected a FHIR id matching "
            "[A-Za-z0-9-\\.]{1,64}."
        )
    return candidate


def _get_request_headers(ctx: Optional[Context]) -> Dict[str, str]:
    if ctx is None:
        return {}
    try:
        request_obj = ctx.request_context.request
    except Exception:
        return {}
    if request_obj is None:
        return {}

    headers: Dict[str, str] = {}
    if hasattr(request_obj, "headers"):
        try:
            headers = dict(request_obj.headers)
        except Exception:
            headers = {}

    if not headers and hasattr(request_obj, "scope"):
        scope = getattr(request_obj, "scope")
        if isinstance(scope, dict):
            raw_headers = scope.get("headers") or []
            for key, value in raw_headers:
                try:
                    headers[key.decode("latin-1")] = value.decode("latin-1")
                except Exception:
                    continue

    return {str(k).lower(): str(v) for k, v in headers.items()}


def _get_meta_value(meta: Any, key: str) -> Optional[str]:
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta.get(key)
    return getattr(meta, key, None)


def _extract_sharp_context(
    ctx: Optional[Context],
    patient_id_param: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    headers = _get_request_headers(ctx)
    meta = None
    try:
        meta = ctx.request_context.meta if ctx else None
    except Exception:
        meta = None

    patient_id = (
        headers.get("x-patient-id")
        or _get_meta_value(meta, "patient_id")
        or patient_id_param
    )
    fhir_token = headers.get("x-fhir-access-token") or _get_meta_value(meta, "fhir_access_token")
    fhir_server_url = headers.get("x-fhir-server-url") or _get_meta_value(meta, "fhir_server_url")

    if fhir_token and fhir_token.lower().startswith("bearer "):
        fhir_token = fhir_token[7:].strip()

    if not patient_id or not fhir_token:
        raise ValueError(
            "Missing SHARP context. Provide X-Patient-ID and X-FHIR-Access-Token headers."
        )

    return _normalize_patient_id(patient_id), fhir_token, fhir_server_url


def _fhir_auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def medication_to_rxnorm(med_name: str) -> Optional[Tuple[str, str]]:
    if not med_name:
        return None
    normalized = med_name.strip().lower()
    mapping = {
        "paracetamol": ("161", "Paracetamol"),
        "acetaminophen": ("161", "Acetaminophen"),
        "amoxicillin": ("723", "Amoxicillin"),
        "ibuprofen": ("5640", "Ibuprofen"),
        "azithromycin": ("18631", "Azithromycin"),
        "metformin": ("6809", "Metformin"),
        "omeprazole": ("7646", "Omeprazole"),
        "pantoprazole": ("40790", "Pantoprazole"),
    }
    for key, value in mapping.items():
        if key in normalized:
            return value
    return None


def _parse_frequency(frequency: str) -> Optional[Tuple[int, float, str]]:
    if not frequency:
        return None
    text = frequency.strip().lower()
    if "once daily" in text or "once a day" in text or "daily" == text:
        return (1, 1.0, "d")
    if "twice daily" in text or "twice a day" in text:
        return (2, 1.0, "d")
    if "three times daily" in text or "3 times daily" in text or "thrice daily" in text:
        return (3, 1.0, "d")
    if "every" in text and "hour" in text:
        parts = text.replace("-", " ").split()
        for i, token in enumerate(parts):
            if token.isdigit() and i + 1 < len(parts) and "hour" in parts[i + 1]:
                return (1, float(token), "h")
    return None


def _parse_duration(duration: str) -> Optional[Tuple[float, str]]:
    if not duration:
        return None
    text = duration.strip().lower().replace("-", " ")
    parts = text.split()
    for i, token in enumerate(parts):
        if token.replace(".", "", 1).isdigit() and i + 1 < len(parts):
            unit = parts[i + 1]
            if unit.startswith("day"):
                return (float(token), "d")
            if unit.startswith("week"):
                return (float(token), "wk")
            if unit.startswith("month"):
                return (float(token), "mo")
    return None


def _build_dosage(frequency: str, duration: str, instructions: str) -> Dosage:
    dosage = Dosage()
    dosage.text = " ".join(part for part in [frequency, duration, instructions] if part).strip() or None

    repeat = TimingRepeat()
    freq_tuple = _parse_frequency(frequency)
    if freq_tuple:
        repeat.frequency, repeat.period, repeat.periodUnit = freq_tuple
    dur_tuple = _parse_duration(duration)
    if dur_tuple:
        repeat.duration, repeat.durationUnit = dur_tuple

    if repeat.frequency or repeat.duration:
        dosage.timing = Timing(repeat=repeat)

    return dosage


def _to_medication_request(
    item: Dict[str, Any],
    patient_id: Optional[str],
) -> MedicationRequest:
    med_name = str(item.get("medication_name") or "").strip()
    dosage = str(item.get("dosage") or "").strip()
    frequency = str(item.get("frequency") or "").strip()
    duration = str(item.get("duration") or "").strip()
    instructions = str(item.get("instructions") or "").strip()

    rxnorm = medication_to_rxnorm(med_name)
    coding_list = []
    notes = []

    if rxnorm:
        code, display = rxnorm
        coding_list.append(Coding(system="http://www.nlm.nih.gov/research/umls/rxnorm", code=code, display=display))
    else:
        notes.append(Annotation(text="RxNorm code not found."))

    if str(item.get("confidence", "")).lower() == "low":
        notes.append(Annotation(text="Confidence: low."))

    med_concept = CodeableConcept(text=med_name or None, coding=coding_list or None)
    dosage_instruction = _build_dosage(frequency, duration, " ".join([dosage, instructions]).strip())

    subject_ref = Reference(reference=f"Patient/{patient_id}") if patient_id else None

    return MedicationRequest(
        status="active",
        intent="order",
        medicationCodeableConcept=med_concept,
        subject=subject_ref,
        dosageInstruction=[dosage_instruction],
        note=notes or None,
    )


def _load_config() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "model": "gpt-4.1-mini",
        "max_output_tokens": 512,
        "temperature": 0.2,
        "image_detail": "auto",
    }
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            defaults.update(data)
    return defaults


def _get_openai_client(config: Dict[str, Any]):
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "OpenAI SDK is not installed. Install it with: pip install openai"
        ) from exc

    api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key. Set the {api_key_env} environment variable.")

    base_url = config.get("base_url") or None
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


@mcp.tool()
def decode_prescription(
    image_data: str,
    patient_id: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """Decode a base64-encoded prescription image into MedicationRequest resources.

    Expected SHARP context (Streamable HTTP headers):
    - X-Patient-ID: FHIR Patient id (regex [A-Za-z0-9-\\.]{1,64})
    - X-FHIR-Access-Token: Bearer token (with or without "Bearer " prefix)
    - X-FHIR-Server-URL: Optional, used for downstream FHIR calls
    """
    if not isinstance(image_data, str) or not image_data.strip():
        raise ValueError("image_data is required and must be a non-empty base64 string.")

    normalized = _normalize_base64(image_data)

    try:
        decoded = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_data must be valid base64-encoded image data.") from exc

    if not decoded:
        raise ValueError("image_data decoded to empty bytes.")

    patient_id_from_ctx, fhir_token, fhir_server_url = _extract_sharp_context(ctx, patient_id)
    # Placeholder for Phase 3: use fhir_token/fhir_server_url with _fhir_auth_headers(...)
    data_url = _to_data_url(image_data, decoded)

    config = _load_config()
    client = _get_openai_client(config)

    try:
        response = client.responses.create(
            model=config.get("model"),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": config.get("image_detail", "auto"),
                        },
                    ],
                }
            ],
            temperature=float(config.get("temperature", 0.2)),
            max_output_tokens=int(config.get("max_output_tokens", 512)),
        )
    except Exception as exc:
        raise RuntimeError(f"Vision API call failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("Vision API returned no text output.")

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Vision API response was not valid JSON.") from exc

    if not isinstance(parsed, list):
        raise RuntimeError("Vision API response JSON must be an array.")

    resources = []
    for item in parsed:
        if isinstance(item, dict):
            resources.append(_to_medication_request(item, patient_id_from_ctx).dict())
    return resources


def _test_decode_prescription() -> None:
    dummy_png_header = b"\x89PNG\r\n\x1a\n"
    dummy_b64 = base64.b64encode(dummy_png_header).decode("ascii")
    try:
        result = decode_prescription(dummy_b64)
        print("decode_prescription result:", result)
    except Exception as exc:
        print("decode_prescription failed:", exc)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test_decode_prescription()
    else:
        mcp.run(transport="streamable-http")
