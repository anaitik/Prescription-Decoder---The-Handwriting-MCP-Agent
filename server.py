import base64
import binascii
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import Context, FastMCP

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

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
TRANSLATION_PROMPT = (
    "Translate the following medical text into {language}. "
    "Preserve medication names, dosages, and instructions accurately. "
    "Do not add new medical advice. Text:\n\n{text}"
)
FHIR_ID_PATTERN = re.compile(r"^[A-Za-z0-9\-\.]{1,64}$")

mcp = FastMCP("Prescription Decoder", json_response=True)
LOGGER = logging.getLogger("prescription_mcp")

if load_dotenv:
    load_dotenv(os.getenv("MCP_DOTENV_PATH", ".env"))


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
        "aspirin": ("1191", "Aspirin"),
        "amoxicillin": ("723", "Amoxicillin"),
        "ibuprofen": ("5640", "Ibuprofen"),
        "azithromycin": ("18631", "Azithromycin"),
        "warfarin": ("11289", "Warfarin"),
        "clopidogrel": ("32968", "Clopidogrel"),
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


def _normalize_med_name(name: str) -> str:
    if not name:
        return ""
    text = name.strip().lower()
    aliases = {
        "acetylsalicylic acid": "aspirin",
        "aspirin": "aspirin",
        "warfarin": "warfarin",
        "coumadin": "warfarin",
        "ibuprofen": "ibuprofen",
        "naproxen": "naproxen",
        "clopidogrel": "clopidogrel",
        "paracetamol": "paracetamol",
        "acetaminophen": "paracetamol",
    }
    for key, value in aliases.items():
        if key in text:
            return value
    return text


def _med_name_from_request(med_request: MedicationRequest) -> str:
    med_cc = getattr(med_request, "medicationCodeableConcept", None)
    if med_cc:
        if getattr(med_cc, "text", None):
            return med_cc.text
        coding = getattr(med_cc, "coding", None) or []
        if coding:
            display = getattr(coding[0], "display", None)
            if display:
                return display
            code = getattr(coding[0], "code", None)
            if code:
                return code
    med_ref = getattr(med_request, "medicationReference", None)
    if med_ref and getattr(med_ref, "reference", None):
        return med_ref.reference
    return ""


def _fetch_patient_medications(
    patient_id: str,
    fhir_token: str,
    fhir_server_url: Optional[str],
) -> List[MedicationRequest]:
    _ = _fhir_auth_headers(fhir_token)
    # Mock implementation for development. In production, call:
    # GET {fhir_server_url}/MedicationRequest?subject=Patient/{patient_id}
    mock_items = [
        {
            "medication_name": "Warfarin",
            "dosage": "5 mg",
            "frequency": "once daily",
            "duration": "",
            "instructions": "",
        },
        {
            "medication_name": "Aspirin",
            "dosage": "75 mg",
            "frequency": "once daily",
            "duration": "",
            "instructions": "",
        },
    ]
    return [_to_medication_request(item, patient_id) for item in mock_items]


def _fetch_patient_profile(
    patient_id: str,
    fhir_token: str,
    fhir_server_url: Optional[str],
) -> Dict[str, Any]:
    config = _load_config()
    use_mock = bool(config.get("use_mock_fhir", True))
    if use_mock or not fhir_server_url:
        # Mock implementation for development.
        return {
            "patient_id": patient_id,
            "phone_number": "+919000000000",
            "preferred_language": "hi",
            "display_name": "Patient",
        }

    try:
        import requests
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("requests is required for real FHIR calls. Install: pip install requests") from exc

    url = f"{fhir_server_url.rstrip('/')}/Patient/{patient_id}"
    headers = _fhir_auth_headers(fhir_token)
    headers["Accept"] = "application/fhir+json"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        patient = response.json()
    except Exception as exc:
        # Avoid logging PHI in errors; keep message generic.
        LOGGER.warning("FHIR patient profile fetch failed.")
        raise RuntimeError("Failed to fetch patient profile from FHIR server.") from exc

    phone_number = None
    telecom = patient.get("telecom") or []
    for entry in telecom:
        if entry.get("system") == "phone" and entry.get("use") == "mobile":
            phone_number = entry.get("value")
            break
    if not phone_number:
        for entry in telecom:
            if entry.get("system") == "phone":
                phone_number = entry.get("value")
                break

    preferred_language = None
    communications = patient.get("communication") or []
    preferred_entry = next(
        (entry for entry in communications if entry.get("preferred") is True),
        None,
    )
    language_entry = preferred_entry or (communications[0] if communications else None)
    if language_entry:
        language_cc = language_entry.get("language") or {}
        coding = language_cc.get("coding") or []
        if coding:
            preferred_language = coding[0].get("code") or coding[0].get("display")
        if not preferred_language:
            preferred_language = language_cc.get("text")

    name = patient.get("name") or []
    display_name = "Patient"
    if name:
        name_obj = name[0]
        if name_obj.get("text"):
            display_name = name_obj["text"]
        else:
            given = " ".join(name_obj.get("given") or [])
            family = name_obj.get("family") or ""
            display_name = " ".join(part for part in [given, family] if part) or display_name

    return {
        "patient_id": patient_id,
        "phone_number": phone_number,
        "preferred_language": preferred_language,
        "display_name": display_name,
    }


def _find_interactions(
    new_meds: List[str],
    existing_meds: List[str],
) -> List[Dict[str, str]]:
    interaction_db: Dict[frozenset, Dict[str, str]] = {
        frozenset(["warfarin", "aspirin"]): {
            "severity": "high",
            "description": "Increased risk of bleeding when warfarin is combined with aspirin.",
            "recommendation": "Avoid combination if possible; monitor INR and bleeding closely.",
        },
        frozenset(["warfarin", "ibuprofen"]): {
            "severity": "medium",
            "description": "NSAIDs may increase bleeding risk and alter anticoagulant effect.",
            "recommendation": "Use with caution; consider alternative analgesic.",
        },
        frozenset(["clopidogrel", "aspirin"]): {
            "severity": "medium",
            "description": "Dual antiplatelet therapy increases bleeding risk.",
            "recommendation": "Ensure clinical indication and monitor for bleeding.",
        },
    }

    interactions: List[Dict[str, str]] = []
    seen_pairs = set()
    for new_med in new_meds:
        for existing_med in existing_meds:
            pair = frozenset([new_med, existing_med])
            if pair in interaction_db and pair not in seen_pairs:
                entry = interaction_db[pair]
                interactions.append(
                    {
                        "medications": ", ".join(sorted(pair)),
                        "severity": entry["severity"],
                        "description": entry["description"],
                        "recommendation": entry["recommendation"],
                    }
                )
                seen_pairs.add(pair)

    return interactions


def _fetch_patient_allergies(
    patient_id: str,
    fhir_token: str,
    fhir_server_url: Optional[str],
) -> List[Dict[str, Any]]:
    _ = _fhir_auth_headers(fhir_token)
    # Mock implementation for development. In production, call:
    # GET {fhir_server_url}/AllergyIntolerance?patient=Patient/{patient_id}
    return [
        {"substance": "Penicillin", "criticality": "high", "reaction": "Rash"},
        {"substance": "Sulfa", "criticality": "high", "reaction": "Anaphylaxis"},
    ]


def _med_class_from_name(name: str) -> Optional[str]:
    text = _normalize_med_name(name)
    beta_lactams = [
        "penicillin",
        "amoxicillin",
        "ampicillin",
        "cephalexin",
        "ceftriaxone",
        "cefixime",
        "cephalosporin",
    ]
    sulfonamides = [
        "sulfa",
        "sulfamethoxazole",
        "sulfonamide",
        "co-trimoxazole",
        "trimethoprim-sulfamethoxazole",
    ]
    if any(item in text for item in beta_lactams):
        return "beta-lactam"
    if any(item in text for item in sulfonamides):
        return "sulfonamide"
    if "aspirin" in text or "ibuprofen" in text or "naproxen" in text:
        return "nsaid"
    return None


def _normalize_allergy_name(name: str) -> str:
    if not name:
        return ""
    text = name.strip().lower()
    aliases = {
        "penicillin": "penicillin",
        "pcn": "penicillin",
        "beta-lactam": "beta-lactam",
        "cephalosporin": "beta-lactam",
        "sulfa": "sulfonamide",
        "sulfonamide": "sulfonamide",
        "nsaid": "nsaid",
        "aspirin": "nsaid",
    }
    for key, value in aliases.items():
        if key in text:
            return value
    return text


def _check_allergy_matches(
    new_meds: List[str],
    allergies: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    normalized_allergies = [
        _normalize_allergy_name(str(item.get("substance") or "")) for item in allergies
    ]

    for med in new_meds:
        med_name = _normalize_med_name(med)
        med_class = _med_class_from_name(med_name)
        for allergy_raw, allergy_norm in zip(allergies, normalized_allergies):
            allergy_name = allergy_norm
            if not allergy_name:
                continue
            direct_match = allergy_name in med_name or med_name in allergy_name
            cross_reactive = False
            if allergy_name == "penicillin" and med_class == "beta-lactam":
                cross_reactive = True
            if allergy_name == "beta-lactam" and med_class == "beta-lactam":
                cross_reactive = True
            if allergy_name == "sulfonamide" and med_class == "sulfonamide":
                cross_reactive = True
            if allergy_name == "nsaid" and med_class == "nsaid":
                cross_reactive = True

            if direct_match or cross_reactive:
                warnings.append(
                    {
                        "medication": med,
                        "allergy": str(allergy_raw.get("substance") or allergy_name),
                        "criticality": str(allergy_raw.get("criticality") or "unknown"),
                        "reaction": str(allergy_raw.get("reaction") or "unspecified"),
                        "note": "Direct match" if direct_match else "Possible cross-reactivity",
                    }
                )
    return warnings


def _load_config() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "model": "gpt-4.1-mini",
        "translation_model": "gpt-4.1-mini",
        "groq_api_key_env": "GROQ_API_KEY",
        "groq_base_url": "https://api.groq.com/openai/v1",
        "groq_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "groq_translation_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_output_tokens": 512,
        "temperature": 0.2,
        "image_detail": "auto",
        "default_language": "hi",
        "prescription_view_url": "https://example.com/prescriptions",
        "twilio_account_sid_env": "TWILIO_ACCOUNT_SID",
        "twilio_auth_token_env": "TWILIO_AUTH_TOKEN",
        "twilio_from_number": "",
        "use_mock_fhir": True,
        "use_mock_vision": False,
        "openai_max_retries": 2,
        "openai_retry_backoff_seconds": 1.0,
    }
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            defaults.update(data)
    return defaults


def _resolve_provider_config(config: Dict[str, Any]) -> Dict[str, Any]:
    provider = str(config.get("provider") or "openai").strip().lower()
    resolved = dict(config)
    if provider == "groq":
        resolved["api_key_env"] = resolved.get("groq_api_key_env") or "GROQ_API_KEY"
        if not resolved.get("base_url"):
            resolved["base_url"] = resolved.get("groq_base_url") or "https://api.groq.com/openai/v1"
        if resolved.get("model") in (None, "", "gpt-4.1-mini"):
            resolved["model"] = resolved.get("groq_model") or "meta-llama/llama-4-scout-17b-16e-instruct"
        if resolved.get("translation_model") in (None, ""):
            resolved["translation_model"] = (
                resolved.get("groq_translation_model") or resolved["model"]
            )
    return resolved


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


def _is_insufficient_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


def _call_openai_with_retries(client, *, max_retries: int, backoff_seconds: float, **kwargs):
    attempt = 0
    while True:
        try:
            return client.responses.create(**kwargs)
        except Exception as exc:
            if _is_insufficient_quota_error(exc):
                raise RuntimeError(
                    "LLM quota exceeded. Check your plan/billing or set use_mock_vision=true."
                ) from exc
            if attempt >= max_retries:
                raise
            sleep_for = backoff_seconds * (2 ** attempt)
            time.sleep(sleep_for)
            attempt += 1


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

    config = _resolve_provider_config(_load_config())
    if config.get("use_mock_vision"):
        mock_items = [
            {
                "medication_name": "Paracetamol",
                "dosage": "500 mg",
                "frequency": "twice daily",
                "duration": "3 days",
                "instructions": "After meals",
            },
            {
                "medication_name": "Amoxicillin",
                "dosage": "250 mg",
                "frequency": "three times daily",
                "duration": "5 days",
                "instructions": "",
            },
        ]
        resources = [
            _to_medication_request(item, patient_id_from_ctx).dict()
            for item in mock_items
        ]
        return resources

    client = _get_openai_client(config)

    try:
        response = _call_openai_with_retries(
            client,
            max_retries=int(config.get("openai_max_retries", 2)),
            backoff_seconds=float(config.get("openai_retry_backoff_seconds", 1.0)),
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


@mcp.tool()
def check_drug_interactions(
    medications: List[Dict[str, Any]],
    patient_id: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Check for interactions between new medications and a patient's current meds.

    Expected SHARP context (Streamable HTTP headers):
    - X-Patient-ID: FHIR Patient id (regex [A-Za-z0-9-\\.]{1,64})
    - X-FHIR-Access-Token: Bearer token (with or without "Bearer " prefix)
    - X-FHIR-Server-URL: Optional, used for downstream FHIR calls

    Note: The interaction database is mocked. In production, connect to a real
    interaction DB such as RxNav or Drugs.com APIs.
    """
    if not isinstance(medications, list) or not medications:
        raise ValueError("medications must be a non-empty list of MedicationRequest objects.")

    patient_id_from_ctx, fhir_token, fhir_server_url = _extract_sharp_context(ctx, patient_id)

    new_requests: List[MedicationRequest] = []
    for idx, item in enumerate(medications):
        if not isinstance(item, dict):
            raise ValueError(f"MedicationRequest at index {idx} must be an object.")
        try:
            new_requests.append(MedicationRequest(**item))
        except Exception as exc:
            raise ValueError(f"Invalid MedicationRequest at index {idx}: {exc}") from exc

    existing_requests = _fetch_patient_medications(
        patient_id=patient_id_from_ctx,
        fhir_token=fhir_token,
        fhir_server_url=fhir_server_url,
    )

    new_med_names = [
        _normalize_med_name(_med_name_from_request(req)) for req in new_requests
    ]
    existing_med_names = [
        _normalize_med_name(_med_name_from_request(req)) for req in existing_requests
    ]

    new_med_names = [name for name in new_med_names if name]
    existing_med_names = [name for name in existing_med_names if name]

    interactions = _find_interactions(new_med_names, existing_med_names)

    return {
        "patient_id": patient_id_from_ctx,
        "interaction_count": len(interactions),
        "interactions": interactions,
        "checked_new_medications": new_med_names,
        "checked_existing_medications": existing_med_names,
        "notes": [
            "Interaction checks use a mocked database for development.",
            "Replace with a real drug interaction source (e.g., RxNav or Drugs.com) in production.",
        ],
    }


@mcp.tool()
def check_allergies(
    new_medications: List[Any],
    patient_id: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Check new medications against patient allergies (AllergyIntolerance).

    Expected SHARP context (Streamable HTTP headers):
    - X-Patient-ID: FHIR Patient id (regex [A-Za-z0-9-\\.]{1,64})
    - X-FHIR-Access-Token: Bearer token (with or without "Bearer " prefix)
    - X-FHIR-Server-URL: Optional, used for downstream FHIR calls

    Note: Uses mocked allergy data for development.
    """
    if not isinstance(new_medications, list) or not new_medications:
        raise ValueError("new_medications must be a non-empty list.")

    patient_id_from_ctx, fhir_token, fhir_server_url = _extract_sharp_context(ctx, patient_id)

    med_names: List[str] = []
    for idx, item in enumerate(new_medications):
        if isinstance(item, str):
            med_names.append(item)
        elif isinstance(item, dict):
            # Accept MedicationRequest dict or generic medication object
            try:
                req = MedicationRequest(**item)
                med_names.append(_med_name_from_request(req))
            except Exception:
                med_names.append(str(item.get("medication_name") or item.get("name") or ""))
        else:
            raise ValueError(f"Medication entry at index {idx} is not supported.")

    med_names = [name for name in med_names if name]
    allergies = _fetch_patient_allergies(
        patient_id=patient_id_from_ctx,
        fhir_token=fhir_token,
        fhir_server_url=fhir_server_url,
    )
    warnings = _check_allergy_matches(med_names, allergies)

    return {
        "patient_id": patient_id_from_ctx,
        "warning_count": len(warnings),
        "warnings": warnings,
        "checked_new_medications": med_names,
        "allergies_checked": allergies,
        "notes": [
            "Allergy checks use mocked FHIR data for development.",
            "Cross-reactivity rules are simplified (e.g., penicillin -> beta-lactams).",
        ],
    }


@mcp.tool()
def translate_to_hindi(text: str, target_language: str = "hi") -> Dict[str, str]:
    """Translate English medical text into Hindi or another Indian language.

    target_language can be a language code like "hi" or a language name like "Hindi".
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    config = _resolve_provider_config(_load_config())
    client = _get_openai_client(config)

    language = target_language or config.get("default_language", "hi")
    prompt = TRANSLATION_PROMPT.format(language=language, text=text.strip())

    try:
        response = client.responses.create(
            model=config.get("translation_model") or config.get("model"),
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            temperature=0.2,
            max_output_tokens=512,
        )
    except Exception as exc:
        raise RuntimeError(f"Translation API call failed: {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("Translation API returned no text output.")

    return {"language": language, "translated_text": output_text.strip()}


@mcp.tool()
def send_sms(phone_number: str, message: str) -> Dict[str, str]:
    """Send an SMS to a patient using Twilio (or compatible provider)."""
    if not phone_number or not isinstance(phone_number, str):
        raise ValueError("phone_number must be a non-empty string.")
    if not message or not isinstance(message, str):
        raise ValueError("message must be a non-empty string.")

    config = _load_config()
    account_sid_env = config.get("twilio_account_sid_env") or "TWILIO_ACCOUNT_SID"
    auth_token_env = config.get("twilio_auth_token_env") or "TWILIO_AUTH_TOKEN"
    from_number = config.get("twilio_from_number") or ""

    account_sid = os.getenv(account_sid_env)
    auth_token = os.getenv(auth_token_env)
    if not account_sid or not auth_token:
        raise ValueError(
            f"Missing Twilio credentials. Set {account_sid_env} and {auth_token_env}."
        )
    if not from_number:
        raise ValueError("Missing Twilio from number in config.json (twilio_from_number).")

    try:
        from twilio.rest import Client
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Twilio SDK is not installed. Install it with: pip install twilio"
        ) from exc

    try:
        client = Client(account_sid, auth_token)
        message_obj = client.messages.create(
            body=message,
            from_=from_number,
            to=phone_number,
        )
    except Exception as exc:
        raise RuntimeError(f"SMS send failed: {exc}") from exc

    return {"status": "sent", "sid": getattr(message_obj, "sid", ""), "to": phone_number}


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
