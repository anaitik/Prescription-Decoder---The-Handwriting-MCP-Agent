import base64
import binascii
import json
import os
import sys
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config.json"))
PROMPT = (
    "Extract all medications, dosages, and instructions from this doctor's prescription. "
    "Format the output as a JSON array with fields: medication_name, dosage, frequency, "
    "duration, and instructions. If something is unclear, make your best guess and note "
    "it with a 'confidence: low' field."
)

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
def decode_prescription(image_data: str) -> List[Dict[str, Any]]:
    """Decode a base64-encoded prescription image into medication text."""
    if not isinstance(image_data, str) or not image_data.strip():
        raise ValueError("image_data is required and must be a non-empty base64 string.")

    normalized = _normalize_base64(image_data)

    try:
        decoded = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_data must be valid base64-encoded image data.") from exc

    if not decoded:
        raise ValueError("image_data decoded to empty bytes.")

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

    return parsed


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
