import base64
import binascii
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Prescription Decoder", json_response=True)


def _normalize_base64(image_data: str) -> str:
    data = image_data.strip()
    if data.startswith("data:"):
        parts = data.split(",", 1)
        if len(parts) == 2:
            data = parts[1]
    return data


@mcp.tool()
def decode_prescription(image_data: str) -> dict:
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

    # Mock response for Phase 1
    return {"medications": ["Paracetamol 500mg", "Amoxicillin 250mg"]}


def _test_decode_prescription() -> None:
    dummy_png_header = b"\x89PNG\r\n\x1a\n"
    dummy_b64 = base64.b64encode(dummy_png_header).decode("ascii")
    result = decode_prescription(dummy_b64)
    print("decode_prescription result:", result)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test_decode_prescription()
    else:
        mcp.run(transport="streamable-http")
