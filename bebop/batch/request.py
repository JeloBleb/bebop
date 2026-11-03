"""The Batch request line and its inverse: the interactive read spelled in
Google's native generateContent shape, so the batched sample matches the cache key."""

from __future__ import annotations

import base64
import json

from .. import config
from ..stages import read as reader

# Issue #97's measured figure: ~2.4c per table read at Gemini 3.7-flash
# interactive; Batch Mode bills ~50% of interactive. Estimation only —
# the cache entry records the real token counts per response.
INTERACTIVE_USD_PER_TABLE = 0.024
BATCH_DISCOUNT = 0.5


def estimated_cost_usd(n_tables: int) -> float:
    return round(n_tables * INTERACTIVE_USD_PER_TABLE * BATCH_DISCOUNT, 4)


def generate_content_request(png: bytes) -> dict:
    """One GenerateContentRequest body for one crop."""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.standard_b64encode(png).decode(),
                        }
                    },
                    {"text": reader.PROMPT},
                ],
            }
        ],
        "generationConfig": {
            "temperature": config.READER_TEMPERATURE,
            "maxOutputTokens": config.READER_MAX_TOKENS,
        },
    }


def jsonl_line(key: str, png: bytes) -> str:
    """One line of the Batch input file: {"key", "request"}. The key is
    the crop's image sha — the request is identified by exactly what the
    cache keys on, so two documents sharing a crop share one request."""
    return json.dumps({"key": key, "request": generate_content_request(png)},
                      separators=(",", ":"))


def wire_model() -> str:
    """Model id in the batch URL — the reader's own wire spelling for the
    Google host (bare, prefix stripped), so the batch reads with the model
    the cache dir names."""
    return config.READER_MODEL.rpartition("/")[2]


def response_text(response: dict) -> tuple[str | None, dict]:
    """A GenerateContentResponse -> (text, usage) shaped like the reader's
    transport return: text None when no candidate text came back (a
    blocked or empty candidate), usage as the reader's _usage() dict."""
    meta = response.get("usageMetadata") or {}
    usage = {
        "cost": 0.0,
        "prompt_tokens": int(meta.get("promptTokenCount", 0) or 0),
        "completion_tokens": int(meta.get("candidatesTokenCount", 0) or 0),
    }
    candidates = response.get("candidates") or []
    if not candidates:
        return None, usage
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought")]
    if not texts:
        return None, usage
    return "".join(texts), usage
