from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

_NON_SLUG_PATTERN = re.compile(r"[^a-z0-9._-]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def canonical_json_dumps(value: object) -> str:
    """Serialize a value to a canonical JSON string."""
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: object) -> bytes:
    return canonical_json_dumps(value).encode("utf-8")


def sha256_fingerprint(value: object | str | bytes) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def parse_utc_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC timezone designator")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return parsed.astimezone(UTC)


def format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    normalized = value.astimezone(UTC)
    timespec = "microseconds" if normalized.microsecond else "seconds"
    return normalized.isoformat(timespec=timespec).replace("+00:00", "Z")


def seeded_rng(seed: int | str) -> random.Random:
    if isinstance(seed, int):
        seed_value = seed
    else:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        seed_value = int.from_bytes(digest[:8], byteorder="big")
    return random.Random(seed_value)


def stable_token(value: object, length: int = 12) -> str:
    if length <= 0:
        raise ValueError("length must be positive")
    return sha256_fingerprint(value)[:length]


def stable_id(namespace: str, *parts: object, suffix: object | None = None) -> str:
    slugged = [_slugify(namespace), *(_slugify(part) for part in parts)]
    if len(slugged) < 2:
        raise ValueError("stable_id requires a namespace and at least one part")
    if suffix is not None:
        slugged.append(stable_token(suffix))
    return ":".join(slugged)


def _json_default(value: object) -> Any:
    if isinstance(value, datetime):
        return format_utc_timestamp(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    raise TypeError(f"Object of type {type(value).__name__!r} is not JSON serializable")


def _slugify(value: object) -> str:
    normalized = _WHITESPACE_PATTERN.sub("-", str(value).strip().lower())
    normalized = _NON_SLUG_PATTERN.sub("-", normalized).strip("-")
    if not normalized:
        raise ValueError("stable ID parts must not be blank")
    return normalized
