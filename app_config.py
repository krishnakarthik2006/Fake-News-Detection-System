from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_USER_AGENT = "FakeNewsDetector/4.0 (explainable-verification)"
DEFAULT_REQUEST_TIMEOUT = 2.5
DEFAULT_FACT_CHECK_PAGE_SIZE = 4
DEFAULT_WIKIPEDIA_PAGE_SIZE = 3
DEFAULT_WIKIDATA_PAGE_SIZE = 3
DEFAULT_MAX_CLAIMS = 5
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return max(minimum, int(raw_value))
    except ValueError:
        return default


def _read_float_env(name: str, default: float, minimum: float = 0.1) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return max(minimum, float(raw_value))
    except ValueError:
        return default


@dataclass(frozen=True)
class RuntimeConfig:
    google_fact_check_api_key: str
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    fact_check_page_size: int = DEFAULT_FACT_CHECK_PAGE_SIZE
    wikipedia_page_size: int = DEFAULT_WIKIPEDIA_PAGE_SIZE
    wikidata_page_size: int = DEFAULT_WIKIDATA_PAGE_SIZE
    max_claims: int = DEFAULT_MAX_CLAIMS
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    user_agent: str = DEFAULT_USER_AGENT


def get_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        google_fact_check_api_key=os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip(),
        request_timeout=_read_float_env("FND_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
        fact_check_page_size=_read_int_env("FND_FACT_CHECK_PAGE_SIZE", DEFAULT_FACT_CHECK_PAGE_SIZE),
        wikipedia_page_size=_read_int_env("FND_WIKIPEDIA_PAGE_SIZE", DEFAULT_WIKIPEDIA_PAGE_SIZE),
        wikidata_page_size=_read_int_env("FND_WIKIDATA_PAGE_SIZE", DEFAULT_WIKIDATA_PAGE_SIZE),
        max_claims=_read_int_env("FND_MAX_CLAIMS", DEFAULT_MAX_CLAIMS),
        host=os.getenv("FND_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
        port=_read_int_env("FND_PORT", DEFAULT_PORT),
        user_agent=os.getenv("FND_USER_AGENT", DEFAULT_USER_AGENT).strip() or DEFAULT_USER_AGENT,
    )
