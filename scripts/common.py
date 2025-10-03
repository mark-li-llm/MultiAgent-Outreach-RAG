import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import ssl
import certifi
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


USER_AGENT = os.environ.get("AR_USER_AGENT", "AccountResearchMVP/1.0")
GLOBAL_RPS = float(os.environ.get("AR_GLOBAL_RPS", "6"))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1_8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def slugify(text: str, max_len: int = 80) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t).strip("-")
    return t[:max_len]


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "igshid",
}


def strip_tracking_params(url: str) -> str:
    try:
        parsed = urlparse(url)
        q = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k not in TRACKING_PARAMS and not k.startswith("utm_")]
        new_query = urlencode(q)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
    except Exception:
        return url


def domain_of(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def extract_title(html: str) -> Optional[str]:
    """Extract title from HTML with improved og:title parsing"""

    # 1. Prefer og:title - multiple formats
    og_patterns = [
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']*)["\']',
        r'<meta\s+content=["\']([^"\']*?)["\']\s+property=["\']og:title["\']',
        r'<meta[^>]+property\s*=\s*["\']og:title["\']\s*content\s*=\s*["\']([^"\']*)["\']',
        r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"'](.*?)[\"']",
    ]

    for pattern in og_patterns:
        m = re.search(pattern, html, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()

    # 2. Twitter title
    m = re.search(r'<meta[^>]+name=["\']twitter:title["\']\s+content=["\']([^"\']*)["\']', html, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # 3. h1 tag
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        title = re.sub(r"\s+", " ", txt).strip()
        if title:
            return title

    # 4. title tag (fallback)
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        title = re.sub(r"\s+", " ", txt).strip()
        if title:
            return title

    return None


def try_parse_date_from_meta(html: str) -> Optional[str]:
    # Try common meta tags for published time
    patterns = [
        r"<meta[^>]+property=[\"']article:published_time[\"'][^>]+content=[\"'](.*?)[\"']",
        r"<meta[^>]+name=[\"']pubdate[\"'][^>]+content=[\"'](.*?)[\"']",
        r"<meta[^>]+name=[\"']date[\"'][^>]+content=[\"'](.*?)[\"']",
        r"<meta[^>]+itemprop=[\"']datePublished[\"'][^>]+content=[\"'](.*?)[\"']",
    ]
    for p in patterns:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            d = coerce_date(m.group(1).strip())
            if d:
                return d
    # <time datetime="...">
    m = re.search(r"<time[^>]+datetime=\"(.*?)\"", html, re.IGNORECASE)
    if m:
        d = coerce_date(m.group(1).strip())
        if d:
            return d
    # Attempt to find dates like Month DD, YYYY
    m = re.search(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", html)
    if m:
        d = coerce_date(m.group(1))
        if d:
            return d
    return None


def coerce_date(s: str) -> Optional[str]:
    # Return YYYY-MM-DD if possible
    s = s.strip()
    # RFC 822 via email.utils
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(s)
        if dt:
            return dt.date().isoformat()
    except Exception:
        pass
    # ISO / common formats
    for fmt in [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%b %d, %Y",
        "%B %d, %Y",
        "%m/%d/%Y",
    ]:
        try:
            from datetime import datetime

            dt = datetime.strptime(s, fmt)
            return dt.date().isoformat()
        except Exception:
            continue
    # Try partial ISO without timezone
    try:
        # fromisoformat may handle Z offsets in modern Python
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return None


class RateLimiter:
    def __init__(self, rps: float = GLOBAL_RPS):
        self.rps = rps
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
            self._next_time = max(now, self._next_time) + 1.0 / self.rps


def build_logger() -> Tuple[logging.Logger, str]:
    ensure_dir("logs/fetch")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", "fetch", f"{ts}.log")
    logger = logging.getLogger(f"fetch_{ts}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger, log_path


@dataclass
class FetchResult:
    status: int
    final_url: str
    redirect_chain: List[str]
    content_type: str
    body: Optional[bytes]
    latency_ms: float
    err: Optional[str] = None


async def fetch_with_retries(
    session: aiohttp.ClientSession,
    limiter: RateLimiter,
    url: str,
    max_redirects: int = 5,
    logger: Optional[logging.Logger] = None,
    timeout_s: int = 30,
) -> FetchResult:
    await limiter.wait()
    url = strip_tracking_params(url)
    backoffs = [1, 2, 4]
    last_err = None
    redirect_chain: List[str] = []
    t0 = time.perf_counter()
    try:
        for attempt, back in enumerate([0] + backoffs):
            if back > 0:
                await asyncio.sleep(back)
            try:
                t_req = time.perf_counter()
                from urllib.parse import urlparse as _urlparse
                host = _urlparse(url).netloc.lower()
                # Use SSL verification via certifi for most hosts; disable only for sec.gov
                if host.endswith("sec.gov"):
                    ssl_opt = False
                else:
                    try:
                        ssl_opt = ssl.create_default_context(cafile=certifi.where())
                    except Exception:
                        ssl_opt = None
                async with session.get(
                    url,
                    allow_redirects=True,
                    max_redirects=max_redirects,
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                    ssl=ssl_opt,
                ) as resp:
                    history = [strip_tracking_params(str(h.url)) for h in resp.history]
                    final_url = strip_tracking_params(str(resp.url))
                    body = await resp.read() if resp.status == 200 else None
                    ct = resp.headers.get("Content-Type", "")
                    latency_ms = (time.perf_counter() - t_req) * 1000.0
                    return FetchResult(
                        status=resp.status,
                        final_url=final_url,
                        redirect_chain=history,
                        content_type=ct,
                        body=body,
                        latency_ms=latency_ms,
                        err=None,
                    )
            except aiohttp.ClientResponseError as e:
                last_err = f"ClientResponseError: {e}"
                if e.status in (429,) or 500 <= e.status < 600:
                    continue
                else:
                    raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return FetchResult(status=0, final_url=url, redirect_chain=redirect_chain, content_type="", body=None, latency_ms=latency_ms, err=last_err)
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if logger:
            logger.error(f"Fetch failed: {url} -> {e}")
        return FetchResult(status=0, final_url=url, redirect_chain=redirect_chain, content_type="", body=None, latency_ms=latency_ms, err=str(e))


def write_bytes(path: str, data: bytes) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_exists(path: str) -> bool:
    return os.path.exists(path)


def default_session_headers() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "*/*"}


def build_doc_id(doctype: str, date_str: Optional[str], slug_base: str, url_for_hash: str) -> str:
    date_part = date_str or "unknown"
    slug = slugify(slug_base or "document")
    tail = sha1_8(strip_tracking_params(url_for_hash))
    return f"crm::{doctype}::{date_part}::{slug}::{tail}"


# ===== MCP Connection Manager =====

from enum import Enum


class FallbackMode(Enum):
    """Three-mode system for MCP service fallback behavior."""
    DEFAULT = "default"  # Silent fallback (production)
    WARN = "warn"        # Log downgrades, mark as WARN
    STRICT = "strict"    # Fail fast, no fallback


@dataclass
class DowngradeEvent:
    """Record of a service downgrade event."""
    from_service: str    # "internal_stub" | "external" | "online"
    to_service: str      # "external" | "offline"
    reason: str
    timestamp: str
    exception_type: str


class MCPConnectionManager:
    """
    Manages MCP service connections with three-mode fallback support.

    Modes:
    - DEFAULT: Silent fallback (internal_stub → external → offline)
    - WARN: Fallback allowed but logged, triggers WARN status
    - STRICT: No fallback, fail immediately if service unavailable
    """

    def __init__(self, config: Dict[str, Any], mode: FallbackMode):
        self.config = config
        self.mode = mode
        self.downgrades: List[DowngradeEvent] = []
        self.service_type: Optional[str] = None

    async def connect(
        self,
        start_stub_fn,
        test_external_fn,
        setup_offline_fn
    ) -> Tuple[str, bool, List[DowngradeEvent]]:
        """
        Connect to MCP service with fallback logic.

        Args:
            start_stub_fn: Async function to start internal stub servers
            test_external_fn: Async function to test external service
            setup_offline_fn: Async function to setup offline mode

        Returns:
            Tuple of (service_type, use_offline, downgrade_events)
            - service_type: "internal_stub" | "external" | "offline"
            - use_offline: bool
            - downgrade_events: List of DowngradeEvent

        Raises:
            RuntimeError: In STRICT mode if service unavailable
        """
        if self.mode == FallbackMode.STRICT:
            return await self._connect_strict(start_stub_fn)
        elif self.mode == FallbackMode.WARN:
            return await self._connect_warn(start_stub_fn, test_external_fn, setup_offline_fn)
        else:
            return await self._connect_default(start_stub_fn, test_external_fn, setup_offline_fn)

    async def _connect_strict(self, start_stub_fn) -> Tuple[str, bool, List[DowngradeEvent]]:
        """Strict mode: fail fast, no fallback."""
        try:
            await start_stub_fn()
            self.service_type = "internal_stub"
            return "internal_stub", False, []
        except Exception as e:
            raise RuntimeError(f"MCP service unavailable in strict mode: {type(e).__name__}: {e}") from e

    async def _connect_warn(
        self,
        start_stub_fn,
        test_external_fn,
        setup_offline_fn
    ) -> Tuple[str, bool, List[DowngradeEvent]]:
        """Warning mode: fallback allowed but logged."""
        # Try internal stub
        try:
            await start_stub_fn()
            self.service_type = "internal_stub"
            return "internal_stub", False, []
        except Exception as e:
            self.downgrades.append(DowngradeEvent(
                from_service="internal_stub",
                to_service="external",
                reason=str(e),
                timestamp=now_iso(),
                exception_type=type(e).__name__
            ))

        # Try external service
        try:
            await test_external_fn()
            self.service_type = "external"
            return "external", False, self.downgrades
        except Exception as e:
            self.downgrades.append(DowngradeEvent(
                from_service="external",
                to_service="offline",
                reason=str(e),
                timestamp=now_iso(),
                exception_type=type(e).__name__
            ))

        # Fall back to offline
        await setup_offline_fn()
        self.service_type = "offline"
        return "offline", True, self.downgrades

    async def _connect_default(
        self,
        start_stub_fn,
        test_external_fn,
        setup_offline_fn
    ) -> Tuple[str, bool, List[DowngradeEvent]]:
        """Default mode: silent fallback (backward compatible)."""
        # Try internal stub
        try:
            await start_stub_fn()
            self.service_type = "internal_stub"
            return "internal_stub", False, []
        except Exception:
            pass

        # Try external service
        try:
            await test_external_fn()
            self.service_type = "external"
            return "external", False, []
        except Exception:
            pass

        # Fall back to offline
        await setup_offline_fn()
        self.service_type = "offline"
        return "offline", True, []


def load_fallback_mode(config: Dict[str, Any]) -> FallbackMode:
    """
    Load fallback mode from config or environment.
    Environment variable AG_MCP_FALLBACK_MODE takes precedence.
    """
    mode_str = os.getenv("AG_MCP_FALLBACK_MODE") or (config.get("fallback") or {}).get("mode") or "default"
    try:
        return FallbackMode(mode_str.lower())
    except ValueError:
        print(f"⚠ Invalid fallback mode '{mode_str}', using 'default'")
        return FallbackMode.DEFAULT
