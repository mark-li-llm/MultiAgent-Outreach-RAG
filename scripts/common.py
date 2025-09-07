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
    # Prefer og:title
    m = re.search(r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"'](.*?)[\"']", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # h1
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        return re.sub(r"\s+", " ", txt).strip()
    # title tag
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        txt = re.sub(r"<[^>]+>", " ", m.group(1))
        return re.sub(r"\s+", " ", txt).strip()
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
                async with session.get(
                    url,
                    allow_redirects=True,
                    max_redirects=max_redirects,
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                    ssl=False,
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
