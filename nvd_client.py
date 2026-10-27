"""UI-independent NVD API v2 client, with replayable raw-response snapshots.

Policy: CVSS v3.1 only; prefer Primary, then nvd@nist.gov, then source and
vector lexical order. No silent v3.0/v4 conversion. NVD source documentation:
https://nvd.nist.gov/developers/vulnerabilities
https://nvd.nist.gov/developers/start-here
Snapshots are cached indefinitely until refresh=True; experiments use offline
snapshots, not an implicitly changing live response. Never caches API keys.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
SOURCE_POLICY = "cvss31-primary-then-nvd-source-lexical-v1"
_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}\Z")
_RATE_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def normalize_nvd_response(payload: dict, cve_id: str) -> dict[str, Any]:
    """Normalize one exact CVE; retain configuration, metric and source provenance."""
    if not _CVE_PATTERN.fullmatch(cve_id):
        raise ValueError("Invalid CVE ID; expected literal CVE-YYYY-NNNN format.")
    entries = payload.get("vulnerabilities", [])
    cve = next((item.get("cve", {}) for item in entries
                if item.get("cve", {}).get("id") == cve_id), None)
    if cve is None:
        raise ValueError("Requested CVE not found in NVD response/snapshot.")
    metrics = cve.get("metrics", {})
    candidates = [m for m in metrics.get("cvssMetricV31", [])
                  if m.get("cvssData", {}).get("version") == "3.1"
                  and str(m.get("cvssData", {}).get("vectorString", "")).startswith("CVSS:3.1/")]
    candidates.sort(key=lambda m: (m.get("type") != "Primary",
                                  m.get("source") != "nvd@nist.gov",
                                  m.get("source", ""),
                                  m.get("cvssData", {}).get("vectorString", "")))
    selected = candidates[0] if candidates else {}
    data = selected.get("cvssData", {})
    versions = sorted({str(m.get("cvssData", {}).get("version"))
                       for group in metrics.values() if isinstance(group, list)
                       for m in group if isinstance(m, dict) and m.get("cvssData", {}).get("version")})
    return {
        "cve_id": cve_id,
        "description": next((d["value"] for d in cve.get("descriptions", [])
                             if d.get("lang") == "en"), "Description not provided by NVD."),
        "base_score": data.get("baseScore"),
        "vector_string": data.get("vectorString", ""),
        "configurations": cve.get("configurations", []),
        "cvss_version": "3.1" if selected else None,
        "cvss_status": "supported" if selected else ("unsupported_version" if versions else "missing"),
        "available_cvss_versions": versions,
        "nvd_cvss_source": selected.get("source"),
        "cvss_metric_type": selected.get("type"),
        "metric_selection_policy": SOURCE_POLICY,
        "cvss_candidates": metrics.get("cvssMetricV31", []),
        "published": cve.get("published"), "last_modified": cve.get("lastModified"),
        "vuln_status": cve.get("vulnStatus"),
        "vector_origin": "nvd", "source_url": NVD_URL + "?cveId=" + cve_id,
    }


def load_nvd_snapshot(path: str | Path, cve_id: str) -> dict:
    """Read a raw NVD response or this client's timestamped snapshot envelope."""
    raw_bytes = Path(path).read_bytes()
    stored = json.loads(raw_bytes)
    payload = stored.get("response", stored)
    snapshot_id = _digest(payload)
    if "response" in stored and stored.get("snapshot_id") != snapshot_id:
        raise ValueError("NVD snapshot hash mismatch.")
    data = normalize_nvd_response(payload, cve_id)
    data.update(nvd_snapshot_id=snapshot_id, nvd_snapshot_path=str(path),
                nvd_retrieved_at=stored.get("retrieved_at"),
                nvd_snapshot_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                nvd_cache_hit=True, nvd_retry_count=0)
    return data


def fetch_cve_from_nvd(cve_id: str, *, cache_dir: str | Path = ".cache/nvd",
                       offline: bool = False, refresh: bool = False,
                       api_key: str = "", max_retries: int = 3,
                       timeout: float = 30, min_interval: float | None = None,
                       session=None, sleep=time.sleep) -> tuple[dict | None, str]:
    """Fetch/cache one CVE, or replay offline. Message never includes credentials.

    Retries timeouts/connections and 429/5xx only; bounded exponential backoff
    honors numeric Retry-After up to 120s. Process-local serialized pacing
    defaults to 6s without a key / 0.6s with a key. No live fetch on corrupt
    cached snapshots unless the caller explicitly requests refresh.
    """
    global _LAST_REQUEST
    if not isinstance(cve_id, str) or not _CVE_PATTERN.fullmatch(cve_id):
        return None, "Invalid CVE ID; expected literal CVE-YYYY-NNNN format."
    if max_retries < 0:
        return None, "max_retries must be nonnegative."
    target = Path(cache_dir) / (cve_id + ".json")
    if target.exists() and (offline or not refresh):
        try:
            return load_nvd_snapshot(target, cve_id), "Loaded NVD snapshot."
        except (ValueError, OSError, TypeError, KeyError):
            return None, "Invalid or unreadable NVD snapshot; explicit refresh required."
    if offline:
        return None, "NVD snapshot unavailable in offline mode."
    interval = (0.6 if api_key else 6.0) if min_interval is None else max(0, min_interval)
    client = session or requests
    for attempt in range(max_retries + 1):
        retry_after = None
        try:
            with _RATE_LOCK:
                delay = interval - (time.monotonic() - _LAST_REQUEST)
                if delay > 0:
                    sleep(delay)
                _LAST_REQUEST = time.monotonic()
                response = client.get(NVD_URL, params={"cveId": cve_id},
                                      headers={"apiKey": api_key} if api_key else {}, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                try:
                    retry_after = max(0.0, min(120.0, float(response.headers.get("Retry-After"))))
                except (TypeError, ValueError):
                    pass
            elif response.status_code != 200:
                return None, f"NVD HTTP {response.status_code}; no retry."
            else:
                payload = response.json()
                data = normalize_nvd_response(payload, cve_id)
                envelope = {"snapshot_id": _digest(payload),
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "source_url": data["source_url"], "response": payload}
                target.parent.mkdir(parents=True, exist_ok=True)
                # Atomic replacement avoids a partially written cached response.
                import os
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                                 prefix=target.name + ".", delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(envelope, handle, ensure_ascii=False, indent=2)
                try:
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                data = load_nvd_snapshot(target, cve_id)
                data.update(nvd_cache_hit=False, nvd_retry_count=attempt)
                return data, "Fetched and snapshotted NVD response."
        except (requests.Timeout, requests.ConnectionError):
            pass
        except (ValueError, TypeError, KeyError):
            return None, "NVD returned an invalid response or the requested CVE was not found."
        except requests.RequestException:
            return None, "NVD request failed."
        except OSError:
            return None, "Unable to persist NVD snapshot."
        if attempt < max_retries:
            sleep(retry_after if retry_after is not None else min(2 ** attempt, 30))
    return None, f"NVD request failed after {max_retries + 1} attempts."
