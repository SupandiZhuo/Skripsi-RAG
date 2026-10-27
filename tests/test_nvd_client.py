"""Offline contract fixtures, not empirical vulnerability research data."""
import json
from unittest.mock import Mock

import pytest
import requests

from nvd_client import fetch_cve_from_nvd, normalize_nvd_response

VECTOR = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


def payload(metrics=None):
    return {"vulnerabilities": [{"cve": {"id": "CVE-2024-1234", "descriptions": [
        {"lang": "en", "value": "Unit-test description"}], "metrics": metrics or {}}}]}


def metric(source="nvd@nist.gov", kind="Primary", version="3.1"):
    return {"source": source, "type": kind, "cvssData": {
        "version": version, "vectorString": VECTOR.replace("3.1", version), "baseScore": 9.8}}


def test_primary_then_nvd_source_selection():
    data = payload({"cvssMetricV31": [metric("other", "Secondary"), metric()]})
    result = normalize_nvd_response(data, "CVE-2024-1234")
    assert result["nvd_cvss_source"] == "nvd@nist.gov"
    assert result["cvss_metric_type"] == "Primary"
    assert result["vector_string"] == VECTOR


def test_v30_is_explicitly_unsupported_without_silent_fallback():
    result = normalize_nvd_response(payload({"cvssMetricV30": [metric(version="3.0")]}), "CVE-2024-1234")
    assert result["vector_string"] == ""
    assert result["base_score"] is None
    assert result["cvss_status"] == "unsupported_version"
    assert result["available_cvss_versions"] == ["3.0"]
    assert result["configurations"] == []


def test_missing_score_is_explicit():
    result = normalize_nvd_response(payload(), "CVE-2024-1234")
    assert result["cvss_status"] == "missing"


def test_retry_and_snapshot_replay(tmp_path):
    response = Mock(status_code=200, headers={})
    response.json.return_value = payload({"cvssMetricV31": [metric()]})
    throttled = Mock(status_code=429, headers={"Retry-After": "0"})
    session = Mock()
    session.get.side_effect = [throttled, response]
    result, message = fetch_cve_from_nvd("CVE-2024-1234", cache_dir=tmp_path,
                                      session=session, sleep=lambda _: None, min_interval=0)
    assert result is not None, message
    assert result["nvd_retry_count"] == 1
    assert session.get.call_count == 2
    replay, _ = fetch_cve_from_nvd("CVE-2024-1234", cache_dir=tmp_path, offline=True)
    assert replay["nvd_snapshot_id"] == result["nvd_snapshot_id"]
    assert replay["nvd_cache_hit"] is True


def test_bad_id_never_reaches_network(tmp_path):
    session = Mock()
    result, _ = fetch_cve_from_nvd(" CVE-2024-1234", session=session, cache_dir=tmp_path)
    assert result is None
    session.get.assert_not_called()


def test_offline_missing_snapshot_never_calls_network(tmp_path):
    session = Mock()
    result, _ = fetch_cve_from_nvd("CVE-2024-1234", cache_dir=tmp_path, offline=True, session=session)
    assert result is None
    session.get.assert_not_called()


def test_server_error_exhaustion(tmp_path):
    session = Mock()
    session.get.side_effect = requests.Timeout("offline unit fixture")
    result, _ = fetch_cve_from_nvd("CVE-2024-1234", cache_dir=tmp_path, session=session,
                                sleep=lambda _: None, max_retries=1, min_interval=0)
    assert result is None
    assert session.get.call_count == 2


def test_server_503_retries_then_succeeds(tmp_path):
    retry = Mock(status_code=503, headers={})
    success = Mock(status_code=200, headers={})
    success.json.return_value = payload({"cvssMetricV31": [metric()]})
    session = Mock()
    session.get.side_effect = [retry, success]
    result, _ = fetch_cve_from_nvd("CVE-2024-1234", cache_dir=tmp_path, session=session,
                                sleep=lambda _: None, max_retries=1, min_interval=0)
    assert result["nvd_retry_count"] == 1
    assert session.get.call_count == 2


def test_http_404_is_not_retried(tmp_path):
    session = Mock()
    session.get.return_value = Mock(status_code=404)
    result, message = fetch_cve_from_nvd("CVE-2024-1234", cache_dir=tmp_path, session=session,
                                      sleep=lambda _: None, min_interval=0)
    assert result is None
    assert "404" in message
    assert session.get.call_count == 1


def test_snapshot_hash_mismatch_rejected(tmp_path):
    path = tmp_path / "CVE-2024-1234.json"
    path.write_text(json.dumps({"snapshot_id": "wrong", "response": payload()}))
    result, message = fetch_cve_from_nvd("CVE-2024-1234", cache_dir=tmp_path, offline=True)
    assert result is None
    assert "snapshot" in message.lower()


def test_wrong_cve_snapshot_rejected():
    with pytest.raises(ValueError):
        normalize_nvd_response(payload(), "CVE-2024-9999")
