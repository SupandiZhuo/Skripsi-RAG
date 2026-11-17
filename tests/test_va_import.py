import csv
import io

import pytest

from skripsi_rag.va_import import parse_acunetix_csv, results_csv


HEADERS = ["Target", "Name", "Affects", "ModuleName", "CVEList", "Severity", "IsFalsePositive",
           "CVSS3 Descriptor", "CVSS3 Score", "CVSS4 Descriptor", "CVSS4 SI", "CVSS4 SA", "Reference (Name|Url)"]


def csv_bytes(rows, headers=HEADERS):
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([row.get(h, "") for h in headers] if isinstance(row, dict) else row)
    return out.getvalue().encode("utf-8-sig")


def finding(**changes):
    row = {"Target": "https://portal.example.test", "Name": "Library vulnerability",
           "Affects": "/one", "ModuleName": "library-check", "CVEList": '"1234:2024:CVE-2024-1234"',
           "Severity": "High", "CVSS3 Descriptor": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
           "CVSS3 Score": "9.8"}
    row.update(changes)
    return row


def test_multiline_quotes_bom_and_explicit_multi_cve_identifiers():
    report = parse_acunetix_csv(csv_bytes([finding(Name='A, "quoted"\nname',
        CVEList='"1234:2024:CVE-2024-1234, cve-2023-99999, CVE-2024-1234"')]))
    f = report["findings"][0]
    assert f["name"] == 'A, "quoted"\nname'
    assert f["cve_ids"] == ["CVE-2023-99999", "CVE-2024-1234"]
    assert f["scanner_cvss_vector"].startswith("CVSS:3.0/")
    assert f["row_number"] == 2


def test_empty_quoted_cve_is_not_inferred_from_name_or_references():
    f = parse_acunetix_csv(csv_bytes([finding(Name="CVE-2024-1234", CVEList='""',
        **{"Reference (Name|Url)": "CVE-2024-1234|https://example.test"})]))["findings"][0]
    assert not f["cve_ids"]
    assert not f["issues"]


def test_known_omitted_cvss4_cells_restores_reference_with_diagnostic():
    row = [finding().get(h, "") for h in HEADERS]
    reference = '"CVE-2024-1234|https://nvd.nist.gov/vuln/detail/CVE-2024-1234"'
    row[-1] = reference
    del row[-3:-1]
    f = parse_acunetix_csv(csv_bytes([row]))["findings"][0]
    assert f["import_status"] == "ready"
    assert f["references"] == reference
    assert f["cve_ids"] == ["CVE-2024-1234"]
    assert f["issues"]


def test_unknown_column_mismatch_keeps_invalid_row_without_cve():
    report = parse_acunetix_csv(csv_bytes([["target", "name"], finding()]))
    assert len(report["findings"]) == 2
    assert report["findings"][0]["import_status"] == "invalid_row"
    assert not report["findings"][0]["cve_ids"]
    assert report["findings"][1]["import_status"] == "ready"


@pytest.mark.parametrize("content", [b"", b"Name,URL,FullURL\na,b,c\n", b"Target,Target,Name\na,b,c\n",
                                     b'Name,Target,CVEList,Severity,Affects\n"broken', b"\xff"])
def test_invalid_reports_are_rejected(content):
    with pytest.raises(ValueError):
        parse_acunetix_csv(content)


def test_field_limit_error_is_distinct_from_bad_quoting():
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(64)
        with pytest.raises(ValueError, match="field limit"):
            parse_acunetix_csv(csv_bytes([finding(Name="x" * 65)]))
    finally:
        csv.field_size_limit(previous_limit)


def test_bom_only_csv_reports_missing_header():
    with pytest.raises(ValueError, match="no header"):
        parse_acunetix_csv(b"\xef\xbb\xbf")


def test_false_positive_flag_and_raw_requests_are_not_exported():
    content = csv_bytes([finding(IsFalsePositive="true", Request="Cookie: secret")], HEADERS + ["Request"])
    report = parse_acunetix_csv(content)
    assert report["findings"][0]["import_status"] == "false_positive"
    assert "secret" not in str(report)


def test_csv_export_escapes_formulas_and_preserves_numeric_scores():
    rows = list(csv.DictReader(io.StringIO(results_csv([{"Finding": " =WEBSERVICE(x)", "Score": 9.8}]).decode("utf-8-sig"))))
    assert rows[0]["Finding"].startswith("'")
    assert rows[0]["Score"] == "9.8"
