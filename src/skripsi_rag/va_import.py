"""Offline Acunetix CSV ingestion. Scanner evidence never becomes model context.

Keep every CSV record, including invalid rows and CVE-less findings. Only
explicit CVEList identifiers are eligible for the NVD assessment pipeline.
"""
import csv
import hashlib
import io
import re

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_FINDINGS = 10000
_CVE = re.compile(r"(?<![A-Za-z0-9_-])CVE-\d{4}-\d{4,}(?![A-Za-z0-9_-])", re.I)
_REQUIRED = {"Target", "Name", "CVEList", "Severity", "Affects"}


def parse_acunetix_csv(content: bytes, filename: str = "report.csv") -> dict:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload a non-empty CSV up to 10 MB.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must use UTF-8 encoding.") from exc
    report_id = hashlib.sha256(content).hexdigest()
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = next(reader)
        if {"Name", "URL", "FullURL"} <= set(headers) and "CVEList" not in headers:
            raise ValueError("This is a Locations export. Upload the Acunetix Vulnerabilities CSV instead.")
        if len(headers) != len(set(headers)) or not _REQUIRED <= set(headers):
            raise ValueError("Invalid Vulnerabilities CSV: required unique columns are Target, Name, CVEList, Severity, Affects.")
        findings = []
        warnings = []
        for row_number, values in enumerate(reader, start=2):
            if not values or not any(value.strip() for value in values):
                continue
            if len(findings) >= MAX_FINDINGS:
                raise ValueError("CSV exceeds the 10,000 finding limit.")
            status = "ready"
            issues = []
            # Some exports omit two empty CVSS4 cells before the final reference.
            # Repair only that layout; other column shifts remain invalid.
            cvss4 = headers.index("CVSS4 Descriptor") if "CVSS4 Descriptor" in headers else -1
            if (len(values) == len(headers) - 2 and cvss4 >= 0
                    and headers[-3:] == ["CVSS4 SI", "CVSS4 SA", "Reference (Name|Url)"]
                    and all(not v.strip() for v in values[cvss4:-1])
                    and (not values[-1].strip() or "|http" in values[-1])):
                values = values[:-1] + ["", "", values[-1]]
                issues.append("Two omitted empty CVSS4 cells restored; final reference preserved.")
            elif len(values) != len(headers):
                status = "invalid_row"
                issues.append(f"Expected {len(headers)} columns; found {len(values)}. Review source CSV.")
            row = dict(zip(headers, values))
            target = row.get("Target", "").strip()
            name = row.get("Name", "").strip()
            if not name:
                status = "invalid_row"
                issues.append("Finding name is missing.")
            cve_text = row.get("CVEList", "")
            cves = sorted({value.upper() for value in _CVE.findall(cve_text)}) if status == "ready" else []
            if not cves and cve_text.strip(' \t\r\n"[]'):
                issues.append("CVEList contains no valid explicit CVE identifier.")
            false_positive = row.get("IsFalsePositive", "").strip().lower() in {"true", "yes", "1"}
            if false_positive and status == "ready":
                status = "false_positive"
            finding = {
                "finding_id": f"{report_id[:16]}:{row_number}", "row_number": row_number,
                "target": target, "name": name, "affects": row.get("Affects", ""),
                "parameter": row.get("Parameter", ""), "module": row.get("ModuleName", ""),
                "description": row.get("Description", ""), "impact": row.get("Impact", ""),
                "recommendation": row.get("Recommendation", ""),
                "cve_ids": cves, "scanner_severity": row.get("Severity", ""),
                "scanner_cvss_vector": row.get("CVSS3 Descriptor", ""),
                "scanner_cvss_score": row.get("CVSS3 Score", ""),
                "scanner_cvss4_vector": row.get("CVSS4 Descriptor", ""),
                "scanner_cvss4_score": row.get("CVSS4 Score", ""),
                "references": row.get("Reference (Name|Url)", "") if status != "invalid_row" else "",
                "import_status": status, "issues": issues,
            }
            findings.append(finding)
            warnings.extend(f"Row {row_number}: {issue}" for issue in issues)
    except StopIteration as exc:
        raise ValueError("CSV contains no header row.") from exc
    except csv.Error as exc:
        raise ValueError(f"CSV parsing failed: {exc}") from exc
    if not findings:
        raise ValueError("CSV contains no findings.")
    return {"report_id": report_id, "filename": filename, "source": "Acunetix CSV",
            "findings": findings, "warnings": warnings}


def findings_preview(report: dict) -> list[dict]:
    return [{"Row": f["row_number"], "Finding": f["name"], "Target": f["target"],
             "Location": f["affects"], "CVE": ", ".join(f["cve_ids"]),
             "Scanner Severity": f["scanner_severity"], "Scanner CVSS": f["scanner_cvss_score"],
             "Import Status": f["import_status"], "Issues": " ".join(f["issues"])}
            for f in report["findings"]]


def results_csv(rows: list[dict]) -> bytes:
    """Spreadsheet-safe export; untrusted cells cannot become formulas."""
    if not rows:
        return b""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return out.getvalue().encode("utf-8-sig")
