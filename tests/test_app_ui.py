"""Headless regressions for the three-page Asset Intel workflow (offline fakes)."""

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest
from test_va_import import csv_bytes, finding


ASSETS = [
    {
        "asset_id": "SRV-001",
        "asset_name": "Payment Gateway Database",
        "ip_address": "10.0.0.10",
        "description": "Stores payment data.",
        "asset_type": "database",
        "business_criticality": "Mission Critical",
        "internet_exposed": "no",
        "network_zone": "Restricted",
        "data_sensitivity": "Restricted",
        "environment": "production",
        "confidentiality_requirement": "H",
        "integrity_requirement": "H",
        "availability_requirement": "H",
        "software_inventory_json": "[]",
    },
    {
        "asset_id": "SRV-002",
        "asset_name": "Corporate Web Portal",
        "ip_address": "10.0.0.20",
        "description": "Public-facing portal.",
        "asset_type": "web_server",
        "business_criticality": "High",
        "internet_exposed": "yes",
        "network_zone": "DMZ",
        "data_sensitivity": "Public",
        "environment": "production",
        "confidentiality_requirement": "L",
        "integrity_requirement": "L",
        "availability_requirement": "L",
        "software_name": "Apache HTTP Server",
        "software_version": "2.4.51",
        "cpe": "cpe:2.3:a:apache:http_server:2.4.51:*:*:*:*:*:*:*",
        "software_inventory_json": json.dumps([{
            "name": "Apache HTTP Server",
            "version": "2.4.51",
            "cpe": "cpe:2.3:a:apache:http_server:2.4.51:*:*:*:*:*:*:*",
        }]),
    },
]


class _FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, documents):
        return [[0.0, 1.0] for _ in documents]


class _FakeVectorStore:
    last_instance = None
    instances = {}

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("collection_name", "assets")
        self.metadata = kwargs.get("collection_metadata", {})
        self.embeddings = kwargs.get("embedding_function")
        self.records = {row["asset_id"]: dict(row) for row in ASSETS}
        self.document_texts = {}
        self._collection = self
        self.search_calls = []
        self.documents = [(_document_fixture("SRV-002"), 0.9), (_document_fixture("SRV-001"), 0.99)]
        self.instances[self.name] = self
        _FakeVectorStore.last_instance = self

    def get(self, ids=None, include=None):
        if ids is not None:
            found = [doc_id.removeprefix("asset:") for doc_id in ids if doc_id.removeprefix("asset:") in self.records]
            return {
                "ids": [f"asset:{asset_id}" for asset_id in found],
                "metadatas": [dict(self.records[asset_id]) for asset_id in found],
                "documents": [self.document_texts.get(asset_id, "") for asset_id in found],
            }
        ids = [f"asset:{asset_id}" for asset_id in self.records]
        return {"ids": ids, "metadatas": [dict(row) for row in self.records.values()]}

    def update(self, ids, metadatas):
        for doc_id, updates in zip(ids, metadatas):
            asset_id = doc_id.removeprefix("asset:")
            self.records[asset_id].update(updates)

    def add(self, ids, documents, metadatas, embeddings):
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            asset_id = doc_id.removeprefix("asset:")
            if asset_id in self.records:
                raise ValueError("Duplicate ID")
            self.records[asset_id] = dict(metadata)
            self.document_texts[asset_id] = document

    def update_documents(self, ids, documents):
        for doc_id, document in zip(ids, documents):
            asset_id = doc_id.removeprefix("asset:")
            self.records[asset_id] = dict(document.metadata)
            self.document_texts[asset_id] = document.page_content

    def similarity_search_with_relevance_scores(self, query, k=3, filter=None):
        self.search_calls.append({"query": query, "k": k, "filter": filter})
        if self.name == "organizational_context":
            # Return an unrelated document too, testing the defensive post-filter.
            return self.documents[:k]
        docs = [
            (SimpleNamespace(metadata=dict(row), page_content=row.get("description", "")), 0.9 - i * 0.1)
            for i, row in enumerate(self.records.values())
        ]
        return docs[:k]


def _patch_streamlit_dependencies(monkeypatch):
    from skripsi_rag import assessment_history
    import langchain_chroma
    import langchain_huggingface
    import streamlit as st

    monkeypatch.setattr(langchain_chroma, "Chroma", _FakeVectorStore)
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", _FakeEmbeddings)
    history = []
    monkeypatch.setattr(assessment_history, "load_history", lambda **kwargs: list(history))
    monkeypatch.setattr(assessment_history, "save_assessment", history.append)
    monkeypatch.setattr(assessment_history, "delete_record", lambda record: history.remove(record))
    _FakeVectorStore.instances = {}
    st.cache_data.clear()
    st.cache_resource.clear()
    return history


def _app_test(monkeypatch, page="Vulnerability Analysis"):
    _patch_streamlit_dependencies(monkeypatch)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.session_state["nav_page"] = page
    return app


def test_theme_is_native_and_both_theme_variants_are_configured(monkeypatch):
    import tomllib

    configuration = tomllib.loads(
        (Path(__file__).resolve().parents[1] / ".streamlit/config.toml").read_text()
    )
    assert configuration["theme"]["base"] == "light"
    assert configuration["theme"]["paddingTop"] == "0.5rem"
    assert configuration["client"]["toolbarMode"] == "viewer"
    for variant in ("light", "dark"):
        assert configuration["theme"][variant]["backgroundColor"]
        assert configuration["theme"][variant]["textColor"]
        assert configuration["theme"][variant]["secondaryBackgroundColor"]
        assert configuration["theme"][variant]["sidebar"]["backgroundColor"]
    app = _app_test(monkeypatch).run(timeout=60)
    assert not app.exception
    assert not any(button.key == "theme_toggle" for button in app.button)


def test_navigation_has_exactly_three_pages_and_defaults_to_analysis(monkeypatch):
    app = _app_test(monkeypatch).run(timeout=60)

    assert not app.exception
    assert app.radio[0].options == [
        "Vulnerability Analysis",
        "Asset Inventory",
        "Analysis History",
    ]
    assert app.session_state["nav_page"] == "Vulnerability Analysis"
    assert app.get("file_uploader")
    assert not any(element.key == "cve_input" for element in app.text_input)
    assert app.text_input(key="gemini_model").value
    assert any("Settings" in expander.label for expander in app.expander)

def test_asset_list_selection_opens_matching_asset_details(monkeypatch):
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    assert not app.exception

    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)

    assert not app.exception
    assert app.session_state["inventory_selected_asset_id"] == "SRV-002"
    assert any("Corporate Web Portal" in str(element.value) for element in app.subheader)
    assert any("Software/CPE Summary" in str(element.value.columns) for element in app.dataframe)


def test_asset_edit_persists_metadata_without_changing_stable_id(monkeypatch):
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)
    app.text_input(key="am_name_SRV-002").set_value("Edited Portal")

    save_button = next(button for button in app.button if button.label == "Save changes")
    save_button.click().run(timeout=60)

    assert not app.exception
    stored = _FakeVectorStore.last_instance.records["SRV-002"]
    assert stored["asset_name"] == "Edited Portal"
    assert stored["asset_id"] == "SRV-002"


@pytest.mark.parametrize("inventory", ['{"unreadable inventory":', '[{"cpe":"unknown"}]'])
def test_metadata_edit_preserves_inventory_that_cannot_be_edited(monkeypatch, inventory):
    monkeypatch.setitem(ASSETS[1], "software_inventory_json", inventory)
    monkeypatch.setitem(ASSETS[1], "cpe", "unknown")
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)
    assert any("Software inventory could not be loaded" in error.value for error in app.error)
    assert not any(button.key == "sw_save_SRV-002" for button in app.button)
    app.text_input(key="am_name_SRV-002").set_value("Edited without replacing inventory")
    next(button for button in app.button if button.label == "Save changes").click().run(timeout=60)
    assert not app.exception
    stored = _FakeVectorStore.instances["corporate_assets_v2"].records["SRV-002"]
    assert stored["asset_name"] == "Edited without replacing inventory"
    assert stored["software_inventory_json"] == inventory
    assert stored["cpe"] == "unknown"


def test_inventory_analysis_action_preselects_stable_asset_id(monkeypatch):
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)
    app.button(key="inventory_analyze_action").click().run(timeout=60)

    assert not app.exception
    assert app.session_state["nav_page"] == "Vulnerability Analysis"
    assert app.session_state["analysis_asset_select"] == "SRV-002"


def test_software_inventory_verification_is_explicit_and_persisted(monkeypatch):
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)
    assert app.checkbox(key="sw_verified_SRV-002").value is False
    app.checkbox(key="sw_verified_SRV-002").check().run(timeout=60)
    app.button(key="sw_save_SRV-002").click().run(timeout=60)
    assert not app.exception
    assert _FakeVectorStore.last_instance.records["SRV-002"]["inventory_verified"] == "true"


def test_unverified_unknown_cpe_can_be_saved_without_losing_source_metadata(monkeypatch):
    item = {"name": "PostgreSQL", "software_name": "PostgreSQL", "version": "16",
            "software_version": "16", "cpe": "unknown", "source_reference": "Test questionnaire"}
    monkeypatch.setitem(ASSETS[1], "software_inventory_json", json.dumps([item]))
    monkeypatch.setitem(ASSETS[1], "is_synthetic", "true")
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)
    assert any("synthetic/demo" in warning.value for warning in app.warning)
    app.button(key="sw_save_SRV-002").click().run(timeout=60)
    assert not app.exception
    metadata = _FakeVectorStore.instances["corporate_assets_v2"].records["SRV-002"]
    assert json.loads(metadata["software_inventory_json"]) == [item]
    assert metadata["inventory_verified"] == "false"


def test_unknown_cpe_does_not_allow_complete_inventory_verification(monkeypatch):
    monkeypatch.setitem(ASSETS[1], "software_inventory_json", json.dumps([
        {"name": "PostgreSQL", "version": "16", "cpe": "unknown"}]))
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)
    app.checkbox(key="sw_verified_SRV-002").check().run(timeout=60)
    app.button(key="sw_save_SRV-002").click().run(timeout=60)
    assert not app.exception
    assert any("valid CPE for each" in error.value for error in app.error)
    assert _FakeVectorStore.instances["corporate_assets_v2"].records["SRV-002"].get("inventory_verified") != "true"


def _document_fixture(asset_id):
    from skripsi_rag.rag_retrieval import context_chunk_id

    content = f"Operational note for {asset_id}: public portal availability supports business operations."
    metadata = {
        "asset_id": asset_id, "source_id": "ops", "document_id": "portal-note",
        "source": "Operations handbook", "source_reference": "ops/portal.md",
        "last_updated": "2026-10-01", "is_synthetic": "true", "chunk_index": 0,
        "chunk_id": context_chunk_id(asset_id, "ops", "portal-note", 0),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    return SimpleNamespace(page_content=content, metadata=metadata)


def _ready_analysis(monkeypatch, justification="Assessment uses selected asset evidence.", configurations=None):
    from skripsi_rag import llm_assessment
    from skripsi_rag import nvd_client
    from skripsi_rag.llm_assessment import ContextualAssessment

    history = _patch_streamlit_dependencies(monkeypatch)
    calls = {"nvd": [], "model": []}

    def fetch(cve_id):
        calls["nvd"].append(cve_id)
        return {
            "cve_id": cve_id, "description": "Test vulnerability description.",
            "base_score": 9.8, "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "configurations": configurations or [], "nvd_snapshot_id": "fixture-snapshot",
            "nvd_cvss_source": "nvd@nist.gov", "cvss_version": "3.1",
        }, "Fixture NVD snapshot"

    def analyze(api_key, cve_desc, original_vector, asset_context, model):
        description, vector, context = cve_desc, original_vector, asset_context
        calls["model"].append({"description": description, "vector": vector, "context": context, "model": model})
        evidence_id = context["retrieved_evidence"][0]["evidence_id"] if context["retrieved_evidence"] else context["asset_evidence_id"]
        return ContextualAssessment.model_validate({
            "confidentiality_requirement": "H", "integrity_requirement": "H", "availability_requirement": "H",
            "metric_reasons": {metric: {"reason": "Supported by supplied organizational evidence.", "evidence_ids": [evidence_id]}
                               for metric in ("CR", "IR", "AR")},
            "organizational_priority": "P1", "justification": justification,
            "remediation": "Apply the vendor patch.",
        })

    monkeypatch.setattr(nvd_client, "fetch_cve_from_nvd", fetch)
    monkeypatch.setattr(llm_assessment, "analyze_with_llm", analyze)
    from skripsi_rag import pipeline
    monkeypatch.setattr(pipeline, "analyze_with_llm", analyze)
    uploaded = io.BytesIO(csv_bytes([finding(CVEList="CVE-2021-44228")]))
    uploaded.name = "synthetic-va.csv"
    import streamlit as st
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.session_state["analysis_asset_preselect"] = "SRV-002"
    app.run(timeout=60)
    app.text_input(key="gemini_api_key").set_value("test-key").run(timeout=60)
    return app, calls, history


def _submit_analysis(app):
    app.checkbox(key="cloud_consent").check()
    next(button for button in app.button if button.label == "Run Analysis").click().run(timeout=60)
    return app


def test_analysis_uses_selected_asset_document_evidence_and_keeps_result_on_rerun(monkeypatch):
    app, calls, history = _ready_analysis(monkeypatch)
    _submit_analysis(app)
    assert not app.exception
    record = app.session_state["analysis_record"]
    assert record.asset_id == "SRV-002"
    assert record.base_score == 9.8
    assert record.applicability == "Unknown"
    assert record.priority is None
    assert record.investigation_priority in {"P1", "P2", "P3", "P4"}
    context = calls["model"][0]["context"]
    assert context["asset"]["asset_id"] == "SRV-002"
    assert "confidentiality_requirement" not in context["asset"]
    assert len(context["retrieved_evidence"]) == 1
    assert context["retrieved_evidence"][0]["metadata"]["asset_id"] == "SRV-002"
    assert record.evidence["model_context"] == context
    assert record.provenance["model_input"]["asset_context"] == context
    assert record.provenance["assessment"]["nvd_snapshot_id"] == "fixture-snapshot"
    assert record.provenance["source"] == "Acunetix CSV"
    assert record.evidence["scanner_finding"]["cve_ids"] == ["CVE-2021-44228"]
    assert record.model_output["metric_reasons"]["CR"]["evidence_ids"]
    assert "test-key" not in json.dumps(asdict(record))
    store = _FakeVectorStore.instances["organizational_context"]
    assert store.search_calls[0]["filter"] == {"asset_id": "SRV-002"}
    assert record.evidence["document_retrieval"]["rejected"][0]["rejection_reason"] == "asset_id_mismatch"
    assert app.button(key="new_analysis")
    # Reruns must preserve results without making another model call.
    app.run(timeout=60)
    assert not app.exception
    assert app.session_state["analysis_result"]["cve_id"] == "CVE-2021-44228"
    assert len(calls["model"]) == 1
    assert {"Justification Details", "Retrieved Context (RAG)", "Technical Details (CVSS, CPE, etc.)"}.issubset(
        {expander.label for expander in app.expander}
    )
    app.button(key="save_analysis").click().run(timeout=60)
    assert not app.exception
    assert len(history) == 1
    assert app.session_state["analysis_result"] is not None
    app.button(key="new_analysis").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["analysis_result"] is None
    assert app.checkbox(key="cloud_consent").value is False


def test_cloud_consent_is_explicit_and_blocks_external_calls(monkeypatch):
    app, calls, _ = _ready_analysis(monkeypatch)
    next(button for button in app.button if button.label == "Run Analysis").click().run(timeout=60)
    assert not app.exception
    assert not calls["nvd"] and not calls["model"]
    assert any("consent" in error.value for error in app.error)


def test_result_html_escapes_untrusted_model_summary(monkeypatch):
    app, _, _ = _ready_analysis(monkeypatch, justification='<img src=x onerror="alert(1)"> & evidence')
    _submit_analysis(app)
    assert not app.exception
    html = "\n".join(element.proto.body for element in app.get("html"))
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; evidence" in html
    assert '<img src=x onerror="alert(1)">' not in html


def test_no_accepted_document_evidence_is_explicit_and_not_substituted(monkeypatch):
    app, calls, _ = _ready_analysis(monkeypatch)
    monkeypatch.setattr(_FakeVectorStore, "similarity_search_with_relevance_scores", lambda *args, **kwargs: [])
    _submit_analysis(app)
    assert not app.exception
    assert calls["model"][0]["context"]["retrieved_evidence"] == []
    assert calls["model"][0]["context"]["asset"]["asset_id"] == "SRV-002"
    assert any("No accepted document evidence" in warning.value for warning in app.warning)


def test_unmapped_target_is_retained_without_external_calls(monkeypatch):
    app, calls, _ = _ready_analysis(monkeypatch)
    app.selectbox(key="analysis_asset_select").select("")
    next(button for button in app.button if button.label == "Run Analysis").click().run(timeout=60)
    assert not app.exception
    assert not calls["nvd"] and not calls["model"]
    assert app.session_state["va_batch"]["cases"][0]["status"] == "unmapped_asset"


def test_upload_change_clears_results_and_requires_new_consent(monkeypatch):
    app, calls, _ = _ready_analysis(monkeypatch)
    _submit_analysis(app)
    import streamlit as st
    uploaded = io.BytesIO(csv_bytes([finding(CVEList='""')]))
    uploaded.name = "other-report.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app.run(timeout=60)
    assert not app.exception
    assert app.session_state["va_batch"] is None
    assert not app.checkbox(key="cloud_consent").value
    assert app.selectbox(key="analysis_asset_select").value == ""
    assert len(calls["model"]) == 1


def test_confirmed_applicability_produces_separate_deterministic_priority(monkeypatch):
    configs = [{"nodes": [{"operator": "OR", "cpeMatch": [{"vulnerable": True,
        "criteria": ASSETS[1]["cpe"]}]}]}]
    app, calls, _ = _ready_analysis(monkeypatch, configurations=configs)
    _submit_analysis(app)
    assert not app.exception
    case = app.session_state["va_batch"]["cases"][0]
    assert case["status"] == "complete"
    assert case["assessment"]["applicability"]["status"] == "Confirmed"
    assert case["assessment"]["deterministic_priority"] in {"P1", "P2", "P3", "P4"}
    assert case["assessment"]["technical_severity"] == "Critical"
    assert len(calls["model"]) == 1


def test_locations_export_is_rejected_before_any_external_call(monkeypatch):
    app, calls, _ = _ready_analysis(monkeypatch)
    import streamlit as st
    uploaded = io.BytesIO(b"Name,URL,FullURL,Inputs (Name:Type)\na,b,c,d\n")
    uploaded.name = "locations.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app.run(timeout=60)
    assert not app.exception
    assert any("Locations export" in error.value for error in app.error)
    assert not calls["nvd"] and not calls["model"]


def test_cveless_report_keeps_scanner_recommendations_without_cloud_calls(monkeypatch):
    app, calls, _ = _ready_analysis(monkeypatch)
    import streamlit as st
    uploaded = io.BytesIO(csv_bytes([finding(CVEList='""', Recommendation="Set a restrictive policy.")],
        headers=list(finding()) + ["Recommendation"]))
    uploaded.name = "no-cve.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app.run(timeout=60)
    app.text_input(key="gemini_api_key").set_value("")
    next(button for button in app.button if button.label == "Run Analysis").click().run(timeout=60)
    assert not app.exception
    assert not calls["nvd"] and not calls["model"]
    assert app.session_state["va_batch"]["cases"][0]["status"] == "no_cve"
    assert any(element.value == "Set a restrictive policy." for element in app.markdown)
    assert len(app.get("download_button")) == 2


def test_scanner_inventory_conflict_is_reviewable_without_model_inference(monkeypatch):
    configs = [{"nodes": [{"operator": "OR", "cpeMatch": [{"vulnerable": True,
        "criteria": "cpe:2.3:a:other:product:1.0:*:*:*:*:*:*:*"}]}]}]
    app, calls, _ = _ready_analysis(monkeypatch, configurations=configs)
    _FakeVectorStore.instances["corporate_assets_v2"].records["SRV-002"]["inventory_verified"] = True
    _submit_analysis(app)
    assert not app.exception
    case = app.session_state["va_batch"]["cases"][0]
    assert case["status"] == "scanner_inventory_conflict"
    assert case["assessment"]["deterministic_priority"] is None
    assert not calls["model"]
    assert app.session_state["analysis_record"].provenance["case_status"] == "scanner_inventory_conflict"
    assert any("conflict" in warning.value for warning in app.warning)
    assert any("CR/IR/AR have not been assigned" in info.value for info in app.info)


def test_multiple_targets_need_individual_mapping_and_show_import_diagnostics(monkeypatch):
    app, calls, _ = _ready_analysis(monkeypatch)
    import streamlit as st
    uploaded = io.BytesIO(csv_bytes([finding(), finding(Target="https://second.example.test"), ["bad", "row"]]))
    uploaded.name = "multi.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploaded)
    app.run(timeout=60)
    assert not app.exception
    mapping = [select for select in app.selectbox if select.key.startswith("va_map_")]
    assert len(mapping) == 3
    assert all(select.value == "" for select in mapping)
    assert any("Import diagnostics" in exp.label for exp in app.expander)
    next(button for button in app.button if button.label == "Run Analysis").click().run(timeout=60)
    assert not app.exception
    assert not calls["nvd"] and not calls["model"]
    assert len(app.session_state["va_batch"]["cases"]) == 3


def test_inventory_search_filters_real_records_and_add_rejects_duplicate(monkeypatch):
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.text_input(key="inv_search").set_value("10.0.0.20").run(timeout=60)
    assert not app.exception
    table = app.dataframe[0].value
    assert list(table["Asset ID"]) == ["SRV-002"]
    app.button(key="add_asset").click().run(timeout=60)
    for field, value in {
        "asset_id": "SRV-002", "asset_name": "Attempted replacement", "ip_address": "10.0.0.30",
        "description": "New record", "source": "Manual inventory", "source_reference": "ticket-123",
    }.items():
        app.text_input(key=f"new_{field}").set_value(value)
    next(button for button in app.button if button.label == "Create asset").click().run(timeout=60)
    assert not app.exception
    assert any("already exists" in error.value for error in app.error)
    assert _FakeVectorStore.last_instance.records["SRV-002"]["asset_name"] == "Corporate Web Portal"


def test_add_asset_creates_validated_new_record_without_overwriting_inventory(monkeypatch):
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.button(key="add_asset").click().run(timeout=60)
    for field, value in {
        "asset_id": "SRV-003", "asset_name": "New service", "ip_address": "10.0.0.30",
        "description": "New record", "source": "Manual inventory", "source_reference": "ticket-123",
    }.items():
        app.text_input(key=f"new_{field}").set_value(value)
    next(button for button in app.button if button.label == "Create asset").click().run(timeout=60)
    assert not app.exception
    records = _FakeVectorStore.last_instance.records
    assert records["SRV-003"]["asset_name"] == "New service"
    assert records["SRV-002"]["asset_name"] == "Corporate Web Portal"
    assert len(records) == 3


def test_history_table_separates_priority_applicability_and_supports_delete(monkeypatch):
    app, _, history = _ready_analysis(monkeypatch)
    _submit_analysis(app)
    app.button(key="save_analysis").click().run(timeout=60)
    app.radio[0].set_value("Analysis History").run(timeout=60)
    assert not app.exception
    table = app.dataframe[0].value
    assert {"Priority", "Investigation Priority", "Applicability", "Base CVSS", "Env CVSS"}.issubset(table.columns)
    assert table.iloc[0]["Priority"] == "—"
    assert table.iloc[0]["Applicability"] == "Unknown"
    assert app.get("download_button")
    next(button for button in app.button if button.label == "Delete").click().run(timeout=60)
    assert not app.exception
    assert not history


def test_history_read_failure_is_visible_instead_of_an_empty_state(monkeypatch):
    from skripsi_rag import assessment_history

    app = _app_test(monkeypatch, "Analysis History")

    def fail_read(**kwargs):
        raise assessment_history.HistoryReadError("Synthetic invalid history")

    monkeypatch.setattr(assessment_history, "load_history", fail_read)
    app.run(timeout=60)
    assert not app.exception
    assert any("History could not be loaded" in error.value for error in app.error)
    assert not any("No assessments recorded" in info.value for info in app.info)


def test_history_delete_failure_keeps_the_record_and_shows_feedback(monkeypatch):
    from skripsi_rag import assessment_history

    app, _, history = _ready_analysis(monkeypatch)
    _submit_analysis(app)
    app.button(key="save_analysis").click().run(timeout=60)
    app.radio[0].set_value("Analysis History").run(timeout=60)

    def fail_delete(record):
        raise OSError("Synthetic file write failure")

    monkeypatch.setattr(assessment_history, "delete_record", fail_delete)
    app.run(timeout=60)
    next(button for button in app.button if button.label == "Delete").click().run(timeout=60)
    assert not app.exception
    assert len(history) == 1
    assert any("History delete failed" in error.value for error in app.error)
