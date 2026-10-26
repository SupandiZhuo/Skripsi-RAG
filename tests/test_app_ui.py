"""Headless regression tests for the focused four-page Streamlit workflow."""

import json
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


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


class _FakeVectorStore:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.records = {row["asset_id"]: dict(row) for row in ASSETS}
        self._collection = self
        _FakeVectorStore.last_instance = self

    def get(self, ids=None, include=None):
        if ids is not None:
            found = [doc_id.removeprefix("asset:") for doc_id in ids if doc_id.removeprefix("asset:") in self.records]
            return {
                "ids": [f"asset:{asset_id}" for asset_id in found],
                "metadatas": [dict(self.records[asset_id]) for asset_id in found],
            }
        ids = [f"asset:{asset_id}" for asset_id in self.records]
        return {"ids": ids, "metadatas": [dict(row) for row in self.records.values()]}

    def update(self, ids, metadatas):
        for doc_id, updates in zip(ids, metadatas):
            asset_id = doc_id.removeprefix("asset:")
            self.records[asset_id].update(updates)

    def similarity_search_with_relevance_scores(self, query, k=3):
        docs = [
            (SimpleNamespace(metadata=dict(row), page_content=row.get("description", "")), 0.9 - i * 0.1)
            for i, row in enumerate(self.records.values())
        ]
        return docs[:k]


def _patch_streamlit_dependencies(monkeypatch):
    import langchain_chroma
    import langchain_huggingface
    import streamlit as st

    monkeypatch.setattr(langchain_chroma, "Chroma", _FakeVectorStore)
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", _FakeEmbeddings)
    st.cache_data.clear()
    st.cache_resource.clear()


def _app_test(monkeypatch, page="Dashboard"):
    _patch_streamlit_dependencies(monkeypatch)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.session_state["nav_page"] = page
    return app


def test_dark_mode_button_toggles_and_persists_for_session(monkeypatch):
    app = _app_test(monkeypatch).run(timeout=60)
    assert not app.exception
    assert app.session_state["dark_mode"] is False

    app.button(key="theme_toggle").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["dark_mode"] is True
    assert app.button(key="theme_toggle").label == "Switch to light mode"

    app.button(key="theme_toggle").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["dark_mode"] is False


def test_navigation_has_exactly_four_pages_and_no_graph_component(monkeypatch):
    app = _app_test(monkeypatch).run(timeout=60)

    assert not app.exception
    assert app.radio[0].options == [
        "Dashboard",
        "Asset Inventory",
        "Vulnerability Analysis",
        "Analysis History",
    ]

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
    app.text_input(key="am_name_SRV-002").set_value("Edited Portal").run(timeout=60)

    save_button = next(button for button in app.button if button.label == "Save changes")
    save_button.click().run(timeout=60)

    assert not app.exception
    stored = _FakeVectorStore.last_instance.records["SRV-002"]
    assert stored["asset_name"] == "Edited Portal"
    assert stored["asset_id"] == "SRV-002"


def test_inventory_analysis_action_preselects_stable_asset_id(monkeypatch):
    app = _app_test(monkeypatch, "Asset Inventory").run(timeout=60)
    app.selectbox(key="inventory_asset_select").select("SRV-002").run(timeout=60)
    app.button(key="inventory_analyze_action").click().run(timeout=60)

    assert not app.exception
    assert app.session_state["nav_page"] == "Vulnerability Analysis"
    assert app.session_state["analysis_asset_select"] == "SRV-002"


def test_analysis_uses_nvd_vector_selected_asset_and_unknown_applicability(monkeypatch):
    import llm_assessment
    import requests
    from llm_assessment import ContextualAssessment

    _patch_streamlit_dependencies(monkeypatch)

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "vulnerabilities": [{
                    "cve": {
                        "descriptions": [{"lang": "en", "value": "Test vulnerability description."}],
                        "metrics": {"cvssMetricV31": [{"cvssData": {
                            "baseScore": 9.8,
                            "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                        }}]},
                        "configurations": {},
                    }
                }]
            }

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(
        llm_assessment,
        "analyze_with_llm",
        lambda *args, **kwargs: ContextualAssessment(
            confidentiality_requirement="H",
            integrity_requirement="H",
            availability_requirement="H",
            organizational_priority="P1",
            justification="Assessment uses the selected asset context.",
            remediation="Apply the vendor patch.",
        ),
    )

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.session_state["nav_page"] = "Vulnerability Analysis"
    app.session_state["analysis_asset_preselect"] = "SRV-002"
    app.run(timeout=60)
    api_key = next(widget for widget in app.text_input if widget.label == "Gemini API Key")
    api_key.set_value("test-key").run(timeout=60)
    app.text_input(key="cve_input").set_value("CVE-2021-44228").run(timeout=60)
    next(button for button in app.button if button.label == "Run Analysis").click().run(timeout=60)

    assert not app.exception
    assert app.session_state["analysis_record"].asset_id == "SRV-002"
    assert app.session_state["analysis_record"].base_score == 9.8
    assert app.session_state["analysis_record"].applicability == "Unknown"
    rendered_text = [str(element.value) for element in [*app.markdown, *app.caption]]
    assert any("Unknown" in text and "not confirmed" in text for text in rendered_text)
    assert any(metric.label == "Context-Aware CVSS Score" for metric in app.metric)
    assert any(metric.label == "Organizational Priority" for metric in app.metric)
