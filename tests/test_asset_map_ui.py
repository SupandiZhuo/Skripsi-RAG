"""Regression guard that the retired graph UI is not loaded by the app."""

import ast
from pathlib import Path


def test_active_app_has_no_cytoscape_or_asset_graph_dependency():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported_modules = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in imports
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    retired_graph_modules = {"asset_graph", "cytoscape", "streamlit_cytoscape", "pyvis"}
    assert not any(
        module and module.split(".")[0] in retired_graph_modules
        for module in imported_modules
    )
    assert "render_cytoscape_html" not in source
    assert "cytoscape(" not in source.lower()
    # CSS and escaped result cards are allowed; executable embeds are not.
    assert "<script" not in source.lower()
    assert "components.html" not in source
