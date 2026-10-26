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
    html_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "html"
    ]

    assert "asset_graph" not in imported_modules
    assert "render_cytoscape_html" not in source
    assert len(html_calls) == 1
    assert isinstance(html_calls[0].args[0], ast.Name)
    assert html_calls[0].args[0].id == "DARK_MODE_STYLES"
