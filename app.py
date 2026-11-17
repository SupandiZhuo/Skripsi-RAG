"""Streamlit entry point; execute the UI afresh on every Streamlit rerun."""
import runpy

runpy.run_module("skripsi_rag.ui", run_name="__main__")
