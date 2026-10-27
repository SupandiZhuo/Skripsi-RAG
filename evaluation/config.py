"""Frozen experiment configuration and runtime/input provenance, without secrets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS = ("E0", "E1", "E2", "E3")


@dataclass(frozen=True)
class ExperimentConfig:
    experiments: tuple[str, ...] = ("E0",)
    model: str = "gemini-2.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    relevance_threshold: float = 0.15
    temperature: int = 0
    seed: int = 0
    repeats: int = 1
    prompt_version: str = "source-hash-recorded"
    rubric_version: str = "source-hash-recorded"
    threshold_status: str = "heuristic_not_validation_selected"
    scope: str = "infrastructure_not_empirical_research"

    def __post_init__(self):
        if not isinstance(self.experiments, tuple) or not self.experiments or len(set(self.experiments)) != len(self.experiments) or any(e not in EXPERIMENTS for e in self.experiments):
            raise ValueError("experiments must be a nonempty tuple of distinct E0-E3 IDs.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer.")
        if isinstance(self.relevance_threshold, bool) or not isinstance(self.relevance_threshold, (int, float)) or not math.isfinite(self.relevance_threshold) or not 0 <= self.relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be finite and between 0 and 1.")
        if self.temperature != 0:
            raise ValueError("Current LLM adapter supports temperature=0 only.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer; provider seed support is not assumed.")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "experiments" in data:
            data["experiments"] = tuple(data["experiments"])
        return cls(**data)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_manifest(config: ExperimentConfig, path, *, input_paths=(), extra=None):
    """Create once, refuse overwrite; capture actual runtime versions and sources.

    A runtime inventory is not a claim of cross-provider determinism. The seed
    is recorded for orchestration only; the current cloud adapter has no seed
    parameter. Cloud provider revisions can still change behavior.
    """
    root = Path(__file__).resolve().parents[1]
    config_data = json.loads(json.dumps(asdict(config)))
    canonical = json.dumps(config_data, sort_keys=True, separators=(",", ":"))
    versions = {}
    for name in ("cvss", "pydantic", "requests", "chromadb", "langchain-chroma",
                 "langchain-core", "langchain-google-genai", "langchain-huggingface", "sentence-transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = ["pipeline.py", "nvd_client.py", "cvss_engine.py", "llm_assessment.py",
               "rag_retrieval.py", "cve_applicability.py", "organizational_priority.py",
               "evaluation/config.py", "evaluation/metrics.py", "evaluation/run_experiments.py",
               "requirements.txt", "requirements.lock"]
    manifest = {
        "manifest_version": "experiment-manifest-v1", "config": config_data,
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "dependency_versions": versions,
        "source_sha256": {name: sha256_file(root / name) for name in sources if (root / name).exists()},
        "inputs_sha256": {str(p): sha256_file(p) for p in input_paths},
        "provider_seed_supported": False, "determinism_guaranteed": False,
        "extra": extra or {},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest
