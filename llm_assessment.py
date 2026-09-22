"""LLM contextual assessment step of the NVD -> RAG -> LLM -> CVSS pipeline.

Kept separate from Streamlit so the schema/call can be reused by a future
evaluation script, and so the cloud LLM provider used here can later be
swapped for a separately fine-tuned Llama/Qwen model without touching RAG
retrieval or CVSS calculation (see `_build_llm`).

Responsibility boundary (do not blur):
- The model receives the CVE description, the ORIGINAL (unmodified) NVD
  CVSS v3.1 vector, and the retrieved RAG asset context.
- The model interprets that input and outputs Environmental Requirement
  metrics (CR/IR/AR) and an organizational_priority (P1-P4), each with
  evidence-based justification.
- The model NEVER outputs a CVSS vector string or a numerical CVSS score,
  and NEVER modifies Base metrics (AV/AC/PR/UI/S/C/I/A). Deterministic code
  (see app.build_environmental_vector) appends the validated CR/IR/AR onto
  the preserved NVD base vector, and the `cvss` library performs the only
  numerical scoring.

Structured output is produced via LangChain's `with_structured_output`, which
the installed `langchain-google-genai` (Gemini) integration supports natively
(schema-validated JSON, not regex/markdown cleanup).
"""

from enum import Enum
from typing import Any, Dict

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

DEFAULT_MODEL = "gemini-2.0-flash"


class CiaRequirement(str, Enum):
    """CVSS v3.1 Environmental Requirement value (CR/IR/AR)."""

    NOT_DEFINED = "X"
    LOW = "L"
    MEDIUM = "M"
    HIGH = "H"


class OrganizationalPriority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class ContextualAssessment(BaseModel):
    """Schema-validated LLM output.

    The model is restricted to Environmental Requirement metrics and an
    organizational priority - it must NOT emit a CVSS vector or score.
    """

    confidentiality_requirement: CiaRequirement = Field(
        description="CVSS Environmental Confidentiality Requirement (CR), independently justified."
    )
    integrity_requirement: CiaRequirement = Field(
        description="CVSS Environmental Integrity Requirement (IR), independently justified."
    )
    availability_requirement: CiaRequirement = Field(
        description="CVSS Environmental Availability Requirement (AR), independently justified."
    )
    organizational_priority: OrganizationalPriority = Field(
        description="Organizational remediation priority P1 (highest) to P4 (lowest)."
    )
    justification: str = Field(
        min_length=1,
        description="Concise evidence-based reasoning for CR/IR/AR and organizational_priority, "
        "citing only the supplied CVE info and asset context.",
    )
    remediation: str = Field(min_length=1, description="Concrete mitigation steps.")

    @field_validator("justification", "remediation")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be blank")
        return v


class LlmAssessmentError(Exception):
    """Raised when the model response is missing, malformed, or refused.
    Callers must stop before any CVSS calculation when this is raised."""


TEMPLATE = """
Anda adalah Senior Cybersecurity Analyst. Tugas Anda HANYA menilai kebutuhan
kontekstual organisasi (Environmental Requirements) dan prioritas remediasi.

DILARANG KERAS:
- Menghitung skor CVSS numerik apa pun.
- Menuliskan atau mengubah string vektor CVSS.
- Mengubah metrik Base (AV, AC, PR, UI, S, C, I, A).

INPUT:
- CVE Description: {cve_desc}
- Original NVD CVSS Vector (Base, JANGAN diubah): {original_vector}
- Retrieved Organizational Asset Context (RAG): {asset_context}

Berdasarkan HANYA pada input di atas, tentukan:
- confidentiality_requirement (CR): X, L, M, atau H
- integrity_requirement (IR): X, L, M, atau H
- availability_requirement (AR): X, L, M, atau H
- organizational_priority: P1 (tertinggi) sampai P4 (terendah)

Isi justification: alasan ringkas dan berbasis bukti untuk CR, IR, AR, dan
organizational_priority, mengacu hanya pada deskripsi CVE dan konteks aset
yang diberikan.
Isi remediation: langkah mitigasi konkret.
"""


def _build_llm(api_key: str, model: str) -> ChatGoogleGenerativeAI:
    """Isolated provider construction. Swap this function (and its return
    type usage in analyze_with_llm) to integrate a separately trained
    Llama/Qwen model later without touching the rest of the pipeline."""
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)


def analyze_with_llm(
    api_key: str,
    cve_desc: str,
    original_vector: str,
    asset_context: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> ContextualAssessment:
    """Call the LLM for contextual CR/IR/AR + organizational_priority only.

    Returns a validated ContextualAssessment. Raises LlmAssessmentError on
    any invalid/refused/malformed response. The LLM never sees nor produces
    a modified CVSS vector or numerical score - those are built/calculated
    afterwards by deterministic code + the `cvss` library.
    """
    llm = _build_llm(api_key, model)
    structured_llm = llm.with_structured_output(ContextualAssessment)

    prompt = PromptTemplate(
        template=TEMPLATE,
        input_variables=["cve_desc", "original_vector", "asset_context"],
    )
    chain = prompt | structured_llm

    try:
        import json as _json

        result = chain.invoke(
            {
                "cve_desc": cve_desc,
                "original_vector": original_vector,
                "asset_context": _json.dumps(asset_context),
            }
        )
    except Exception as e:
        raise LlmAssessmentError(f"Model tidak mengembalikan respons terstruktur yang valid: {e}") from e

    if not isinstance(result, ContextualAssessment):
        raise LlmAssessmentError("Model mengembalikan format respons yang tidak dikenali/tidak valid.")

    return result
