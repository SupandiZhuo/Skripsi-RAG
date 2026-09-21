"""LLM contextual assessment step of the NVD -> RAG -> LLM -> CVSS pipeline.

Kept separate from Streamlit so the schema/call can be reused by a future
evaluation script. The LLM never computes the final numerical CVSS score or
the modified CVSS vector string - both remain fully deterministic (see
app.build_modified_vector / the `cvss` library). The LLM only justifies the
already-deterministic modified vector and is asked to echo it back verbatim
so that any drift/hallucination can be detected and rejected.

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


class NistImpactLevel(str, Enum):
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ContextualAssessment(BaseModel):
    """Schema-validated LLM output. modified_vector must echo the
    deterministic vector given as input - it is validated, never generated."""

    modified_vector: str = Field(
        description="Echo EXACTLY the deterministic 'Modified Vector' given in the input, unchanged."
    )
    justification: str = Field(
        min_length=1,
        description="Reasoning for CR/IR/AR/MAV values based on the asset context.",
    )
    nist_impact_level: NistImpactLevel
    remediation: str = Field(min_length=1, description="Concrete mitigation steps.")

    @field_validator("justification", "remediation")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be blank")
        return v


class LlmAssessmentError(Exception):
    """Raised when the model response is missing, malformed, refused, or
    inconsistent with the deterministic modified vector. Callers must stop
    before any CVSS calculation when this is raised."""


TEMPLATE = """
Anda adalah Senior Cybersecurity Analyst. Jelaskan dampak kontekstual, JANGAN memodifikasi vektor.
DILARANG KERAS menghitung skor angka atau mengubah string vektor.

INPUT:
- CVE Description: {cve_desc}
- Original Vector: {original_vector}
- Modified Vector (deterministik, sudah final): {modified_vector}
- Asset Context: {asset_context}

Kembalikan modified_vector PERSIS seperti input di atas (jangan diubah).
Isi justification: alasan nilai CR, IR, AR, MAV pada modified vector berdasarkan aset.
Isi nist_impact_level: salah satu dari None, Low, Medium, High.
Isi remediation: langkah mitigasi konkret.
"""


def analyze_with_llm(
    api_key: str,
    cve_desc: str,
    original_vector: str,
    modified_vector: str,
    asset_context: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> ContextualAssessment:
    """Call the LLM for contextual justification only. Returns a validated
    ContextualAssessment. Raises LlmAssessmentError on any invalid/refused/
    malformed response, or if the model altered the deterministic vector."""
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)
    structured_llm = llm.with_structured_output(ContextualAssessment)

    prompt = PromptTemplate(
        template=TEMPLATE,
        input_variables=["cve_desc", "original_vector", "modified_vector", "asset_context"],
    )
    chain = prompt | structured_llm

    try:
        import json as _json

        result = chain.invoke(
            {
                "cve_desc": cve_desc,
                "original_vector": original_vector,
                "modified_vector": modified_vector,
                "asset_context": _json.dumps(asset_context),
            }
        )
    except Exception as e:
        raise LlmAssessmentError(f"Model tidak mengembalikan respons terstruktur yang valid: {e}") from e

    if not isinstance(result, ContextualAssessment):
        raise LlmAssessmentError("Model mengembalikan format respons yang tidak dikenali/tidak valid.")

    if result.modified_vector.strip() != modified_vector.strip():
        raise LlmAssessmentError(
            "Model mengubah vektor CVSS deterministik pada responsnya - respons ditolak demi integritas kalkulasi CVSS."
        )

    return result
