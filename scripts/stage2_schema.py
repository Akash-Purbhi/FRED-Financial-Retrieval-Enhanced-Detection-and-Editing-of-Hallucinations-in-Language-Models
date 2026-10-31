"""
scripts/stage2_schema.py
========================
Stage 2 Unified Dataset Design & Schema Validator.

Defines the formal data contracts, type validations, and serialization helpers
for the FRED Stage 2 dataset (FinQA 10k + TAT-QA 8k = 18k train target).

Required 12 Fields:
1.  id: Unique record identifier
2.  source_dataset: 'finqa' | 'tatqa'
3.  question: Natural language financial question
4.  context: Unstructured financial text / footnotes
5.  table: 2D tabular array [[col1, col2, ...], ...]
6.  gold_answer: Canonical verified answer string
7.  evidence: List of grounding citations with location & text
8.  reasoning: Derivation steps
9.  operation: Executable program / math formula / extraction
10. answer_type: 'number' | 'percentage' | 'date' | 'entity' | 'span_extraction' | 'multi_span'
11. error_type: Error taxonomy category (or 'none' for clean)
12. action: 'preserve' | 'edit' | 'abstain' | 'flag_conflict' | 'request_evidence'
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Literal, Tuple
import json

VALID_DATASETS = {"finqa", "tatqa"}

VALID_ANSWER_TYPES = {
    "number",
    "percentage",
    "date",
    "entity",
    "span_extraction",
    "multi_span",
}

VALID_ERROR_TYPES = {
    "none",
    "numerical_substitution",
    "temporal_swap",
    "arithmetic_miscalculation",
    "wrong_unit",
    "entity_swap",
    "relation_inversion",
    "unsupported_claim",
    "missing_evidence",
    "irrelevant_retrieval",
    "contradictory_context",
}

VALID_ACTIONS = {
    "preserve",
    "edit",
    "abstain",
    "flag_conflict",
    "request_evidence",
}


@dataclass
class EvidenceItem:
    source_type: str  # 'context' | 'table'
    location: str     # paragraph index or table row/col like '[1, 2]'
    text: str         # exact text or cell value

    def to_dict(self) -> Dict[str, str]:
        return {
            "source_type": self.source_type,
            "location": self.location,
            "text": self.text,
        }


@dataclass
class Stage2Record:
    id: str
    source_dataset: str
    question: str
    context: str
    table: List[List[str]]
    gold_answer: str
    evidence: List[Dict[str, str]]
    reasoning: str
    operation: str
    answer_type: str
    error_type: str
    action: str
    # Optional metadata fields for downstream model compatibility
    unverified_response: Optional[str] = None
    tagged_xml_response: Optional[str] = None

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all required fields and enum bounds."""
        errors = []

        if not self.id or not isinstance(self.id, str):
            errors.append("Invalid or missing 'id'")

        if self.source_dataset not in VALID_DATASETS:
            errors.append(f"Invalid 'source_dataset': {self.source_dataset}. Must be one of {VALID_DATASETS}")

        if not self.question or not isinstance(self.question, str):
            errors.append("Missing or empty 'question'")

        if not isinstance(self.table, list):
            errors.append("'table' must be a list of rows")

        if not self.gold_answer or not isinstance(self.gold_answer, str):
            errors.append("Missing or empty 'gold_answer'")

        if not isinstance(self.evidence, list):
            errors.append("'evidence' must be a list of citation objects")

        if self.answer_type not in VALID_ANSWER_TYPES:
            errors.append(f"Invalid 'answer_type': {self.answer_type}. Must be one of {VALID_ANSWER_TYPES}")

        if self.error_type not in VALID_ERROR_TYPES:
            errors.append(f"Invalid 'error_type': {self.error_type}. Must be one of {VALID_ERROR_TYPES}")

        if self.action not in VALID_ACTIONS:
            errors.append(f"Invalid 'action': {self.action}. Must be one of {VALID_ACTIONS}")

        # Semantic consistency checks
        if self.action == "preserve" and self.error_type != "none":
            errors.append(f"Conflict: action is 'preserve' but error_type is '{self.error_type}' (expected 'none')")

        if self.action == "edit" and self.error_type == "none":
            errors.append("Conflict: action is 'edit' but error_type is 'none'")

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        data = asdict(self)
        # Exclude null optional fields if not populated
        return {k: v for k, v in data.items() if v is not None}

    def to_chatml(self) -> Dict[str, Any]:
        """
        Format record into ChatML message structure for Qwen 3B / 7B fine-tuning.
        """
        # Render table as compact text grid
        table_str = "\n".join([" | ".join(row) for row in self.table]) if self.table else "[No Table]"
        candidate = self.unverified_response or self.gold_answer

        user_content = (
            f"Context: {self.context}\n\n"
            f"Table:\n{table_str}\n\n"
            f"Question: {self.question}\n\n"
            f"Response to Verify: {candidate}"
        )

        assistant_payload = {
            "action": self.action,
            "error_type": self.error_type,
            "operation": self.operation,
            "reasoning": self.reasoning,
            "gold_answer": self.gold_answer,
        }
        if self.tagged_xml_response:
            assistant_payload["tagged_xml"] = self.tagged_xml_response

        assistant_content = json.dumps(assistant_payload, ensure_ascii=False)

        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are FRED Stage 2, an evidence-aware financial hallucination verifier. "
                        "Given context, tables, a question, and a response, determine the appropriate action: "
                        "preserve, edit, abstain, flag_conflict, or request_evidence."
                    ),
                },
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }


# ── Sample Exemplar Records for Dataset Design Verification ──────────────────

SAMPLE_PRESERVE = Stage2Record(
    id="finqa_sample_001",
    source_dataset="finqa",
    question="What was cash and short-term investments as a percentage of total purchase price?",
    context="Schlumberger acquired Eastern Echo for $838 million in cash. Net assets were allocated as follows.",
    table=[
        ["Asset", "Amount ($M)"],
        ["Cash and short-term investments", "266"],
        ["Total purchase price", "838"],
    ],
    gold_answer="31.74%",
    evidence=[
        {"source_type": "table", "location": "[1, 1]", "text": "266"},
        {"source_type": "table", "location": "[2, 1]", "text": "838"},
    ],
    reasoning="Dividing cash (266) by total price (838) yields 31.74%.",
    operation="divide(266, 838)",
    answer_type="percentage",
    error_type="none",
    action="preserve",
    unverified_response="Cash and short-term investments accounted for 31.74% of the total purchase price.",
)

SAMPLE_EDIT = Stage2Record(
    id="tatqa_sample_002",
    source_dataset="tatqa",
    question="What was the operating profit change from 2021 to 2022?",
    context="Operating results for the fiscal year ended December 31, 2022.",
    table=[
        ["Year", "Operating Profit"],
        ["2021", "$100M"],
        ["2022", "$120M"],
    ],
    gold_answer="20.0%",
    evidence=[
        {"source_type": "table", "location": "[1, 1]", "text": "$100M"},
        {"source_type": "table", "location": "[2, 1]", "text": "$120M"},
    ],
    reasoning="((120 - 100) / 100) * 100 = 20.0%. Response claimed 15.0%.",
    operation="divide(subtract(120, 100), 100)",
    answer_type="percentage",
    error_type="arithmetic_miscalculation",
    action="edit",
    unverified_response="Operating profit grew by 15.0% between 2021 and 2022.",
    tagged_xml_response="Operating profit grew by <numerical><delete>15.0%</delete><mark>20.0%</mark></numerical> between 2021 and 2022.",
)

SAMPLE_ABSTAIN = Stage2Record(
    id="finqa_sample_003",
    source_dataset="finqa",
    question="What was the effective tax rate in 2023?",
    context="Revenues were $50,000M and operating income was $10,000M in 2023.",
    table=[["Metric", "2023"], ["Revenues", "$50,000M"]],
    gold_answer="Unverifiable from provided context.",
    evidence=[{"source_type": "context", "location": "p1", "text": "Revenues were $50,000M"}],
    reasoning="Income tax expense and pre-tax income are not disclosed in context or table.",
    operation="none",
    answer_type="percentage",
    error_type="missing_evidence",
    action="abstain",
    unverified_response="The effective tax rate in 2023 was 21.0%.",
)


if __name__ == "__main__":
    print("Testing Stage2 Dataset Design & Schema Validator...")
    for sample in [SAMPLE_PRESERVE, SAMPLE_EDIT, SAMPLE_ABSTAIN]:
        is_valid, errs = sample.validate()
        assert is_valid, f"Validation failed for {sample.id}: {errs}"
        chatml = sample.to_chatml()
        assert len(chatml["messages"]) == 3, "ChatML must have system, user, assistant"
        print(f"  [PASS] {sample.id} (Action: {sample.action}, ErrorType: {sample.error_type})")

    print("\nStage 2 Dataset Design specification and validation verified successfully!")
