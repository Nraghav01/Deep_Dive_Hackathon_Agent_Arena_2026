from typing import TypedDict, Any


class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]

    iteration: int
    max_retries: int
    retry_query: str
    needs_research: bool

    audit_trail: list[dict[str, Any]]
    final_report: dict[str, Any]
    error: str


