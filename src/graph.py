from datetime import datetime, timezone
from typing import Any

from langgraph.graph import StateGraph, START, END

from src.state import ResearchState


def log_event(state: ResearchState, event: str, **details: Any):
    """Append a timestamped event to the audit trail."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": state.get("iteration", 0),
        "event": event,
        **details,
    }

    return state.get("audit_trail", []) + [entry]


# --------------------------------------------------
# 1. PLANNER
# --------------------------------------------------

def planner_node(state: ResearchState) -> dict:
    question = state["question"]

    sub_questions = [
        {
            "id": "SQ01",
            "question": (
                "What products did fictional company NovaCell launch?"
            ),
            "supported_claims": 0,
        },
        {
            "id": "SQ02",
            "question": (
                "What was NovaCell's revenue growth in 2025?"
            ),
            "supported_claims": 0,
        },
    ]

    return {
        "sub_questions": sub_questions,
        "audit_trail": log_event(
            state,
            "planner_completed",
            original_question=question,
            sub_question_ids=["SQ01", "SQ02"],
        ),
    }


# --------------------------------------------------
# 2. RESEARCHER (FIXED TEST PASSAGES)
# --------------------------------------------------

INITIAL_EVIDENCE = [
    {
        "passage_id": "E001",
        "sub_question_id": "SQ01",
        "url": "https://example.com/novacell-products",
        "title": "NovaCell Product Announcement (TEST FIXTURE)",
        "passage_text": (
            "NovaCell launched two new battery products in 2025."
        ),
    }
]

RETRY_EVIDENCE = [
    {
        "passage_id": "E002",
        "sub_question_id": "SQ02",
        "url": "https://example.com/novacell-financials",
        "title": "NovaCell Financial Update (TEST FIXTURE)",
        "passage_text": (
            "NovaCell's revenue grew by 12% in 2025."
        ),
    }
]


def researcher_node(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0)
    evidence = list(state.get("evidence", []))

    if iteration == 0:
        new_evidence = INITIAL_EVIDENCE
        query = "Initial research for NovaCell"
    else:
        # A fixed retry response for graph testing.
        new_evidence = RETRY_EVIDENCE
        query = state.get("retry_query", "Retry research")

    existing_ids = {item["passage_id"] for item in evidence}

    for item in new_evidence:
        if item["passage_id"] not in existing_ids:
            evidence.append(item)
            existing_ids.add(item["passage_id"])

    return {
        "evidence": evidence,
        "audit_trail": log_event(
            state,
            "research_completed",
            query=query,
            new_evidence_ids=[
                item["passage_id"]
                for item in new_evidence
                if item["passage_id"] not in {
                    e["passage_id"]
                    for e in state.get("evidence", [])
                }
            ],
        ),
    }


# --------------------------------------------------
# 3. SYNTHESIZER (DETERMINISTIC TEST CLAIMS)
# --------------------------------------------------

def synthesizer_node(state: ResearchState) -> dict:
    evidence_ids = {
        item["passage_id"] for item in state.get("evidence", [])
    }

    claims = []

    if "E001" in evidence_ids:
        claims.append({
            "claim_id": "C001",
            "sub_question_id": "SQ01",
            "text": (
                "NovaCell launched two new battery products in 2025."
            ),
            "evidence_ids": ["E001"],
        })

    if "E002" in evidence_ids:
        claims.append({
            "claim_id": "C002",
            "sub_question_id": "SQ02",
            "text": (
                "NovaCell's revenue grew by 12% in 2025."
            ),
            "evidence_ids": ["E002"],
        })
    else:
        # Deliberately unsupported claim to test rejection.
        claims.append({
            "claim_id": "C002",
            "sub_question_id": "SQ02",
            "text": (
                "NovaCell's revenue grew by 40% in 2025."
            ),
            "evidence_ids": ["E001"],
        })

    return {
        "claims": claims,
        "audit_trail": log_event(
            state,
            "synthesis_completed",
            claim_ids=[claim["claim_id"] for claim in claims],
        ),
    }


# --------------------------------------------------
# 4. VERIFIER (DETERMINISTIC TEST VERIFICATION)
# --------------------------------------------------

def verifier_node(state: ResearchState) -> dict:
    evidence_by_id = {
        item["passage_id"]: item
        for item in state.get("evidence", [])
    }

    verdicts = []

    for claim in state.get("claims", []):
        cited_passages = [
            evidence_by_id[evidence_id]
            for evidence_id in claim["evidence_ids"]
            if evidence_id in evidence_by_id
        ]

        supporting_quote = None

        # Test verifier: exact substring support only.
        for passage in cited_passages:
            if claim["text"] in passage["passage_text"]:
                supporting_quote = claim["text"]
                break

        if supporting_quote:
            verdict = "supported"
            reason = "The exact claim appears in the cited passage."
        elif cited_passages:
            verdict = "unsupported"
            reason = (
                "The cited passage does not establish the complete claim."
            )
        else:
            verdict = "unsupported"
            reason = "No valid cited evidence passage was found."

        verdicts.append({
            "claim_id": claim["claim_id"],
            "sub_question_id": claim["sub_question_id"],
            "verdict": verdict,
            "supporting_quote": supporting_quote,
            "reason": reason,
        })

    supported_by_question = {
        sq["id"]: 0 for sq in state.get("sub_questions", [])
    }

    for verdict in verdicts:
        if verdict["verdict"] == "supported":
            sq_id = verdict["sub_question_id"]
            supported_by_question[sq_id] = (
                supported_by_question.get(sq_id, 0) + 1
            )

    # A sub-question needs at least one supported claim.
    needs_research = any(
        count == 0 for count in supported_by_question.values()
    )

    retry_query = ""
    if needs_research:
        missing = [
            sq["question"]
            for sq in state.get("sub_questions", [])
            if supported_by_question.get(sq["id"], 0) == 0
        ]
        retry_query = (
            "Find direct evidence for: " + "; ".join(missing)
        )

    return {
        "verdicts": verdicts,
        "needs_research": needs_research,
        "retry_query": retry_query,
        "audit_trail": log_event(
            state,
            "verification_completed",
            verdicts=verdicts,
            supported_by_question=supported_by_question,
            retry_required=needs_research,
            retry_query=retry_query,
        ),
    }


# --------------------------------------------------
# 5. CONDITIONAL ROUTING AND FINALISATION
# --------------------------------------------------

def route_after_verification(state: ResearchState) -> str:
    if not state.get("needs_research", False):
        return "finalize"

    if state.get("iteration", 0) < state.get("max_retries", 2):
        return "retry"

    return "finalize"


def retry_node(state: ResearchState) -> dict:
    next_iteration = state.get("iteration", 0) + 1

    return {
        "iteration": next_iteration,
        "audit_trail": log_event(
            state,
            "research_retry_started",
            next_iteration=next_iteration,
            query=state.get("retry_query", ""),
        ),
    }


def finalize_node(state: ResearchState) -> dict:
    verdict_by_id = {
        verdict["claim_id"]: verdict
        for verdict in state.get("verdicts", [])
    }

    final_claims = []

    for claim in state.get("claims", []):
        verdict = verdict_by_id.get(claim["claim_id"], {})

        if verdict.get("verdict") == "supported":
            final_claims.append({
                **claim,
                "verdict": "supported",
                "supporting_quote": verdict["supporting_quote"],
            })

    unresolved = [
        sq["id"]
        for sq in state.get("sub_questions", [])
        if not any(
            v["sub_question_id"] == sq["id"]
            and v["verdict"] == "supported"
            for v in state.get("verdicts", [])
        )
    ]

    report = {
        "question": state.get("question", ""),
        "claims": final_claims,
        "unresolved_sub_questions": unresolved,
        "iterations_used": state.get("iteration", 0),
        "status": "complete" if not unresolved else "incomplete",
    }

    return {
        "final_report": report,
        "audit_trail": log_event(
            state,
            "final_report_created",
            status=report["status"],
            supported_claim_count=len(final_claims),
            unresolved_sub_questions=unresolved,
        ),
    }


# --------------------------------------------------
# 6. GRAPH CONSTRUCTION
# --------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "retry": "retry",
            "finalize": "finalize",
        },
    )

    graph.add_edge("retry", "researcher")
    graph.add_edge("finalize", END)

    return graph.compile()