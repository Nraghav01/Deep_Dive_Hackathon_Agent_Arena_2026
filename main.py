import json
from pathlib import Path

from src.graph import build_graph


def main():
    app = build_graph()

    initial_state = {
        "question": (
            "What products did NovaCell launch, "
            "and what was its revenue growth in 2025?"
        ),
        "sub_questions": [],
        "evidence": [],
        "claims": [],
        "verdicts": [],
        "iteration": 0,
        "max_retries": 2,
        "retry_query": "",
        "needs_research": False,
        "audit_trail": [],
    }

    result = app.invoke(initial_state)

    print("\n=== FINAL REPORT ===")
    print(json.dumps(result["final_report"], indent=2))

    print("\n=== FINAL VERDICTS ===")
    print(json.dumps(result["verdicts"], indent=2))

    print("\n=== AUDIT TRAIL ===")
    print(json.dumps(result["audit_trail"], indent=2))

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "demo_run.json"
    output_file.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"\nFull run saved to: {output_file}")


if __name__ == "__main__":
    main()