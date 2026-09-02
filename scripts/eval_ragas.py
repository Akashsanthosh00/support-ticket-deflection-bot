"""
RAGAS evaluation: faithfulness, answer relevance, and context relevance for
the retrieval + generation pipeline.

This is deliberately separate from scripts/eval.py, which handles the
project-specific abstention metrics (deflection rate, false-answer rate)
that RAGAS doesn't know how to score.

Usage:
    python -m scripts.eval_ragas
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from app.db import init_db
from app.retrieval import ensure_index_ready
from app.abstention import process_ticket

# Only cases where we expect a real answer make sense for RAGAS — it scores
# the quality of an answer against retrieved context and a ground truth, not
# escalation decisions.
RAG_TEST_CASES = [
    {
        "question": "How do I reset my password?",
        "ground_truth": "Go to the login page, click 'Forgot Password', and use the emailed reset link (expires in 30 minutes).",
    },
    {
        "question": "How do I update my billing information?",
        "ground_truth": "Go to Account Settings > Billing > Payment Methods and edit or add a card.",
    },
]


def build_ragas_dataset() -> Dataset:
    init_db()
    ensure_index_ready()

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for case in RAG_TEST_CASES:
        result = process_ticket(case["question"])
        if result.decision != "answered":
            print(f"[skipped, escalated] {case['question']}")
            continue
        rows["question"].append(case["question"])
        rows["answer"].append(result.answer)
        rows["contexts"].append([s["title"] for s in result.sources])
        rows["ground_truth"].append(case["ground_truth"])

    return Dataset.from_dict(rows)


def run_eval():
    dataset = build_ragas_dataset()
    if len(dataset) == 0:
        print("No answered cases to evaluate — every test case was escalated.")
        return

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print("\n=== RAGAS Eval Results ===\n")
    print(result)


if __name__ == "__main__":
    run_eval()
