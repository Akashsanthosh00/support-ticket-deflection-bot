"""
Ablation study: hybrid retrieval + cross-encoder reranking vs. a dense-only
baseline with no reranking, on the same eval set.

This answers the "did adding a reranker actually help?" question with a real
before/after number, instead of a single unqualified accuracy figure. It's
the source for any "improved X by Y%" resume claim about the retrieval
architecture — and because it runs the same 28 cases both ways, you can look
at exactly which cases flipped and explain why.

Only checks decision accuracy (answered/escalated correctness) — not answer
quality/false-answer rate, which is a separate concern already covered by
scripts/eval.py. This keeps API usage roughly half of running eval.py twice.

Usage:
    python -m scripts.eval_ablation
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db
from app.retrieval import ensure_index_ready, knowledge_index
from app.abstention import process_ticket
from scripts.eval import TEST_CASES


def run_condition(rerank_enabled: bool) -> dict:
    knowledge_index.rerank_enabled = rerank_enabled
    label = "WITH reranking" if rerank_enabled else "WITHOUT reranking (dense-only baseline)"
    print(f"\n--- Running: {label} ---")

    correct = 0
    results = []
    for case in TEST_CASES:
        result = process_ticket(case["message"])
        is_correct = result.decision == case["expected"]
        correct += int(is_correct)
        results.append({
            "message": case["message"],
            "expected": case["expected"],
            "actual": result.decision,
            "correct": is_correct,
            "retrieval_top_score": result.retrieval_top_score,
        })
        mark = "✓" if is_correct else "✗"
        print(f"{mark} [{case['expected']:>9} -> {result.decision:>9}] (score={result.retrieval_top_score:.3f}) {case['message'][:60]}")

    accuracy = correct / len(TEST_CASES)
    print(f"\n{label}: {correct}/{len(TEST_CASES)} correct ({accuracy*100:.1f}%)")
    return {"label": label, "accuracy": accuracy, "correct": correct, "results": results}


def run_ablation():
    init_db()
    ensure_index_ready()

    with_rerank = run_condition(rerank_enabled=True)
    without_rerank = run_condition(rerank_enabled=False)
    knowledge_index.rerank_enabled = True  # restore default before exiting

    print("\n" + "=" * 60)
    print("=== Ablation Summary ===")
    print("=" * 60)
    print(f"{with_rerank['label']}:    {with_rerank['correct']}/{len(TEST_CASES)} ({with_rerank['accuracy']*100:.1f}%)")
    print(f"{without_rerank['label']}: {without_rerank['correct']}/{len(TEST_CASES)} ({without_rerank['accuracy']*100:.1f}%)")

    diff_points = (with_rerank["accuracy"] - without_rerank["accuracy"]) * 100
    print(f"\nDecision accuracy change from adding cross-encoder reranking: {diff_points:+.1f} percentage points")

    # Show exactly which cases flipped — this is what makes the number defensible
    print("\n--- Cases that differ between the two conditions ---")
    flipped = 0
    for a, b in zip(with_rerank["results"], without_rerank["results"]):
        if a["correct"] != b["correct"]:
            flipped += 1
            print(f"  \"{a['message'][:70]}\"")
            print(f"    WITH reranking:    {'✓ correct' if a['correct'] else '✗ wrong'} (retrieval score {a['retrieval_top_score']:.3f})")
            print(f"    WITHOUT reranking: {'✓ correct' if b['correct'] else '✗ wrong'} (retrieval score {b['retrieval_top_score']:.3f})")
    if flipped == 0:
        print("  (none — both conditions produced identical decisions on this eval set)")


if __name__ == "__main__":
    run_ablation()