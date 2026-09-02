"""
Abstention-specific evaluation: metrics that are specific to this project
(deflection rate, false-answer rate) and aren't something a generic library
like RAGAS scores out of the box. For faithfulness / answer relevance /
context relevance, see scripts/eval_ragas.py instead.

Two things happen here:
  1. Decision accuracy — does the bot's answered/escalated decision match
     the expected label for each labeled test case?
  2. False-answer rate — for cases the bot DID answer, an LLM-as-judge call
     (Claude) checks whether the generated answer actually matches the
     ground-truth resolution. An autonomous answer that's wrong is far more
     costly than an unnecessary escalation, so this is tracked separately
     from plain decision accuracy.

Usage:
    python -m scripts.eval
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db
from app.retrieval import ensure_index_ready
from app.abstention import process_ticket
from app.config import settings

# Each case: the customer message, whether it should be answered or escalated,
# and (for cases expected to be answered) a ground-truth resolution used by
# the LLM-judge to check for false answers.
TEST_CASES = [
    # --- Should be answered: directly covered by an FAQ ---
    {
        "message": "How do I reset my password?",
        "expected": "answered",
        "ground_truth": "Go to the login page, click 'Forgot Password', enter your account email, and use the reset link sent to you (expires in 30 minutes; check spam or request a new one after 5 minutes if it doesn't arrive).",
    },
    {
        "message": "How can I export my data as a CSV?",
        "expected": "answered",
        "ground_truth": "Go to Account Settings > Data > Export and choose CSV or JSON. Exports over 100,000 records can take up to 24 hours; you'll get an email when it's ready.",
    },
    {
        "message": "What browsers does your app support?",
        "expected": "answered",
        "ground_truth": "The latest two versions of Chrome, Firefox, Safari, and Edge are supported. Internet Explorer is not supported.",
    },
    {
        "message": "How do I invite a new team member?",
        "expected": "answered",
        "ground_truth": "Go to Team Settings > Members > Invite, enter their email, and pick a role (Admin, Editor, or Viewer). The invite email expires after 7 days.",
    },
    {
        "message": "How do I turn on two-factor authentication?",
        "expected": "answered",
        "ground_truth": "Go to Account Settings > Security > Two-Factor Authentication, click Enable, scan the QR code with an authenticator app, and confirm with the 6-digit code. Save the backup codes.",
    },
    {
        "message": "How do I update my credit card on file?",
        "expected": "answered",
        "ground_truth": "Go to Account Settings > Billing > Payment Methods and edit the existing card or add a new one. Changes apply to new charges immediately.",
    },
    {
        "message": "What's your subscription cancellation policy?",
        "expected": "answered",
        "ground_truth": "Cancel anytime from Account Settings > Subscription > Cancel Plan. Access continues until the end of the current billing period; refunds for early cancellation aren't offered by default (exceptions handled by support case-by-case).",
    },
    {
        "message": "How do I upgrade my plan?",
        "expected": "answered",
        "ground_truth": "Go to Account Settings > Subscription > Change Plan. Upgrades apply immediately with prorated billing; downgrades apply at the next billing cycle.",
    },
    {
        "message": "Do you have an API I can use?",
        "expected": "answered",
        "ground_truth": "Yes, a REST API is available on Pro and Enterprise plans. Generate a key from Account Settings > Developer > API Keys. Rate limits are 1,000 req/hr on Pro and 10,000 req/hr on Enterprise.",
    },
    {
        "message": "Why is my dashboard taking so long to load?",
        "expected": "answered",
        "ground_truth": "Usually caused by wide date ranges or too many active filters. Narrow the date range to 30 days or fewer, reduce active filters, and try clearing the browser cache.",
    },
    # A paraphrase of an existing FAQ, worded differently than the source doc —
    # tests whether retrieval matches on meaning, not just keyword overlap.
    {
        "message": "I want to add a coworker who can only view reports, not edit anything.",
        "expected": "answered",
        "ground_truth": "Invite them from Team Settings > Members > Invite and assign the 'Viewer' role, which grants read-only access to reports and dashboards.",
    },
    # Generic, non-account-specific troubleshooting advice from a past ticket —
    # tests that useful guidance in resolved tickets (not just FAQs) gets used.
    {
        "message": "My integration suddenly started getting 429 errors, what should I do?",
        "expected": "answered",
        "ground_truth": "429 errors usually mean you're hitting your plan's rate limit. Batch requests, add exponential backoff, or upgrade to a higher-limit plan if you need more throughput.",
    },

    # --- Should be escalated: forced keyword topics ---
    {"message": "I want a refund for my subscription.", "expected": "escalated", "ground_truth": None},
    {"message": "I'm going to sue you if this isn't fixed.", "expected": "escalated", "ground_truth": None},
    {"message": "There's an unauthorized charge on my card, please cancel my account.", "expected": "escalated", "ground_truth": None},
    {"message": "Please delete my data under GDPR.", "expected": "escalated", "ground_truth": None},
    {"message": "I need a chargeback for last month's payment.", "expected": "escalated", "ground_truth": None},
    {"message": "This is becoming a legal matter, please connect me with your legal team.", "expected": "escalated", "ground_truth": None},
    # Forced keyword buried in an otherwise mild, polite message — tests that
    # keyword-forcing overrides tone/sentiment rather than only firing on
    # obviously angry messages.
    {"message": "Hi! Quick question — what's the process to cancel my plan, and are refunds possible?", "expected": "escalated", "ground_truth": None},

    # --- Should be escalated: frustration/negative sentiment ---
    {"message": "This is absolutely ridiculous, your app is garbage and I'm furious, nothing works and I've wasted hours.", "expected": "escalated", "ground_truth": None},
    {"message": "I am extremely angry, this is the worst customer service I have ever experienced.", "expected": "escalated", "ground_truth": None},
    {"message": "I've been trying to fix this for three days straight and I'm about to cancel everything. Absolutely terrible experience.", "expected": "escalated", "ground_truth": None},

    # --- Should be escalated: out of scope / not in knowledge base ---
    {"message": "Can you tell me the exact amount of my last invoice and when it was charged?", "expected": "escalated", "ground_truth": None},
    {"message": "What's the weather like today?", "expected": "escalated", "ground_truth": None},
    {"message": "Can you write me a poem about clouds?", "expected": "escalated", "ground_truth": None},
    {"message": "What's your company's revenue this year?", "expected": "escalated", "ground_truth": None},

    # --- Should be escalated: HARD cases — topically similar to something in
    # the knowledge base, but the matching past ticket itself required human
    # escalation (billing disputes, identity verification), so a confident
    # autonomous answer here would be wrong even though retrieval will likely
    # find a related document. These are the cases that actually test whether
    # the abstention logic is doing real work, not just topic-matching. ---
    {"message": "I see two charges on my card this billing cycle for the same plan.", "expected": "escalated", "ground_truth": None},
    {"message": "I lost my phone along with my 2FA backup codes and I'm locked out of my account.", "expected": "escalated", "ground_truth": None},
]

JUDGE_PROMPT = """You are grading a customer support bot's answer against a \
known-correct resolution. Respond with ONLY "CORRECT" or "WRONG" (nothing else).

Question: {question}
Ground-truth resolution: {ground_truth}
Bot's answer: {answer}

Does the bot's answer correctly and accurately match the ground-truth \
resolution (minor wording differences are fine, factual errors are not)?"""


def judge_answer(question: str, ground_truth: str, answer: str) -> bool:
    """LLM-as-judge check, using whichever provider is configured (see LLM_PROVIDER in .env)."""
    prompt = JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer)

    if settings.LLM_PROVIDER == "anthropic":
        from app.llm import _get_anthropic_client
        client = _get_anthropic_client()
        resp = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    else:
        from app.llm import _get_gemini_client
        client = _get_gemini_client()
        resp = client.models.generate_content(model=settings.GEMINI_MODEL, contents=prompt)
        verdict = resp.text or ""

    return "CORRECT" in verdict.upper()


def run_eval():
    init_db()
    ensure_index_ready()

    correct = 0
    false_answers = 0
    judged = 0
    results = []

    for case in TEST_CASES:
        result = process_ticket(case["message"])
        is_correct = result.decision == case["expected"]
        correct += int(is_correct)

        is_false_answer = None
        if result.decision == "answered" and case.get("ground_truth"):
            try:
                judged += 1
                is_false_answer = not judge_answer(case["message"], case["ground_truth"], result.answer)
                false_answers += int(is_false_answer)
            except Exception as e:
                judged -= 1
                is_false_answer = None
                print(f"    [judge call failed, skipping false-answer check for this case: {e}]")

        results.append({
            "message": case["message"],
            "expected": case["expected"],
            "actual": result.decision,
            "correct": is_correct,
            "false_answer": is_false_answer,
            "reasons": result.escalation_reasons,
        })

    print("\n=== Deflection Bot Eval Results ===\n")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        flag = " [FALSE ANSWER]" if r["false_answer"] else ""
        print(f"{mark} [{r['expected']:>9} -> {r['actual']:>9}]{flag} {r['message'][:70]}")
        if not r["correct"]:
            print(f"    reasons: {r['reasons']}")

    accuracy = correct / len(TEST_CASES)
    deflection_rate = sum(1 for r in results if r["actual"] == "answered") / len(TEST_CASES)
    false_answer_rate = (false_answers / judged) if judged else None

    print(f"\nDecision accuracy: {correct}/{len(TEST_CASES)} ({accuracy*100:.1f}%)")
    print(f"Deflection rate:   {deflection_rate*100:.1f}%")
    if false_answer_rate is not None:
        print(f"False-answer rate: {false_answers}/{judged} judged answers ({false_answer_rate*100:.1f}%)")
    else:
        print("False-answer rate: no ground_truth provided for any answered case — add some to TEST_CASES")


if __name__ == "__main__":
    run_eval()
