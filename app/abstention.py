"""
Core abstention decision logic.

Combines THREE independent signals to decide whether the bot should answer
autonomously or escalate to a human:

  1. Retrieval confidence  — is the top similarity score from the knowledge
     base high enough that we likely have relevant grounding at all?
  2. Forced-escalation keywords/topics — certain topics (refunds, legal,
     fraud, account deletion, etc.) ALWAYS go to a human regardless of how
     confident the model is, because getting them wrong is costly.
  3. Sentiment/frustration — a sufficiently negative sentiment score routes
     to a human even if the bot could technically answer the factual
     question, since frustrated customers benefit from human empathy/judgment.
  4. Model self-rated answer confidence — even if retrieval + keywords +
     sentiment pass, if Claude itself says it can't answer confidently from
     the given context, we still escalate.

Escalation happens if ANY of these signals fire — this is intentionally
conservative (favors precision of auto-answers over deflection rate), which
is the safer default for a support bot. Thresholds are tunable in .env.
"""
from dataclasses import dataclass, field

from app.config import settings
from app.retrieval import RetrievedItem, knowledge_index
from app.sentiment import get_sentiment_score
from app.llm import generate_answer


@dataclass
class DeflectionResult:
    decision: str  # 'answered' | 'escalated'
    answer: str | None
    escalation_reasons: list[str] = field(default_factory=list)
    answer_confidence: float | None = None
    retrieval_top_score: float | None = None
    sentiment_score: float | None = None
    matched_keyword: str | None = None
    sources: list[dict] = field(default_factory=list)


def check_forced_keywords(text: str) -> str | None:
    """Returns the first matched keyword/topic that forces escalation, or None."""
    lowered = text.lower()
    for kw in settings.FORCE_ESCALATE_KEYWORDS:
        if kw in lowered:
            return kw
    return None


def process_ticket(customer_message: str) -> DeflectionResult:
    reasons: list[str] = []

    # --- Signal 1: forced keyword/topic check (cheapest check, run first) ---
    matched_keyword = check_forced_keywords(customer_message)
    if matched_keyword:
        reasons.append(f"forced escalation topic matched: '{matched_keyword}'")

    # --- Signal 2: sentiment/frustration check ---
    sentiment_score = get_sentiment_score(customer_message)
    if sentiment_score <= settings.FRUSTRATION_SENTIMENT_THRESHOLD:
        reasons.append(
            f"customer sentiment too negative ({sentiment_score:.2f} <= "
            f"{settings.FRUSTRATION_SENTIMENT_THRESHOLD})"
        )

    # --- Signal 3: retrieval confidence check ---
    retrieved: list[RetrievedItem] = knowledge_index.search(customer_message)
    top_score = retrieved[0].score if retrieved else 0.0
    if top_score < settings.MIN_RETRIEVAL_SIMILARITY:
        reasons.append(
            f"low retrieval similarity ({top_score:.2f} < "
            f"{settings.MIN_RETRIEVAL_SIMILARITY})"
        )

    # If keyword or sentiment already forced escalation, skip the LLM call
    # entirely — no need to spend a generation call we're going to discard.
    # We still generate an answer if only retrieval was borderline, since a
    # human reviewer benefits from seeing what the bot *would* have said.
    llm_result = None
    if not matched_keyword and sentiment_score > settings.FRUSTRATION_SENTIMENT_THRESHOLD:
        llm_result = generate_answer(customer_message, retrieved)

        if not llm_result["can_answer"]:
            reasons.append(f"model declined to answer: {llm_result['reasoning']}")
        elif llm_result["confidence"] < settings.MIN_ANSWER_CONFIDENCE:
            reasons.append(
                f"model confidence too low ({llm_result['confidence']:.2f} < "
                f"{settings.MIN_ANSWER_CONFIDENCE})"
            )

    sources = [
        {"id": r.source_id, "type": r.source_type, "title": r.title, "score": round(r.score, 4)}
        for r in retrieved
    ]

    if reasons:
        return DeflectionResult(
            decision="escalated",
            answer=None,
            escalation_reasons=reasons,
            answer_confidence=llm_result["confidence"] if llm_result else None,
            retrieval_top_score=top_score,
            sentiment_score=sentiment_score,
            matched_keyword=matched_keyword,
            sources=sources,
        )

    return DeflectionResult(
        decision="answered",
        answer=llm_result["answer"],
        escalation_reasons=[],
        answer_confidence=llm_result["confidence"],
        retrieval_top_score=top_score,
        sentiment_score=sentiment_score,
        matched_keyword=None,
        sources=sources,
    )
