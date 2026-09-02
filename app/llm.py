"""
Wraps calls to an LLM for answer generation. Supports two providers behind
the same interface (set LLM_PROVIDER in .env):

  - "gemini"    (default) — Google's Gemini API, free tier (Flash models),
                 no payment method required. Good for development/testing.
  - "anthropic" — Claude API, paid. Swap to this for final/production runs.

Both providers get the same prompt and are asked to do two things at once:
1. Answer the customer's question using ONLY the provided context (grounded,
   no fabrication).
2. Self-report a confidence score (0-1) and whether the context was
   sufficient to answer at all.

This self-rated confidence is a second (independent) abstention signal on
top of the raw retrieval similarity score — together they catch cases where
retrieval looks fine on paper but the content doesn't actually answer the
question.
"""
import json
import re

from app.config import settings
from app.retrieval import RetrievedItem

SYSTEM_PROMPT = """You are a customer support assistant. You answer customer \
questions using ONLY the context provided below (a mix of FAQ articles and \
past resolved support tickets). You must never invent information that is \
not present in the context.

Respond with ONLY a JSON object (no markdown fences, no preamble) with this \
exact shape:
{
  "can_answer": true or false,
  "confidence": a number from 0.0 to 1.0,
  "answer": "the answer text, written directly to the customer" or null if can_answer is false,
  "reasoning": "one short sentence on why you can or cannot answer confidently from the context"
}

Rules:
- Set "can_answer" to false if the context does not contain enough information \
to confidently and accurately answer the question.
- Set "confidence" low (below 0.5) if you are guessing, extrapolating beyond \
the context, or the context only partially covers the question.
- Set "confidence" high (above 0.8) only if the context directly and \
completely answers the question.
- Never answer questions about account-specific data you don't have (exact \
charges, specific account status, personal data) — treat these as low confidence.
- Keep answers concise, friendly, and directly actionable.
"""


def _format_context(items: list[RetrievedItem]) -> str:
    if not items:
        return "(no relevant context found)"
    blocks = []
    for it in items:
        label = "FAQ" if it.source_type == "faq" else "Past Ticket"
        blocks.append(f"[{label} — {it.title} (similarity: {it.score:.2f})]\n{it.text}")
    return "\n\n---\n\n".join(blocks)


def _extract_json(text: str) -> dict:
    """LLMs should return raw JSON, but strip markdown fences defensively."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


# --- Anthropic (Claude) backend ---
_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic_client


def _call_anthropic(user_prompt: str) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


# --- Gemini backend (default — free tier, no payment method needed) ---
_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _call_gemini(user_prompt: str) -> str:
    from google.genai import types

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=800,
            response_mime_type="application/json",
        ),
    )
    return response.text


_PROVIDERS = {
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
}


def generate_answer(customer_message: str, context_items: list[RetrievedItem]) -> dict:
    """
    Calls the configured LLM provider to attempt an answer. Returns a dict:
    {can_answer, confidence, answer, reasoning}
    Falls back to a safe "cannot answer" response if the API call fails or
    the response can't be parsed, so a bad API call never means a bad autonomous answer.
    """
    context_str = _format_context(context_items)
    user_prompt = f"Context:\n{context_str}\n\nCustomer question:\n{customer_message}"

    try:
        call_fn = _PROVIDERS.get(settings.LLM_PROVIDER)
        if call_fn is None:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}' — expected 'anthropic' or 'gemini'"
            )
        raw_text = call_fn(user_prompt)
        parsed = _extract_json(raw_text)

        return {
            "can_answer": bool(parsed.get("can_answer", False)),
            "confidence": float(parsed.get("confidence", 0.0)),
            "answer": parsed.get("answer"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        # Fail safe: any error in the LLM call/parse means we abstain, not guess.
        return {
            "can_answer": False,
            "confidence": 0.0,
            "answer": None,
            "reasoning": f"LLM call failed, defaulting to escalation: {e}",
        }


def get_client():
    """
    Returns the active provider's raw client — used by scripts/eval.py for
    the LLM-as-judge false-answer-rate check, which needs direct access
    regardless of which provider is configured.
    """
    if settings.LLM_PROVIDER == "anthropic":
        return _get_anthropic_client()
    return _get_gemini_client()