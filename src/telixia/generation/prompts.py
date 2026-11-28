"""Grounded-answer prompt: numbered sources, mandatory citations, explicit refusal.

The LLM only sees the sources retrieved for the question. Every factual sentence must
cite the source number it comes from ([1], [2]...). If the sources do not answer the
question, the model must reply with a fixed refusal sentence: it is detected in code
(no hallucinated answer reaches the analyst) and counted in the evaluation.
"""

from __future__ import annotations

import re

REFUSAL_FR = "Je ne trouve pas cette information dans les documents Orange dont je dispose."
REFUSAL_EN = "I could not find this information in the Orange documents available to me."
LANGUAGE_NAMES = {"fr": "French", "en": "English"}

SYSTEM_PROMPT = """You are Telixia, an assistant for the marketing and CRM analysts of Orange France.
You answer questions about Orange offers, prices and contract terms using ONLY the numbered sources in the user message.

Rules:
1. Use only facts written in the sources. Never use outside knowledge. Never guess a price, a duration, a fee or a condition.
2. Cite every factual sentence with the number of its source in square brackets, for example [1] or [2][3]. Only cite numbers that exist in the sources.
3. If the sources do not contain the answer, reply with this exact sentence and nothing else:
   - question in French: "{refusal_fr}"
   - question in English: "{refusal_en}"
4. If the sources answer only part of the question, give that part with its citations, then say clearly what is missing.
5. Several sources may describe different offers. Always name the offer a fact belongs to. If sources disagree, say so and cite each one.
6. Answer in {language}. Be concise: at most 6 sentences, or a short bullet list.
"""


def system_prompt(language: str) -> str:
    return SYSTEM_PROMPT.format(refusal_fr=REFUSAL_FR, refusal_en=REFUSAL_EN,
                                language=LANGUAGE_NAMES.get(language, "French"))


def user_prompt(question: str, sources: list[dict]) -> str:
    """sources: [{"n": 1, "title": ..., "section": ..., "text": ...}, ...]"""
    blocks = [f"[{s['n']}] {s['title']} | {s['section']}\n{s['text'].strip()}" for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


CITATION = re.compile(r"\[(\d+)\]")


def check_answer(answer: str, n_sources: int) -> dict:
    """Compliance check on the generated answer (no LLM involved).

    status: refusal          the model said the corpus does not cover the question
            ok               at least one citation, all of them point to a real source
            invalid_citation a cited number does not exist (e.g. [7] with 5 sources)
            no_citation      factual answer without any citation (not verifiable)
    """
    cited = sorted({int(n) for n in CITATION.findall(answer)})
    invalid = [n for n in cited if not 1 <= n <= n_sources]
    text = " ".join(answer.split()).lower()
    if REFUSAL_FR.lower() in text or REFUSAL_EN.lower() in text:
        status = "refusal"
    elif invalid:
        status = "invalid_citation"
    elif not cited:
        status = "no_citation"
    else:
        status = "ok"
    return {"status": status, "cited": cited, "invalid": invalid}