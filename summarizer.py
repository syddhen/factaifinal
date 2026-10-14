"""
summarizer.py

Generates a short, natural-language explanation of a claim's verdict
using a local, free LLM served by Ollama (https://ollama.com).

No API key, no paid service, nothing sent to a third party: Ollama
runs entirely on your own machine. If Ollama isn't installed or isn't
running, `generate_summary` falls back to a plain templated sentence
built from the evidence counts, so the app still works without it --
just with a less natural-sounding summary.

Setup (optional but recommended):
    1. Install Ollama: https://ollama.com/download
    2. Pull a small model once:   ollama pull llama3.2
    3. Leave Ollama running in the background (it starts a local
       server at http://localhost:11434 automatically).
"""

import requests
from timing import timed

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PING_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
REQUEST_TIMEOUT = 30
PING_TIMEOUT = 1.5

STANCE_LABEL = {
    "entailment": "supports",
    "contradiction": "contradicts",
    "neutral": "is neutral toward",
}


def _ollama_available(timing=None) -> bool:
    try:
        with timed(timing, "ollama_ping"):
            requests.get(OLLAMA_PING_URL, timeout=PING_TIMEOUT)
        return True
    except Exception:
        return False


def _fallback_summary(
    verdict: str, support_count: int, refute_count: int,
    claim: str = "", evidence: list = None,
) -> str:
    evidence = evidence or []
    if verdict == "False":
        excerpt = next(
            ((item.get("excerpt") or "").strip() for item in evidence
             if item.get("stance") == "contradiction" and item.get("excerpt")),
            "",
        )
        if excerpt:
            if len(excerpt) > 220:
                excerpt = excerpt[:220].rsplit(" ", 1)[0]
            return f'The claim says “{claim}”, but the retrieved text says “{excerpt}”.'
    if verdict == "True":
        return "Multiple sources corroborate this claim and none of the top results contradict it."
    if verdict == "False":
        return "Multiple sources directly contradict this claim."
    if verdict == "Misleading":
        return "Sources are split — some support this claim while others contradict it, so it appears only partly accurate."
    return "No source in the top search results could reliably confirm or deny this claim."


def generate_summary(claim: str, verdict: str, evidence: list, model: str = DEFAULT_MODEL, timing=None) -> str:
    """
    claim: the claim text
    verdict: "True" | "False" | "Misleading" | "Unverified"
    evidence: list of dicts with keys "stance", "title", "excerpt"
    model: the local Ollama model name to use

    Returns a short (<35 word) plain-English sentence explaining the verdict.
    Grounded strictly in the retrieved evidence -- the model is not asked
    to use outside knowledge, only to phrase the reasoning naturally.
    """
    support_count = sum(1 for e in evidence if e["stance"] == "entailment")
    refute_count = sum(1 for e in evidence if e["stance"] == "contradiction")

    if not _ollama_available(timing):
        return _fallback_summary(verdict, support_count, refute_count, claim, evidence)

    evidence_lines = []
    for e in evidence[:6]:
        stance = STANCE_LABEL.get(e["stance"], "is neutral toward")
        excerpt = (e.get("excerpt") or "").strip()
        if not excerpt:
            continue
        evidence_lines.append(f'- This source {stance} the claim: "{excerpt}" ({e.get("title", "source")})')
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "No relevant excerpts were retrieved."

    prompt = f"""You are a neutral fact-checker. Using ONLY the retrieved excerpts below -- no outside knowledge -- write ONE concise sentence explaining the verdict. Directly compare the claim with the specific fact in the excerpts; for a False or Misleading verdict, state what fact contradicts or qualifies the claim. Preserve relevant dates and numbers. Do not invent missing details. Output only the sentence, nothing else, no quotation marks.

Claim: "{claim}"
Verdict: {verdict}

What was found:
{evidence_block}

Sentence:"""

    try:
        with timed(timing, "summary_llm"):
            resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
            )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip().strip('"').strip()
        if text:
            return text
    except Exception:
        pass

    return _fallback_summary(verdict, support_count, refute_count, claim, evidence)