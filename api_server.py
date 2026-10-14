"""Optional local HTTP API adapter for other user interfaces.

This file deliberately delegates all claim extraction, web retrieval and NLI
verification to the existing FactAI modules; it does not replace their logic.
Run: .venv/Scripts/python api_server.py
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import json
import os
import tempfile
import traceback

from claim_extractor import extract_claims
from fact_checker import load_nli_model, verify_claim
from summarizer import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from timing import total_seconds
from transcriber import transcribe_media
from web_search import official_office_query, search_web
from ui_config import valid_claim_input

_model = None
ALLOWED_ORIGIN = os.environ.get("FACTAI_ALLOWED_ORIGIN", "http://localhost:5173")


def _nli_model():
    global _model
    if _model is None:
        _model = load_nli_model()
    return _model


def _counterevidence_query(claim: str) -> str | None:
    """Ask a second, neutral query for evidence that could limit absolutes.

    This is retrieval coverage, not a verdict rule: all returned material still
    goes through the existing verifier. It prevents claims such as "zero over
    the entire life cycle" from retrieving only promotional results.
    """
    lower = (claim or "").lower()
    if "electric vehicle" in lower and any(term in lower for term in ("zero", "life cycle", "manufacturing", "emission")):
        return "electric vehicle lifecycle emissions manufacturing electricity generation study"
    if any(term in lower.split() for term in ("all", "always", "never", "only", "zero")):
        return f"{claim} evidence limitations exceptions"
    return None


def check_text(text: str) -> dict:
    extraction_timing = {}
    claims = extract_claims(text, model=DEFAULT_SUMMARY_MODEL, max_claims=3, timing=extraction_timing)
    claims = claims or [text]
    tokenizer, model = _nli_model()
    checked = []
    for claim in claims:
        timing = {}
        sources = [
            {**source, "retrieval_role": "supporting / neutral query"}
            for source in search_web(claim, max_results=5, timing=timing)
        ]
        counter_query = _counterevidence_query(claim)
        if counter_query:
            known_urls = {source.get("url") for source in sources}
            sources.extend(
                {**source, "retrieval_role": "counterevidence query"}
                for source in search_web(counter_query, max_results=4, timing=timing)
                if source.get("url") and source.get("url") not in known_urls
            )
        official_query = official_office_query(claim)
        if official_query:
            known_urls = {source.get("url") for source in sources}
            sources.extend({**source, "retrieval_role": "official-source query"} for source in search_web(official_query, max_results=3, timing=timing)
                           if source.get("url") not in known_urls)
        result = verify_claim(claim, sources, tokenizer, model, summarizer_model=DEFAULT_SUMMARY_MODEL, timing=timing)
        result["elapsed"] = round(total_seconds(timing), 1)
        checked.append(result)
    return {"claims": checked, "elapsed": round(total_seconds(extraction_timing) + sum(item["elapsed"] for item in checked), 1)}


def check_media(filename: str, encoded_content: str) -> dict:
    """Transcribe supported local media, then use the unchanged text pipeline."""
    extension = os.path.splitext(filename)[1].lower()
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg"}
    if extension not in allowed:
        raise ValueError("Upload an audio or video file supported by Whisper.")
    raw = base64.b64decode(encoded_content, validate=True)
    if len(raw) > 100 * 1024 * 1024:
        raise ValueError("Media must be 100 MB or smaller.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as media_file:
        media_file.write(raw)
        media_path = media_file.name
    try:
        transcript = transcribe_media(media_path, model_size="base")
    finally:
        if os.path.exists(media_path):
            os.unlink(media_path)
    if not transcript.strip():
        raise ValueError("No speech could be transcribed from this media file.")
    result = check_text(transcript)
    result["transcript"] = transcript
    return result


class FactAIHandler(BaseHTTPRequestHandler):
    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Friendly status page for users who open the API port in a browser."""
        if self.path in {"/", "/health"}:
            self._json(200, {
                "status": "ok",
                "message": "FactAI API is running. The included interface is Streamlit; run streamlit run app.py in the project folder.",
            })
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path not in {"/api/check", "/api/check-media"}:
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            if self.path == "/api/check-media":
                self._json(200, check_media(str(data.get("filename", "")), str(data.get("content", ""))))
            else:
                text = str(data.get("text", "")).strip()
                if not valid_claim_input(text):
                    self._json(400, {"error": "Enter a complete factual statement to check."})
                    return
                self._json(200, check_text(text))
        except Exception as error:
            traceback.print_exc()
            self._json(500, {"error": f"Fact check failed: {error}"})

    def log_message(self, format, *args):
        print("API", format % args)


if __name__ == "__main__":
    host = os.environ.get("FACTAI_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("FACTAI_PORT", "8000")))
    print(f"FactAI API available at http://{host}:{port}")
    ThreadingHTTPServer((host, port), FactAIHandler).serve_forever()
