"""
app.py

"FactAI" -- a local claim checker styled to match the
reference design. Uses free, local tools throughout:
  - Whisper for audio/video transcription
  - spaCy heuristics for claim extraction
  - DuckDuckGo search (free, keyless)
  - A local NLI model to weigh evidence
  - A local Ollama LLM (optional) to write natural-language summaries

Run with: streamlit run app.py
"""

import os
import hashlib
import tempfile
import time
from urllib.parse import urlparse

import streamlit as st

from transcriber import transcribe_media
from claim_extractor import extract_claims
from web_search import search_web, official_office_query
from fact_checker import verify_claim, load_nli_model, is_tenure_duration_claim
from summarizer import DEFAULT_MODEL as DEFAULT_SUMMARY_MODEL
from agent_checker import investigate_claim, DEFAULT_MODEL as DEFAULT_AGENT_MODEL
from timing import total_seconds
from ui_config import history_prompt, verdict_tone, valid_claim_input, MIN_CLAIM_LENGTH

st.set_page_config(page_title="FactAI — evidence you can inspect", page_icon="F", layout="wide")

EXAMPLE = (
    "In her speech, the senator said renewable energy now supplies more than "
    "half of all U.S. electricity generation, and that national unemployment "
    "has fallen below 2% for the first time since World War II."
)

STAGE_LABELS = [
    "Reading the text",
    "Pulling out claims",
    "Searching the web",
    "Weighing the evidence",
]

VERDICT_STYLE = {
    "True": {"color": "#2E7D5B", "rotate": "-3deg"},
    "False": {"color": "#B4423B", "rotate": "4deg"},
    "Misleading": {"color": "#C08A2E", "rotate": "-2deg"},
    "Unverified": {"color": "#767A82", "rotate": "3deg"},
}


def hostname(url: str) -> str:
    try:
        h = urlparse(url).hostname or url
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return url


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');

:root {
  --paper: #F0FDFA;
  --paper-2: #FFFFFF;
  --ink: #123B39;
  --ink-soft: #526B69;
  --line: #BCE9E2;
  --accent: #0F766E;
}

.stApp { background: var(--paper); }
.block-container { max-width: 1080px; padding-top: 32px; padding-bottom: 64px; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

/* Hide Streamlit's own chrome so only the paper design shows */
header[data-testid="stHeader"] { background: var(--paper); box-shadow: none; }
#MainMenu, footer, div[data-testid="stDecoration"] { visibility: hidden; height: 0; }
div[data-testid="stToolbar"] { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--paper-2);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--ink); }
section[data-testid="stSidebar"] h2 { font-family: 'Source Serif 4', serif; font-size: 20px; }

/* Radio (Text / Audio-Video toggle) */
div[data-testid="stRadio"] label {
  background: var(--paper-2);
  border: 1px solid var(--line);
  padding: 6px 14px;
  margin-right: 8px;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
  border-color: var(--ink-soft) !important;
}
div[data-testid="stRadio"] input:checked + div {
  background-color: var(--accent) !important;
  border-color: var(--accent) !important;
}

/* Sliders */
div[data-testid="stSlider"] [role="slider"] {
  background-color: var(--accent) !important;
  border-color: var(--accent) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
  background: var(--accent) !important;
}

/* Selectbox / text input */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input {
  background: var(--paper-2) !important;
  border-color: var(--line) !important;
  border-radius: 0 !important;
  color: var(--ink) !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
  background: var(--paper-2) !important;
  border: 1px dashed var(--line) !important;
  border-radius: 0 !important;
}
section[data-testid="stFileUploaderDropzone"] button {
  border-radius: 0 !important;
  border-color: var(--line) !important;
  color: var(--ink) !important;
  background: var(--paper) !important;
}

/* Default horizontal rule + caption, restyled to match the palette */
hr { border-color: var(--line) !important; margin: 32px 0 !important; }
div[data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; }
div[data-testid="stAlert"] { border-radius: 0; }

.desk-title {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: clamp(32px, 4vw, 48px);
  margin: 0 0 6px 0;
  letter-spacing: -0.01em;
  color: var(--ink);
}
.desk-tagline {
  font-size: 17px;
  color: var(--ink-soft);
  margin: 0 0 20px 0;
  max-width: 58ch;
  line-height: 1.5;
}
.brand-kicker { color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 0; }
.trust-strip { display:flex; flex-wrap:wrap; gap:8px; margin: 20px 0 28px; }
.trust-pill { border:1px solid var(--line); background:#fff; padding:6px 10px; border-radius:999px; color:var(--ink-soft); font-size:12px; }
.workspace { background:rgba(255,255,255,.76); border:1px solid var(--line); border-radius:18px; padding:20px; }
.section-label { font-size:13px; font-weight:700; color:var(--ink); margin-bottom:5px; }

div[data-testid="stTextArea"] textarea {
  background: var(--paper-2);
  border: 1px solid var(--line);
  color: var(--ink);
  font-family: 'Inter', sans-serif;
  font-size: 15px;
  line-height: 1.55;
  border-radius: 12px;
}
div[data-testid="stTextArea"] textarea:focus { border-color: var(--accent); box-shadow: none; }

div[data-testid="stButton"] button,
div[data-testid="stPopover"] button {
  border-radius: 10px; font-family: 'Inter', sans-serif; font-size: 14px;
}
div[data-testid="stButton"] button[kind="primary"] {
  background: var(--accent); color: #fff; border: none;
  padding: 9px 16px; font-weight: 700;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background: #17453F; color: #F3F1E9; }
div[data-testid="stButton"] button[kind="primary"]:disabled { background: #B9B6AA; color: #F3F1E9; }
div[data-testid="stButton"] button[kind="secondary"] {
  background: none; border: none; color: var(--ink-soft);
  text-decoration: underline; text-underline-offset: 2px; padding: 4px 0;
}
div[data-testid="stButton"] button[kind="secondary"]:hover { color: var(--ink); }

.hint-line { font-size: 13px; color: #A24A3E; margin: 6px 2px 0; }
.status-line { margin-top: 18px; font-size: 14px; color: var(--ink-soft); font-style: italic; }
.error-line { margin-top: 18px; font-size: 14px; color: #B4423B; }
.empty-note { margin-top: 16px; font-size: 13px; color: var(--ink-soft); line-height: 1.5; }

.results-header {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-top: 28px; margin-bottom: 4px;
}
.results-count { font-size: 14px; color: var(--ink-soft); }

.claim-card {
  background:#fff; border: 1px solid var(--line); border-radius: 14px;
  padding: 22px; margin: 14px 0;
  animation: card-in 0.4s ease both;
}
.claim-card.last { margin-bottom: 0; }
.claim-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; }
.claim-text {
  font-family: 'Source Serif 4', serif;
  font-size: 19px; line-height: 1.4; margin: 0; flex: 1; color: var(--ink);
}
.stamp {
  flex-shrink: 0; border: 2px solid; padding: 4px 10px;
  font-family: 'Inter', sans-serif; font-weight: 600; font-size: 12px;
  letter-spacing: 0.08em; text-transform: uppercase;
  transform: rotate(var(--r));
  animation: stamp-in 0.5s cubic-bezier(0.2, 0.9, 0.3, 1.2) both;
}
.claim-summary { font-size: 14.5px; line-height: 1.6; color: var(--ink-soft); margin: 10px 0 0; max-width: 62ch; }
.claim-meta { display: flex; align-items: center; gap: 8px; margin: 8px 0 0; }
.claim-strength { font-size: 12.5px; color: #8A8779; margin: 0; }
.claim-timing { font-size: 12.5px; color: #B9B6AA; margin: 0; }
.claim-timing::before { content: "\00b7"; margin-right: 8px; }
.claim-sources { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.source-chip {
  display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px;
  color: var(--accent); text-decoration: none; border: 1px solid var(--line);
  padding: 4px 9px;
}
.source-chip:hover { border-color: var(--accent); }

@keyframes stamp-in {
  from { opacity: 0; transform: rotate(var(--r)) scale(1.8); }
  to { opacity: 1; transform: rotate(var(--r)) scale(1); }
}
@keyframes card-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
  .stamp, .claim-card { animation: none !important; }
}
/* Reference-inspired cinematic shell */
.stApp { background: radial-gradient(circle at 78% 55%, rgba(255,121,51,.88), transparent 26%), linear-gradient(150deg, #06133a 0%, #16367a 35%, #743d6c 62%, #ff7b35 83%, #17233c 100%) !important; min-height:100vh; }
.stApp:before { content:""; position:fixed; inset:0; pointer-events:none; background:linear-gradient(180deg,rgba(3,9,35,.08),rgba(2,7,28,.56)); z-index:0; }
.block-container { position:relative; z-index:1; max-width:980px; padding-top:18px; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(5,17,52,.97),rgba(3,9,29,.96)) !important; border-right:1px solid rgba(255,255,255,.22) !important; }
section[data-testid="stSidebar"] * { color:#F8FAFC !important; }
.fact-logo { font-size:29px; font-weight:800; letter-spacing:.02em; color:#fff; margin:0 0 2px; }.fact-logo b{color:#FF7A24}.fact-sub{font-size:13px;color:#CFD7EC;margin:0 0 24px}.hero-title{font-size:36px;font-weight:500;text-align:center;color:#fff;margin:128px 0 24px}.hero-title b{color:#FF7A24}.navline{display:flex;justify-content:space-between;align-items:center;color:#fff;font-size:14px;margin:0 0 20px}.navlinks{display:flex;gap:44px}.navlinks span:first-child{border-bottom:2px solid #FF7A24;padding-bottom:12px}.composer-shell{background:rgba(10,13,34,.80);border:1px solid rgba(255,255,255,.45);border-radius:20px;padding:14px 16px 10px;box-shadow:0 20px 50px rgba(0,0,0,.28)}
div[data-testid="stTextArea"] textarea { background:transparent!important;border:0!important;color:#fff!important;min-height:72px!important;box-shadow:none!important; }div[data-testid="stTextArea"] textarea::placeholder{color:#E5E7EB!important}.composer-shell div[data-testid="stButton"] button[kind="primary"]{border-radius:50%;min-width:42px;height:42px;padding:0;background:#FF7A24!important}.composer-shell div[data-testid="stButton"] button[kind="secondary"]{color:#E5E7EB!important;text-decoration:none!important;border:1px solid rgba(255,255,255,.18)!important;background:rgba(255,255,255,.10)!important;border-radius:10px!important;padding:8px 12px!important}.history-title{font-size:12px;color:#B8C5E1!important;margin:18px 0 6px}.sidebar-new div[data-testid="stButton"] button{background:#F8FAFC!important;color:#0A1838!important;width:100%;border-radius:9px!important;font-weight:700}.side-item div[data-testid="stButton"] button{background:transparent!important;color:#F8FAFC!important;border:0!important;text-align:left!important;width:100%;padding:6px!important}.results-header,.claim-card{position:relative;z-index:1}.claim-card{background:rgba(4,12,33,.83)!important;border-color:rgba(255,255,255,.25)!important}.claim-text,.claim-summary,.claim-strength,.results-count{color:#F8FAFC!important}@media(max-width:700px){.hero-title{margin-top:48px;font-size:29px}.navlinks{gap:12px}.navlinks span:nth-child(n+4){display:none}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

REFINED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { --navy:#071238; --ink:#f8fafc; --muted:#cbd5e1; --orange:#ff7a24; --glass:rgba(6,12,37,.76); }
.stApp { background:radial-gradient(ellipse at 78% 44%, rgba(255,122,36,.88) 0%, rgba(188,70,96,.56) 24%, transparent 49%), linear-gradient(125deg,#061338 0%,#142f75 43%,#3f2f61 66%,#ff7a24 100%) !important; min-height:100dvh; overflow-x:hidden; }
.stApp:before { background:linear-gradient(90deg,rgba(4,10,34,.70),rgba(7,12,42,.24) 52%,rgba(8,8,25,.34)) !important; z-index:0; }
.block-container { max-width:1000px !important; min-height:100dvh; padding:20px 34px 64px !important; }
header[data-testid="stHeader"] { display:none !important; }

section[data-testid="stSidebar"] { width:280px !important; background:linear-gradient(180deg,rgba(5,16,50,.96),rgba(3,8,27,.94)) !important; border-right:1px solid rgba(255,255,255,.28) !important; }
section[data-testid="stSidebar"] > div { padding:18px 17px !important; }
.fact-logo { font-family:Inter,sans-serif; font-size:27px !important; font-weight:800 !important; letter-spacing:.025em !important; margin:0 !important; }
.fact-logo .shield { display:inline-grid; place-items:center; width:38px; height:44px; margin-right:8px; clip-path:polygon(50% 0,94% 14%,86% 70%,50% 100%,14% 70%,6% 14%); background:var(--orange); color:#071238; font-size:23px; vertical-align:middle; }
.fact-sub { margin:0 0 23px 54px !important; font-size:12px !important; font-weight:600; }
.sidebar-new div[data-testid="stButton"] button { min-height:43px; text-align:left !important; padding-left:18px !important; }
.side-label { color:#f8fafc !important; font-size:14px; margin:20px 4px 12px !important; }
.history-title { color:#d7e0f2 !important; font-size:13px !important; margin:18px 3px 8px !important; }
.history-label { color:#aebbd6 !important; font-size:12px; font-weight:600; letter-spacing:.04em; margin:17px 2px 7px; }
section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.22) !important; margin:17px 0 !important; }
div[data-testid="stSidebar"] [data-testid="stButton"] button { min-height:36px; }
div[data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible, button:focus-visible, textarea:focus-visible, input:focus-visible { outline:3px solid #fbbf24 !important; outline-offset:3px !important; }

.navline { position:relative; z-index:2; margin:0 !important; padding:3px 8px 0; font-size:14px !important; }
.stButton button[kind="tertiary"] { color:#fff !important; border:0 !important; background:transparent !important; border-radius:0 !important; min-height:42px; font-weight:600; }
.stButton button[kind="tertiary"]:hover { color:#fff !important; border-bottom:2px solid var(--orange) !important; }
.navlinks { gap:42px !important; }
.navlinks span { padding:11px 0 13px; color:#fff; }
.navlinks span:first-child { border-bottom:2px solid var(--orange) !important; }
.nav-cta { border:1px solid rgba(255,255,255,.50); border-radius:9px; padding:9px 18px; background:rgba(6,11,35,.68); font-weight:600; }
.hero-title { position:relative; z-index:2; font-size:33px !important; font-weight:500 !important; margin:143px 0 24px !important; text-shadow:0 2px 16px rgba(0,0,0,.42); }

.composer-shell { position:relative; z-index:2; padding:0 !important; border-radius:18px !important; background:var(--glass) !important; border:1px solid rgba(255,255,255,.58) !important; box-shadow:0 20px 45px rgba(0,0,0,.36) !important; overflow:hidden; }
.composer-shell div[data-testid="stTextArea"] { padding:14px 17px 0; }
.composer-shell div[data-testid="stTextArea"] textarea { min-height:62px !important; font-size:14px; line-height:1.5; resize:none; }
.composer-shell div[data-testid="stFormSubmitButton"] { display:flex; justify-content:flex-end; padding:0 15px 13px; }
.composer-shell div[data-testid="stFormSubmitButton"] button { width:40px; height:40px; min-height:40px; border:0; border-radius:50% !important; background:var(--orange) !important; font-size:0; }
.composer-shell div[data-testid="stFormSubmitButton"] button::after { content:'➜'; font-size:18px; color:white; }
.composer-shell div[data-testid="stFormSubmitButton"] button:hover { background:#ff9149 !important; transform:translateY(-1px); }
div[data-testid="stPopover"] { position:relative; z-index:2; margin-top:11px; }
div[data-testid="stPopover"] button { min-width:42px !important; min-height:42px !important; border-radius:10px !important; background:rgba(6,12,37,.80) !important; border:1px solid rgba(255,255,255,.25) !important; color:#fff !important; }
.hero-note { position:relative; z-index:2; max-width:1000px; margin:176px auto 0; padding-top:24px; border-top:1px solid rgba(255,255,255,.68); color:#e2e8f0; font-size:13px; text-align:left; }
.results-header,.claim-card,.status-line,.error-line,.empty-note { position:relative; z-index:2; }
.claim-card { border-radius:16px !important; }
@media (max-width: 800px) { .block-container{padding:14px 18px 48px!important}.navlinks{gap:16px!important}.navlinks span:nth-child(n+4){display:none}.hero-title{font-size:28px!important;margin-top:72px!important}.hero-note{margin-top:86px}.composer-shell{border-radius:14px!important} }
</style>
"""
st.markdown(REFINED_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- state ----
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "status" not in st.session_state:
    st.session_state.status = "idle"  # idle | done | error
if "claims" not in st.session_state:
    st.session_state.claims = None
if "error_msg" not in st.session_state:
    st.session_state.error_msg = ""
if "total_elapsed" not in st.session_state:
    st.session_state.total_elapsed = 0.0
if "last_media_hash" not in st.session_state:
    st.session_state.last_media_hash = ""
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "info_panel" not in st.session_state:
    st.session_state.info_panel = ""
if "pending_transcript" in st.session_state:
    st.session_state.input_text = st.session_state.pop("pending_transcript")


def load_example():
    st.session_state.input_text = EXAMPLE


def reset_all():
    st.session_state.input_text = ""
    st.session_state.claims = None
    st.session_state.status = "idle"
    st.session_state.error_msg = ""


def load_history(topic):
    st.session_state.input_text = history_prompt(topic) if topic in {"Science", "Health", "Technology", "Climate"} else topic
    st.session_state.claims = None
    st.session_state.status = "idle"


def open_info_panel(panel: str):
    st.session_state.info_panel = panel


# ---------------------------------------------------------------- header ---
nav_fact, nav_how, nav_examples, nav_about, nav_space, nav_start = st.columns([1.15, 1.35, 1.05, .8, 3.3, 1.35])
with nav_fact:
    st.button("Fact check", key="nav_fact", on_click=open_info_panel, args=("",), type="tertiary")
with nav_how:
    st.button("How it works", key="nav_how", on_click=open_info_panel, args=("how",), type="tertiary")
with nav_examples:
    st.button("Examples", key="nav_examples", on_click=load_example, type="tertiary")
with nav_about:
    st.button("About", key="nav_about", on_click=open_info_panel, args=("about",), type="tertiary")
with nav_start:
    st.button("Get started", key="nav_start", on_click=reset_all)

if st.session_state.info_panel == "how":
    st.info("FactAI extracts factual claims, searches public web sources, compares the evidence with a local NLI model, and labels each claim as supported, contradicted, mixed, or unverified. No paid API key is required.")
elif st.session_state.info_panel == "about":
    st.info("FactAI is a local-first fact-checking workspace. Results are evidence-assisted estimates, so open the cited sources before relying on a conclusion.")

st.markdown('<h1 class="hero-title">Describe a claim. We’ll <b>verify</b> it.</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="fact-logo"><span class="shield">✓</span>FACT<b>AI</b></div><p class="fact-sub">Truth. Verified.</p>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-new">', unsafe_allow_html=True)
    st.button("New chat", on_click=reset_all, use_container_width=True)
    st.markdown('</div><p class="history-title">Search chats</p>', unsafe_allow_html=True)
    history_filter = st.text_input("Search previous checks", placeholder="Search chats", label_visibility="collapsed")
    st.markdown('<p class="history-title">History</p><hr><p class="history-label">TODAY</p>', unsafe_allow_html=True)
    history_topics = [entry for entry in st.session_state.search_history if history_filter.lower() in entry.lower()]
    if not history_topics:
        history_topics = [topic for topic in ("Science", "Health", "Technology", "Climate") if history_filter.lower() in topic.lower()]
    for topic in history_topics[:6]:
        label = f"{topic} fact check" if topic in {"Science", "Health", "Technology", "Climate"} else topic[:34]
        st.button(label, key=f"history_{topic}", on_click=load_history, args=(topic,), use_container_width=True)
    st.markdown('---')
    with st.expander("Verification settings"):
        whisper_size = st.selectbox("Whisper model size (audio/video)", ["tiny", "base", "small"], index=1)
        max_results = st.slider("Sources to check per claim", 2, 8, 5)
        min_confidence = st.slider("Minimum evidence confidence", 0.3, 0.9, 0.55, 0.05)
        summary_model = st.text_input("Local Ollama model for summaries", value=DEFAULT_SUMMARY_MODEL)
        agent_mode = st.checkbox("Agentic search (experimental fallback)", value=False)
        agent_model = st.text_input("Agent model", value=DEFAULT_AGENT_MODEL, disabled=not agent_mode)

# ----------------------------------------------------------------- input ---
st.markdown('<div class="composer-shell">', unsafe_allow_html=True)
with st.form("claim_check_form", clear_on_submit=False):
    st.text_area(
        "Paste text here", height=150, key="input_text",
        label_visibility="collapsed", placeholder="Paste a fact, headline, or statement to verify…",
    )
    analyze_clicked = st.form_submit_button("➜", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

attach_col, _ = st.columns([1, 6])
with attach_col:
    with st.popover("＋", help="Upload audio or video to transcribe"):
        media_kind = st.radio(
            "Upload type", ["Audio", "Video"], horizontal=True,
            key="media_kind",
        )
        media_extensions = (
            ["mp3", "wav", "m4a", "ogg", "flac", "aac"] if media_kind == "Audio"
            else ["mp4", "mov", "mkv", "webm", "avi", "mpeg"]
        )
        uploaded = st.file_uploader(
            f"Choose {media_kind.lower()} file", type=media_extensions,
            key=f"media_upload_{media_kind.lower()}",
        )
        if uploaded is not None:
            media_bytes = uploaded.getvalue()
            media_hash = hashlib.sha256(media_bytes).hexdigest()
            if media_hash != st.session_state.last_media_hash:
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(media_bytes)
                    tmp_path = tmp.name
                try:
                    with st.spinner("Transcribing locally with Whisper..."):
                        transcript = transcribe_media(
                            tmp_path, model_size=whisper_size
                        )
                    st.session_state.last_media_hash = media_hash
                    st.session_state.pending_transcript = transcript
                    st.session_state.claims = None
                    st.session_state.status = "idle"
                    st.rerun()
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                st.caption(f"Ready to transcribe: {uploaded.name}")

text_len = len(st.session_state.input_text.strip())

if 0 < text_len < MIN_CLAIM_LENGTH:
    st.markdown(
        f'<p class="hint-line">Enter at least {MIN_CLAIM_LENGTH} characters to check a claim.</p>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- pipeline ----
status_placeholder = st.empty()

if analyze_clicked and valid_claim_input(st.session_state.input_text):
    st.session_state.error_msg = ""
    st.session_state.claims = None

    try:
        status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[0]}\u2026</p>', unsafe_allow_html=True)
        time.sleep(0.4)

        status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[1]}\u2026</p>', unsafe_allow_html=True)
        extraction_timing = {}
        candidate_claims = extract_claims(
            st.session_state.input_text, model=summary_model, max_claims=3, timing=extraction_timing,
        )

        # A failed/empty extraction must not block a user from checking a
        # declarative statement. Send the submitted text through the evidence
        # pipeline as one candidate instead of showing a dead-end empty state.
        if not candidate_claims:
            candidate_claims = [st.session_state.input_text.strip()]

        if candidate_claims:
            model_timing = {}
            from timing import timed
            with timed(model_timing, "nli_model_load"):
                tokenizer, nli_model = load_nli_model()
            results = []
            for i, claim in enumerate(candidate_claims):
                verdict = None
                claim_timing = {}

                # Tenure claims need the fixed path's date comparison, and
                # current Indian office claims need its official-source check.
                # Keep both out of the experimental agent path.
                if (
                    agent_mode
                    and not is_tenure_duration_claim(claim)
                    and not official_office_query(claim)
                ):
                    status_placeholder.markdown(
                        f'<p class="status-line">Agent investigating claim {i + 1} of '
                        f'{len(candidate_claims)}\u2026</p>',
                        unsafe_allow_html=True,
                    )
                    verdict = investigate_claim(claim, model=agent_model, timing=claim_timing)
                    # None means: Ollama unreachable, a request failed, or the model
                    # never returned a valid final answer within the step cap --
                    # fall through to the fixed pipeline below for this claim. The
                    # time already spent in the failed agent attempt still counts,
                    # so it stays in claim_timing rather than being reset.

                if verdict is None:
                    status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[2]}\u2026</p>', unsafe_allow_html=True)
                    search_results = search_web(claim, max_results=max_results, timing=claim_timing)
                    # A targeted official query improves coverage for current
                    # Indian office-holder claims when general search misses it.
                    official_query = official_office_query(claim)
                    if official_query:
                        official_results = search_web(
                            official_query, max_results=3, timing=claim_timing
                        )
                        seen_urls = {item.get("url") for item in search_results}
                        search_results.extend(
                            item for item in official_results
                            if item.get("url") and item.get("url") not in seen_urls
                        )

                    status_placeholder.markdown(f'<p class="status-line">{STAGE_LABELS[3]}\u2026</p>', unsafe_allow_html=True)
                    verdict = verify_claim(
                        claim, search_results, tokenizer, nli_model,
                        min_confidence=min_confidence, summarizer_model=summary_model,
                        timing=claim_timing,
                    )
                for stage, seconds in model_timing.items():
                    claim_timing[stage] = claim_timing.get(stage, 0.0) + seconds
                verdict["timings"] = {k: round(v, 3) for k, v in claim_timing.items()}
                verdict["elapsed"] = round(total_seconds(claim_timing), 1)
                verdict["via"] = "agent" if agent_mode and "steps" in verdict else "fixed"
                results.append(verdict)

            st.session_state.claims = results
            submitted = st.session_state.input_text.strip()
            if submitted and submitted not in st.session_state.search_history:
                st.session_state.search_history.insert(0, submitted)
            st.session_state.total_elapsed = total_seconds(extraction_timing) + sum(
                r["elapsed"] for r in results
            )

            st.session_state.status = "done"
    except Exception as e:
        import traceback
        traceback.print_exc()  # full traceback goes to your terminal
        st.session_state.status = "error"
        st.session_state.error_msg = f"Something broke: {e}"
    finally:
        status_placeholder.empty()

# -------------------------------------------------------------- results ---
if st.session_state.status == "error":
    st.markdown(f'<p class="error-line">{st.session_state.error_msg}</p>', unsafe_allow_html=True)

if st.session_state.status == "done" and st.session_state.claims is not None:
    claims = st.session_state.claims
    if len(claims) == 0:
        st.markdown(
            '<p class="empty-note">No independently checkable claims turned up in that '
            'text \u2014 it may be mostly opinion, or too vague to verify.</p>',
            unsafe_allow_html=True,
        )
    else:
        n = len(claims)
        st.markdown(
            f'<div class="results-header"><span class="results-count">'
            f'{n} claim{"s" if n != 1 else ""} checked · '
            f'{st.session_state.total_elapsed:.1f}s total</span></div>',
            unsafe_allow_html=True,
        )
        st.button("\u21BA Check another", type="secondary", on_click=reset_all)

        cards_html = ""
        for i, item in enumerate(claims):
            style = VERDICT_STYLE.get(item["verdict"], VERDICT_STYLE["Unverified"])
            tone = verdict_tone(item["verdict"])
            sources_html = "".join(
                f'<a href="{s["url"]}" target="_blank" rel="noopener noreferrer" '
                f'class="source-chip" title="{s["title"]}">\u2197 {hostname(s["url"])}</a>'
                for s in item.get("sources", [])
            )
            last_class = " last" if i == n - 1 else ""
            cards_html += f"""
            <div class="claim-card{last_class}">
              <div class="claim-top">
                <p class="claim-text">{item['claim']}</p>
                <div class="stamp" style="color:{style['color']};border-color:{style['color']};--r:{style['rotate']};">
                  {tone['label']}
                </div>
              </div>
              <p class="claim-summary">{item['summary']}</p>
              <p class="claim-strength">{item['strength']} · Verdict: {item['verdict']}</p>
              <p class="claim-strength">Mode: {item.get('via', 'fixed NLI')} · stage timings: {', '.join(f'{k} {v:.2f}s' for k, v in item.get('timings', {}).items())}</p>
              {f'<div class="claim-sources">{sources_html}</div>' if sources_html else ''}
            </div>
            """
        st.markdown(cards_html, unsafe_allow_html=True)

st.markdown('<p class="hero-note">Automated, heuristic estimate based on top web search results and a local language model. Not a substitute for professional fact-checking.</p>', unsafe_allow_html=True)
