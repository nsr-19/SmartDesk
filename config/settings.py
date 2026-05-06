"""
Centralised configuration for SmartDesk AI.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IT_KB_PATH = DATA_DIR / "it_knowledge.json"
HR_KB_PATH = DATA_DIR / "hr_policies.json"

# ── OpenAI ─────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. Copy .env.example → .env and add your key."
    )

# Primary model — reasoning, generation, synthesis
PRIMARY_MODEL = "gpt-4o"
# Lighter model — routing & classification (cheaper, faster)
ROUTER_MODEL = "gpt-4o"

MODEL_TEMPERATURE = 0.2

# ── RAG ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
# Similarity threshold: chunks scoring below this are treated as
# low-confidence and the agent will offer to escalate. Calibrated for
# text-embedding-3-small, where a relevant short-query/long-answer match
# typically scores 0.35–0.55. Was 0.45 — too strict; "leave policy" → 0.41
# was being rejected despite being a perfect topical hit.
CONFIDENCE_THRESHOLD = 0.35
RAG_TOP_K = 3
