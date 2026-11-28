
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()   # reads .env into environment variables

# Project root = the folder that contains src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "wafer_cnn.pt"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

IMG_SIZE = 64

# Below this confidence, the result is reported as "uncertain".
# Chosen from the test evaluation: >=70% -> 96.3% accurate, <70% -> ~60% accurate.
UNCERTAIN_THRESHOLD = 0.70

# ---------- RAG ----------
KB_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
KB_COLLECTION = "wafer_kb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- LLM ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").strip().lower() == "true"
LLM_CACHE_DIR = PROJECT_ROOT / ".cache" / "llm"

# ---------- Similar wafers ----------
SIMILAR_COLLECTION = "wafer_fingerprints"
SIMILAR_K = 5

# ---------- API ----------
SAMPLES_DIR = PROJECT_ROOT / "samples"
API_VERSION = "1.0.0"

PAST_WAFERS_PATH = DATA_DIR / "processed" / "wafers_64.npz"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ---------- Agent ----------
AGENT_MIN_SCORE = float(os.getenv("AGENT_MIN_SCORE", "0.7"))
AGENT_MIN_MARGIN = float(os.getenv("AGENT_MIN_MARGIN", "0.2"))
AGENT_MAX_WRITES = 2