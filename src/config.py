
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
LLM_CACHE_DIR = PROJECT_ROOT / ".cache" / "llm"

# ---------- Similar wafers ----------
SIMILAR_COLLECTION = "wafer_fingerprints"
SIMILAR_K = 5

# ---------- API ----------
SAMPLES_DIR = PROJECT_ROOT / "samples"
API_VERSION = "1.0.0"