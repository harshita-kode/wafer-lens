
from pathlib import Path

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