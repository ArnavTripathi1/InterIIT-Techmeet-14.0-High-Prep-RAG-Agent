import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
POLICY_DIR = DATA_DIR / "policy_docs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODEL_PATH = str(BASE_DIR / "models/ggml-gpt4all-j-v1.3-groovy.gguf")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RISK_THRESHOLDS = {"review": 0.6, "tighten": 0.75, "request_collateral": 0.85}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env file!")