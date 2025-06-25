import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# LLM & embedding models
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Ensure paths are absolute
BASE_DIR = Path(__file__).parent.parent.absolute()
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "vectorstore" / "faiss_index"))
