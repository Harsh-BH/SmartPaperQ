import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"  # Fast LLaMA 3 model on Groq

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval settings
TOP_K = 5

# Application settings
DEBUG = True
