from pathlib import Path
import os

# === Base Paths ===
_resource_root = os.getenv("DK_RESOURCE_ROOT")
_data_root = os.getenv("DK_DATA_DIR")

BASE_DIR = (
    Path(_resource_root).expanduser().resolve()
    if _resource_root
    else Path(__file__).resolve().parent
)
DATA_DIR = Path(_data_root).expanduser().resolve() if _data_root else BASE_DIR
UPLOAD_DIR = DATA_DIR / "documents"
PDF_DIR = DATA_DIR / "pdfs"
MODEL_DIR = DATA_DIR / "tmp_model"

# === ChromaDB ===
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "default_collection"

# === SQL Database ===
_database_path = os.getenv("DATABASE_PATH")
DATABASE_PATH = Path(_database_path).expanduser() if _database_path else DATA_DIR / "app.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "0") == "1"

# === Embedding Model ===
# Always point to a local directory for offline model loading
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDINGS_DEVICE = os.getenv("EMBEDDINGS_DEVICE", "cpu")
EMBEDDINGS_OFFLINE_ONLY = os.getenv("EMBEDDINGS_OFFLINE_ONLY", "0") == "1"

# === Security ===
ALLOWED_DOCUMENT_EXTENSIONS = {".txt", ".pdf", ".md", ".html"}
MIN_TOP_K = 1
MAX_TOP_K = 20

# === Ollama ===
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ibm/granite4.1:3b")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Backwards compatibility for legacy code expecting OLLAMA_URL
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"

# === llama.cpp ===
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8081/v1")
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "granite4.1:3b-q4_K_M")
LLAMA_CPP_REPEAT_PENALTY = float(os.getenv("LLAMA_CPP_REPEAT_PENALTY", "1.12"))
LLAMA_CPP_AVAILABLE = (
    os.getenv("DK_BUNDLED_LLAMA", "0") == "1"
    or bool(os.getenv("LLAMA_CPP_BASE_URL"))
    or bool(os.getenv("LLAMA_CPP_MODEL"))
)
DEFAULT_LLM_PROVIDER = os.getenv(
    "DEFAULT_LLM_PROVIDER",
    "llama_cpp" if LLAMA_CPP_AVAILABLE else "ollama",
)
DEFAULT_LLM_MODEL = os.getenv(
    "DEFAULT_LLM_MODEL",
    LLAMA_CPP_MODEL if DEFAULT_LLM_PROVIDER == "llama_cpp" else "",
)

# === OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

# === Anthropic ===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

# === Gemini ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# === GitHub Models ===
GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN", "")
GITHUB_MODELS_BASE_URL = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.github.ai")
GITHUB_MODELS_MODEL = os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1")
GITHUB_MODELS_API_VERSION = os.getenv("GITHUB_MODELS_API_VERSION", "2026-03-10")
GITHUB_MODELS_ORG = os.getenv("GITHUB_MODELS_ORG", "")

# === Prompt Templates ===
PROMPTS_DIR = DATA_DIR / "prompts"
