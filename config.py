from pathlib import Path
import os

# === Base Paths ===
_resource_root = os.getenv("DK_RESOURCE_ROOT")
_data_root = os.getenv("DK_DATA_DIR")

BASE_DIR = Path(_resource_root).expanduser().resolve() if _resource_root else Path(__file__).resolve().parent
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

# === Prompt Templates ===
PROMPTS_DIR = DATA_DIR / "prompts"
