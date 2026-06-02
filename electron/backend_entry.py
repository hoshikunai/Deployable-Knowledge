import os
import shutil
import sys
from pathlib import Path

import uvicorn


def configure_paths() -> None:
    if getattr(sys, "frozen", False):
        resource_root = Path(sys.executable).resolve().parent.parent
        data_root = Path(os.getenv("DK_DATA_DIR", Path.home() / ".deployable-knowledge"))
    else:
        resource_root = Path(__file__).resolve().parent.parent
        data_root = Path(os.getenv("DK_DATA_DIR", resource_root))

    os.environ.setdefault("DK_RESOURCE_ROOT", str(resource_root))
    os.environ.setdefault("DK_DATA_DIR", str(data_root))
    os.chdir(resource_root)
    if str(resource_root) not in sys.path:
        sys.path.insert(0, str(resource_root))

    for directory in ("documents", "pdfs", "tmp_model", "chroma_db", "prompts"):
        (data_root / directory).mkdir(parents=True, exist_ok=True)

    source_prompts = resource_root / "prompts"
    target_prompts = data_root / "prompts"
    if source_prompts.exists():
        for source in source_prompts.glob("*.json"):
            target = target_prompts / source.name
            if not target.exists():
                shutil.copy2(source, target)


if __name__ == "__main__":
    configure_paths()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
