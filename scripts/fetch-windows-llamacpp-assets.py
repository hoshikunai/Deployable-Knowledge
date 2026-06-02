import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "vendor"
LLAMA_DIR = VENDOR_ROOT / "llama.cpp" / "win-x64"
PDFJS_DIR = ROOT / "app" / "static" / "vendor" / "pdfjs"
USER_AGENT = "deployable-knowledge-builder"

PDFJS_ASSETS = (
    {
        "file": "pdf.min.js",
        "url": "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js",
    },
    {
        "file": "pdf.worker.min.js",
        "url": "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js",
    },
)

GRANITE_MODELS = (
    {
        "label": "Granite 4.1 3B Q4_K_M",
        "file": VENDOR_ROOT / "models" / "granite4.1-3b" / "granite-4.1-3b-Q4_K_M.gguf",
        "url": "https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-Q4_K_M.gguf",
    },
)


def request(url: str, *, github_auth: bool = False) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if github_auth and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return urllib.request.Request(url, headers=headers)


def request_json(url: str) -> dict:
    with urllib.request.urlopen(request(url, github_auth=True)) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")

    try:
        with urllib.request.urlopen(request(url)) as response:
            with partial.open("wb") as file:
                shutil.copyfileobj(response, file)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def expand_archive(archive: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)


def first_matching_asset(assets: list[dict]) -> dict | None:
    patterns = (
        re.compile(r"^llama-.*-bin-win-cpu-x64\.zip$"),
        re.compile(r"^llama-.*-bin-win-avx2-x64\.zip$"),
    )
    for pattern in patterns:
        for asset in assets:
            if pattern.match(asset.get("name", "")):
                return asset
    return None


def fetch_llama_cpp() -> None:
    server_path = LLAMA_DIR / "llama-server.exe"
    if server_path.exists():
        print(f"llama.cpp already present: {server_path}")
        return

    release = request_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
    asset = first_matching_asset(release.get("assets", []))
    if not asset or not asset.get("browser_download_url"):
        raise RuntimeError("Could not find a Windows x64 CPU llama.cpp release asset")

    archive = VENDOR_ROOT / asset["name"]
    extract_dir = VENDOR_ROOT / ".llama.cpp-extract"

    print(f"Downloading {asset['name']}")
    download(asset["browser_download_url"], archive)
    expand_archive(archive, extract_dir)

    extracted_server = next(extract_dir.rglob("llama-server.exe"), None)
    if not extracted_server:
        raise RuntimeError(f"llama-server.exe was not found after extracting {asset['name']}")

    shutil.rmtree(LLAMA_DIR, ignore_errors=True)
    shutil.copytree(extracted_server.parent, LLAMA_DIR)

    if not server_path.exists():
        raise RuntimeError(f"llama-server.exe was not found after extracting {asset['name']}")

    archive.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def fetch_pdfjs() -> None:
    for asset in PDFJS_ASSETS:
        destination = PDFJS_DIR / asset["file"]
        if destination.exists():
            print(f"PDF.js asset already present: {destination}")
            continue

        print(f"Downloading {asset['file']}")
        download(asset["url"], destination)


def fetch_granite() -> None:
    for model in GRANITE_MODELS:
        model_file = model["file"]
        if model_file.exists():
            print(f"{model['label']} already present: {model_file}")
            return

        print(f"Downloading {model['label']}")
        download(model["url"], model_file)
        return

    raise RuntimeError("No Granite GGUF model configured")


def main() -> int:
    try:
        fetch_llama_cpp()
        fetch_pdfjs()
        fetch_granite()
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
