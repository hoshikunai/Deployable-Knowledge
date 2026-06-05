import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = ROOT / "electron-dist" / "win-unpacked" / "resources"


def find_file(start_dir: Path, file_name: str) -> Path | None:
    if not start_dir.exists():
        return None

    return next(start_dir.rglob(file_name), None)


def main() -> int:
    llama_server = find_file(RESOURCES_DIR / "llama.cpp", "llama-server.exe")
    pdf_min = find_file(RESOURCES_DIR / "app" / "static", "pdf.min.js")
    pdf_worker = find_file(RESOURCES_DIR / "app" / "static", "pdf.worker.min.js")
    prompt_dir = RESOURCES_DIR / "prompts"
    prompt_files = list(prompt_dir.glob("*.json")) if prompt_dir.exists() else []

    if not all((llama_server, pdf_min, pdf_worker, prompt_files)):
        print("Windows package is missing required standalone assets.", file=sys.stderr)
        print(f"resourcesDir: {RESOURCES_DIR}", file=sys.stderr)
        print(f"llama-server.exe: {llama_server or 'missing'}", file=sys.stderr)
        print(f"pdf.min.js: {pdf_min or 'missing'}", file=sys.stderr)
        print(f"pdf.worker.min.js: {pdf_worker or 'missing'}", file=sys.stderr)
        print(f"prompt JSON files: {len(prompt_files)}", file=sys.stderr)
        return 1

    print("Windows standalone assets found:")
    print(f"llama-server.exe: {llama_server}")
    print(f"pdf.min.js: {pdf_min}")
    print(f"pdf.worker.min.js: {pdf_worker}")
    print(f"prompt JSON files: {len(prompt_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
