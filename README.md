# Deployable-Knowledge

**Version vA0.2.2**

Offline‑first retrieval‑augmented generation (RAG) stack for disconnected or bandwidth‑constrained environments.

## Overview

Deployable‑Knowledge bundles a local vector store, prompt management and a lightweight web UI around a pluggable large‑language model.  Documents are embedded locally and queried through FastAPI endpoints which power the JavaScript front end.

## Features

- **Document ingestion** for PDF and plaintext sources
- **ChromaDB** vector store with sentence‑transformer embeddings
- **Chat and search** endpoints with optional streaming responses
- **Configurable prompts** and persona editing
- **Authentication middleware** with session and CSRF protection

## Quick Start for Usage

- For verbose start/run, simply run (double-click) `Launch-DeployableKnowledge.bat` or `Launch-DeployableKnowledge.ps1`
- For user-friendly/silent start, simply run (double-click) `Launch-DeployableKnowledge.bat-User` or `Launch-DeployableKnowledge-User.ps1`

## Quick Start for Development

**Unix / macOS:**

```bash
make setup
make run
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use `python -m pip` and `python -m pytest` so installs and tests use the same Python as your shell; this avoids "script location not on PATH" or "pytest not recognized" when the venv Scripts folder is not on PATH.

**Run tests:**

```bash
python -m pytest tests/ -q
```

Visit <http://localhost:8000> once the server starts. Ollama is available by default with the seeded `llama3` model; use **Manage API Keys** in the prompt editor to connect hosted providers.

**If you see "script location not on PATH" or "pytest not recognized":** run `pip` and `pytest` as modules so the active Python is used: `python -m pip install -r requirements.txt` and `python -m pytest tests/ -q`.

## Electron desktop shell

Install the Node dependencies once:

```bash
npm install
```

Run the desktop shell during development:

```bash
npm run electron:dev
```

The Electron app starts the FastAPI backend with `venv/bin/python` when it exists, otherwise it falls back to `python3` on Unix/macOS or `python` on Windows. Set `DK_PYTHON=/path/to/python` to force a specific interpreter.

Build distributable desktop artifacts:

```bash
python -m pip install -r requirements-build.txt
npm run electron:dist
```

Build a Windows MSI installer from a Windows shell or Windows CI runner:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-build.txt
npm install
npm run electron:dist:win:msi
```

The MSI is written to `electron-dist/`. Run the Windows MSI build on Windows so PyInstaller creates `DeployableKnowledgeBackend.exe`; cross-building from Linux/macOS will not produce the correct bundled backend executable. The MSI build runs `npm run electron:assets:win`, which downloads the Windows x64 CPU llama.cpp release and local PDF.js assets. Granite 4.1 3B Q4_K_M is downloaded into the user's app data directory on first launch instead of being embedded in the MSI.

Packaged Electron builds start the bundled `llama-server.exe` as a local fallback. If Ollama is reachable, the app can use installed Ollama models such as `ibm/granite4.1:3b`; otherwise new settings default to the bundled llama.cpp model.

The same build is available in GitHub Actions as `Windows MSI`. It can be run manually from the Actions tab and also runs for tags matching `v*`; the MSI is uploaded as a workflow artifact.

Do not change the configured MSI `upgradeCode` after publishing an installer; Windows Installer uses it to upgrade existing installations.

Packaging uses `electron/backend.spec` to build the FastAPI backend with PyInstaller, then bundles it into the Electron app.

## Architecture overview

The system is split into three layers:

```text
core/  – retrieval, prompt rendering and LLM adapters
api/   – FastAPI routers translating HTTP ↔ core
app/   – static assets and UI routes
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed diagrams and data‑flow breakdowns.

## Documentation

Additional guides live in the [`docs/`](docs) folder:

- [API reference](docs/API_REFERENCE.md)
- [UI overview](docs/UI_OVERVIEW.md)
- [Backend services](docs/BACKEND_SERVICES.md)
- [Configuration guide](docs/CONFIGURATION.md)
- [Prompt & LLM integration](docs/PROMPTS_LLM.md)

## Contributing

1. Create a feature branch off `main`.
2. Add tests and run `python -m pytest tests/` before submitting a pull request.
3. Follow the existing coding style and keep docstrings concise.
4. Open a PR describing the change and link to any relevant issues.

---
Released under the MIT license.
