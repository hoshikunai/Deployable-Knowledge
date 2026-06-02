# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


block_cipher = None
spec_dir = Path(SPECPATH).resolve()
root_dir = spec_dir.parent

datas = []
binaries = []
hiddenimports = []

excluded_module_roots = {
    "dask",
    "pytest",
    "tensorboard",
    "tensorrt",
    "tkinter",
}


def is_runtime_module(module_name):
    parts = module_name.split(".")
    if parts[0] in excluded_module_roots:
        return False
    return not any(part == "tests" or part.startswith("test_") for part in parts)


def is_runtime_data(data_entry):
    source, target = data_entry
    path_parts = set(Path(source).parts) | set(Path(target).parts)
    return "tests" not in path_parts and "__pycache__" not in path_parts

for package in (
    "chromadb",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "rapidocr",
    "pymupdf",
    "markdown2",
    "sklearn",
    "spacy",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += [data for data in pkg_datas if is_runtime_data(data)]
    binaries += pkg_binaries
    hiddenimports += [module for module in pkg_hiddenimports if is_runtime_module(module)]

hiddenimports += [module for module in collect_submodules("uvicorn") if is_runtime_module(module)]
hiddenimports += [module for module in collect_submodules("fastapi") if is_runtime_module(module)]
hiddenimports += [module for module in collect_submodules("starlette") if is_runtime_module(module)]
hiddenimports += [module for module in collect_submodules("app") if is_runtime_module(module)]
hiddenimports += [module for module in collect_submodules("api") if is_runtime_module(module)]
hiddenimports += [module for module in collect_submodules("core") if is_runtime_module(module)]
hiddenimports += ["config"]

a = Analysis(
    [str(spec_dir / "backend_entry.py")],
    pathex=[str(root_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "dask",
        "pytest",
        "tensorboard",
        "tensorrt",
        "tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DeployableKnowledgeBackend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="backend",
)
