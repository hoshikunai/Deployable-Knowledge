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
    "cupy",
    "dask",
    "datasets",
    "detectron2",
    "diffusers",
    "jax",
    "jaxlib",
    "keras",
    "matplotlib",
    "nvidia",
    "onnx",
    "onnxruntime",
    "pandas",
    "pytest",
    "spacy",
    "tensorboard",
    "tensorflow",
    "tensorrt",
    "tkinter",
    "triton",
}


def is_runtime_module(module_name):
    parts = module_name.split(".")
    if parts[0] in excluded_module_roots:
        return False
    return not any(part in {"test", "tests"} or part.startswith("test_") for part in parts)


def is_runtime_data(data_entry):
    source, target = data_entry
    path_parts = set(Path(source).parts) | set(Path(target).parts)
    return not path_parts.intersection({"test", "tests", "__pycache__"})


def is_runtime_artifact(artifact_entry):
    path_parts = set()
    for value in artifact_entry[:2]:
        path_parts.update(Path(str(value)).parts)
    return not path_parts.intersection({"cuda", "nvidia"})

for package in (
    "chromadb",
    "sentence_transformers",
    "huggingface_hub",
    "tokenizers",
    "rapidocr",
    "pymupdf",
    "markdown2",
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
    hookspath=[str(spec_dir / "pyinstaller-hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cupy",
        "dask",
        "datasets",
        "detectron2",
        "diffusers",
        "jax",
        "jaxlib",
        "keras",
        "matplotlib",
        "nvidia",
        "onnx",
        "onnxruntime",
        "pandas",
        "pytest",
        "spacy",
        "tensorboard",
        "tensorflow",
        "tensorrt",
        "tkinter",
        "triton",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a.binaries = [entry for entry in a.binaries if is_runtime_artifact(entry)]
a.datas = [entry for entry in a.datas if is_runtime_artifact(entry)]

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
