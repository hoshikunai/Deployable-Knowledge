from PyInstaller.utils.hooks import (
    PY_DYLIB_PATTERNS,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
    ],
)


def is_runtime_torch_module(module_name):
    return not module_name.startswith(
        (
            "nvidia.",
            "torch.distributed",
            "torch.testing",
            "torch.utils.tensorboard",
        )
    )


hiddenimports = collect_submodules("torch", filter=is_runtime_torch_module)

binaries = collect_dynamic_libs(
    "torch",
    search_patterns=PY_DYLIB_PATTERNS + ["*.so.*", "*.dll"],
)

bindepend_symlink_suppression = ["**/torch/lib/*.so*"]
