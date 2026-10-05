from __future__ import annotations

import time
from pathlib import Path

import onnx
import torch
from gliner import GLiNER

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".cache" / "gliner-parity" / "relex-onnx"
RELEX_REPO = "knowledgator/gliner-relex-large-v0.5"


def exportable_diag_embed(values: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(values.size(-1), dtype=values.dtype, device=values.device)
    return values.unsqueeze(-1) * identity


def main() -> None:
    torch.diag_embed = exportable_diag_embed
    started = time.perf_counter()
    model = GLiNER.from_pretrained(RELEX_REPO).to("cpu")
    model.eval()
    with torch.inference_mode():
        paths = model.export_to_onnx(save_dir=OUT, onnx_filename="model.onnx", quantize=False)
    graph = onnx.load(paths["onnx_path"], load_external_data=False).graph
    print(f"exported in {time.perf_counter() - started:.0f}s -> {paths['onnx_path']}")
    print("inputs:", [(item.name, [d.dim_param or d.dim_value for d in item.type.tensor_type.shape.dim])
                      for item in graph.input])
    print("outputs:", [(item.name, [d.dim_param or d.dim_value for d in item.type.tensor_type.shape.dim])
                       for item in graph.output])


if __name__ == "__main__":
    main()
