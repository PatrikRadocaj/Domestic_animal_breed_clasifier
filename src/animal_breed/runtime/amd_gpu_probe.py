"""Probe the isolated AMD ROCm runtime used for training."""

from __future__ import annotations

import json
import sys

import torch


def build_probe_report() -> dict[str, object]:
    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if not torch.cuda.is_available():
        return report

    device_index = 0
    report["device_name"] = torch.cuda.get_device_name(device_index)
    report["device_properties"] = {
        "total_memory_gib": round(torch.cuda.get_device_properties(device_index).total_memory / (1024**3), 2),
        "multi_processor_count": int(torch.cuda.get_device_properties(device_index).multi_processor_count),
    }

    x = torch.randn((256, 256), device="cuda")
    y = (x @ x).sum()
    report["tensor_op_ok"] = True
    report["tensor_op_value"] = float(y.item())
    return report


def main() -> None:
    report = build_probe_report()
    print(json.dumps(report, indent=2))
    if not report.get("cuda_available"):
        sys.exit(2)


if __name__ == "__main__":
    main()
