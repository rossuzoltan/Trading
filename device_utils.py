from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingRuntimePlan:
    device: str
    accelerator_label: str
    cpu_cores: int
    env_workers: int
    torch_threads: int
    interop_threads: int
    npu_backend: str | None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_npu_backend() -> str | None:
    if _module_available("torch_directml"):
        return "torch-directml"
    if _module_available("onnxruntime"):
        return "onnxruntime"
    if _module_available("openvino"):
        return "openvino"
    if _module_available("intel_extension_for_pytorch"):
        return "ipex"
    return None


def get_best_device() -> str:
    """
    Return the best training device available to the current PyTorch runtime.
    SB3 training uses PyTorch; without a PyTorch-compatible NPU backend, CPU is correct.
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        x = torch.randn(1).cuda()
        _ = torch.tanh(x)
        return "cuda"
    except Exception as exc:
        print(f"[WARN] GPU detected but unusable for training: {exc}")
        print("[INFO] Falling back to CPU for stability.")
        return "cpu"


def _recommended_env_workers(cpu_cores: int) -> int:
    if cpu_cores <= 4:
        return 1
    return max(2, min(8, cpu_cores - 2))


def _recommended_torch_threads(cpu_cores: int, env_workers: int) -> int:
    remaining = max(2, cpu_cores - env_workers)
    return max(2, min(8, remaining))


def configure_training_runtime(requested_env_workers: int | None = None) -> TrainingRuntimePlan:
    cpu_cores = max(1, os.cpu_count() or 1)
    env_workers = requested_env_workers or _recommended_env_workers(cpu_cores)
    env_workers = max(1, min(env_workers, max(1, cpu_cores - 1)))
    torch_threads = _recommended_torch_threads(cpu_cores, env_workers)
    interop_threads = max(1, min(4, torch_threads // 2))

    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(interop_threads)
    torch.set_float32_matmul_precision("high")

    device = get_best_device()
    npu_backend = detect_npu_backend()
    accelerator_label = device.upper()
    if device == "cpu" and npu_backend is not None:
        accelerator_label = f"CPU (NPU backend detected: {npu_backend}, not used by SB3 training)"
    elif device == "cpu":
        accelerator_label = "CPU"

    return TrainingRuntimePlan(
        device=device,
        accelerator_label=accelerator_label,
        cpu_cores=cpu_cores,
        env_workers=env_workers,
        torch_threads=torch_threads,
        interop_threads=interop_threads,
        npu_backend=npu_backend,
    )


if __name__ == "__main__":
    plan = configure_training_runtime()
    print(f"Device: {plan.accelerator_label}")
    print(f"CPU cores: {plan.cpu_cores}")
    print(f"Env workers: {plan.env_workers}")
    print(f"Torch threads: {plan.torch_threads}")
    print(f"Interop threads: {plan.interop_threads}")
