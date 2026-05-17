#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path


MODULE_PATH = (
    "/tmp/glm51/ktransformers/third_party/sglang/"
    "benchmark/kernels/quantization/tuning_block_wise_kernel.py"
)


def load_tuning_module():
    spec = importlib.util.spec_from_file_location("local_tuning_block_wise_kernel", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_main(task: dict) -> dict[int, dict]:
    module = load_tuning_module()
    torch = module.torch

    gpu_id = task["gpu_id"]
    torch.cuda.set_device(gpu_id)

    search_space = module.get_configs_compute_bound()
    search_space = [
        config
        for config in search_space
        if task["block_k"] % config["BLOCK_SIZE_K"] == 0
    ]

    results: dict[int, dict] = {}
    for batch_size in task["batch_sizes"]:
        config = module.tune(
            batch_size,
            task["N"],
            task["K"],
            [task["block_n"], task["block_k"]],
            module.DTYPE_MAP[task["out_dtype"]],
            search_space,
            task["input_type"],
        )
        results[batch_size] = config
    return results


def distribute(values: list[int], slots: int) -> list[list[int]]:
    groups = [[] for _ in range(slots)]
    for index, value in enumerate(values):
        groups[index % slots].append(value)
    return [group for group in groups if group]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32],
    )
    parser.add_argument("--input-type", choices=["fp8", "int8"], default="fp8")
    parser.add_argument(
        "--out-dtype",
        choices=["float32", "float16", "bfloat16", "half"],
        default="bfloat16",
    )
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=0)
    args = parser.parse_args()

    module = load_tuning_module()
    torch = module.torch

    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError("no GPUs available for tuning")

    num_gpus = args.num_gpus or available_gpus
    if num_gpus < 1 or num_gpus > available_gpus:
        raise RuntimeError(f"invalid num_gpus={num_gpus}, available={available_gpus}")

    batch_sizes = sorted(set(args.batch_sizes))
    task_groups = distribute(batch_sizes, num_gpus)

    tasks = [
        {
            "gpu_id": gpu_id,
            "batch_sizes": task_groups[gpu_id],
            "N": args.N,
            "K": args.K,
            "input_type": args.input_type,
            "out_dtype": args.out_dtype,
            "block_n": args.block_n,
            "block_k": args.block_k,
        }
        for gpu_id in range(len(task_groups))
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        partial_results = pool.map(worker_main, tasks)

    merged: dict[int, dict] = {}
    for result in partial_results:
        merged.update(result)

    missing = [batch_size for batch_size in batch_sizes if batch_size not in merged]
    if missing:
        raise RuntimeError(f"missing tuned configs for batch sizes: {missing}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device_name = module.get_device_name().replace(" ", "_")
    output_path = (
        save_dir
        / (
            f"N={args.N},K={args.K},device_name={device_name},"
            f"dtype={args.input_type}_w8a8,block_shape=[{args.block_n}, {args.block_k}].json"
        )
    )
    output_path.write_text(
        json.dumps({str(k): merged[k] for k in batch_sizes}, indent=4) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
