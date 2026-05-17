#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = "Output only valid C++17 source code. Do not include explanations or markdown fences."
USER_PROMPT = "Generate a C++ program to print the first 1000 prime numbers."


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    health_url = f"{base_url}/health"
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy before timeout: {last_error}")


def make_payload(model: str, max_tokens: int, disable_thinking: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def one_request(
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    disable_thinking: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = post_json(
        f"{base_url}/v1/chat/completions",
        make_payload(model, max_tokens, disable_thinking),
        timeout,
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    return {
        "elapsed_s": elapsed,
        "content": choice,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "response": response,
    }


def extract_cpp_source(text: str) -> str:
    fence = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    include_idx = text.find("#include")
    if include_idx != -1:
        return text[include_idx:].strip()

    main_idx = text.find("int main")
    if main_idx != -1:
        prefix = '#include <iostream>\n#include <vector>\n#include <cmath>\nusing namespace std;\n\n'
        return prefix + text[main_idx:].strip()

    return text.strip()


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = math.isqrt(n)
    factor = 3
    while factor <= limit:
        if n % factor == 0:
            return False
        factor += 2
    return True


def validate_generated_code(code: str, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = work_dir / "generated_primes.cpp"
    bin_path = work_dir / "generated_primes"
    cpp_path.write_text(code, encoding="utf-8")

    compile_proc = subprocess.run(
        ["g++", "-std=c++17", "-O2", str(cpp_path), "-o", str(bin_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if compile_proc.returncode != 0:
        return {
            "ok": False,
            "stage": "compile",
            "stderr": compile_proc.stderr,
            "stdout": compile_proc.stdout,
            "cpp_path": str(cpp_path),
        }

    run_proc = subprocess.run(
        [str(bin_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if run_proc.returncode != 0:
        return {
            "ok": False,
            "stage": "run",
            "stderr": run_proc.stderr,
            "stdout": run_proc.stdout,
            "cpp_path": str(cpp_path),
            "bin_path": str(bin_path),
        }

    try:
        numbers = [int(part) for part in run_proc.stdout.split()]
    except ValueError:
        return {
            "ok": False,
            "stage": "parse_output",
            "stdout": run_proc.stdout,
            "cpp_path": str(cpp_path),
            "bin_path": str(bin_path),
        }

    if len(numbers) != 1000:
        return {
            "ok": False,
            "stage": "count",
            "count": len(numbers),
            "preview": numbers[:20],
            "cpp_path": str(cpp_path),
            "bin_path": str(bin_path),
        }

    if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
        return {
            "ok": False,
            "stage": "ordering",
            "preview": numbers[:20],
            "cpp_path": str(cpp_path),
            "bin_path": str(bin_path),
        }

    bad = [value for value in numbers if not is_prime(value)]
    if bad:
        return {
            "ok": False,
            "stage": "prime_check",
            "bad_values": bad[:20],
            "preview": numbers[:20],
            "cpp_path": str(cpp_path),
            "bin_path": str(bin_path),
        }

    return {
        "ok": True,
        "first_10": numbers[:10],
        "last_10": numbers[-10:],
        "cpp_path": str(cpp_path),
        "bin_path": str(bin_path),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    blend = index - lower
    return ordered[lower] * (1 - blend) + ordered[upper] * blend


def run_concurrency(
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    concurrency: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, base_url, model, max_tokens, timeout, disable_thinking)
            for _ in range(concurrency)
        ]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started

    completion_tokens = sum(item["completion_tokens"] for item in results)
    prompt_tokens = sum(item["prompt_tokens"] for item in results)
    request_latencies = [item["elapsed_s"] for item in results]

    return {
        "concurrency": concurrency,
        "elapsed_s": elapsed,
        "requests_per_s": concurrency / elapsed if elapsed else 0.0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "aggregate_completion_tokens_per_s": completion_tokens / elapsed if elapsed else 0.0,
        "mean_request_latency_s": statistics.fmean(request_latencies),
        "p50_request_latency_s": percentile(request_latencies, 0.50),
        "p95_request_latency_s": percentile(request_latencies, 0.95),
        "finish_reasons": [item["finish_reason"] for item in results],
        "sample_output": results[0]["content"],
        "raw_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="glm-5.1-q3ks")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--server-wait-timeout", type=float, default=7200)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--output-dir", default="/home/catid/glm51/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wait_for_server(args.base_url, args.server_wait_timeout)

    warmup = one_request(
        args.base_url,
        args.model,
        args.max_tokens,
        args.timeout,
        args.disable_thinking,
    )
    warmup_path = output_dir / "warmup_response.json"
    warmup_path.write_text(json.dumps(warmup["response"], indent=2), encoding="utf-8")

    generated_code = extract_cpp_source(warmup["content"])
    validation = validate_generated_code(generated_code, output_dir / "validation")
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output_dir / "generated_primes_response.txt").write_text(warmup["content"], encoding="utf-8")
    (output_dir / "generated_primes.cpp").write_text(generated_code, encoding="utf-8")

    if not validation.get("ok"):
        summary = {
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "disable_thinking": args.disable_thinking,
            "warmup_tokens": {
                "prompt_tokens": warmup["prompt_tokens"],
                "completion_tokens": warmup["completion_tokens"],
            },
            "validation": validation,
            "benchmarks": [],
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"validation_failed": validation}, indent=2), file=sys.stderr)
        return 1

    benchmarks = []
    for concurrency in args.concurrency:
        result = run_concurrency(
            args.base_url,
            args.model,
            args.max_tokens,
            args.timeout,
            concurrency,
            args.disable_thinking,
        )
        result_path = output_dir / f"benchmark_{concurrency}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        benchmarks.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"raw_results", "sample_output"}
            }
        )
        print(
            json.dumps(
                {
                    "concurrency": result["concurrency"],
                    "elapsed_s": round(result["elapsed_s"], 3),
                    "completion_tokens": result["completion_tokens"],
                    "aggregate_completion_tokens_per_s": round(result["aggregate_completion_tokens_per_s"], 3),
                    "requests_per_s": round(result["requests_per_s"], 3),
                    "p50_request_latency_s": round(result["p50_request_latency_s"], 3),
                    "p95_request_latency_s": round(result["p95_request_latency_s"], 3),
                }
            )
        )
        sys.stdout.flush()

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "disable_thinking": args.disable_thinking,
        "warmup_tokens": {
            "prompt_tokens": warmup["prompt_tokens"],
            "completion_tokens": warmup["completion_tokens"],
        },
        "validation": validation,
        "benchmarks": benchmarks,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
