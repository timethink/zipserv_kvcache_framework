import argparse
import json
import math
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

FRAMEWORK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRAMEWORK_ROOT.parent
ZIPSERV_ROOT = FRAMEWORK_ROOT / "ZipServ_ASPLOS26"
BUILD_DIR = ZIPSERV_ROOT / "build"
BENCH_DIR = ZIPSERV_ROOT / "kernel_benchmark"
HELPER_SRC = BENCH_DIR / "kv_compress_to_blob.cu"
HELPER_BIN = BENCH_DIR / "kv_compress_to_blob"


def run_cmd(cmd, cwd=None, env=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\\n{result.stdout}"
        )
    return result.stdout


def ensure_zipserv_built():
    if not BUILD_DIR.exists():
        raise FileNotFoundError(BUILD_DIR)
    run_cmd(["make"], cwd=BUILD_DIR)


def ensure_helper_built():
    if not HELPER_SRC.exists():
        raise FileNotFoundError(HELPER_SRC)

    cmd = [
        "nvcc",
        "-O3",
        "--std=c++14",
        "-gencode",
        "arch=compute_80,code=sm_80",
        "-I/usr/local/cuda/include",
        "-I../build",
        "-L../build",
        "-lL_API",
        "-lcublas",
        "-lcusparse",
        "-Xlinker",
        "-rpath,$ORIGIN/../build",
        str(HELPER_SRC.name),
        "-o",
        str(HELPER_BIN.name),
    ]
    run_cmd(cmd, cwd=BENCH_DIR)


def choose_kv_tensors(tensors, tensor_regex, max_tensors):
    keys = [name for name in tensors.keys() if ".key" in name or ".value" in name]
    if tensor_regex:
        pat = re.compile(tensor_regex)
        keys = [name for name in keys if pat.search(name)]
    if max_tensors > 0:
        keys = keys[:max_tensors]
    return keys


def pad_to_64_2d(t: torch.Tensor):
    rows = int(math.prod(t.shape[:-1]))
    cols = int(t.shape[-1])
    t2d = t.reshape(rows, cols).contiguous()
    rows_pad = ((rows + 63) // 64) * 64
    cols_pad = ((cols + 63) // 64) * 64

    padded = torch.zeros((rows_pad, cols_pad), dtype=torch.bfloat16)
    padded[:rows, :cols] = t2d
    return padded, rows, cols


def bf16_to_u16_numpy(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)


def parse_helper_output(text: str):
    stats = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.strip().split("=", 1)
            stats[k] = v
    return stats


def compress_one_tensor(name, tensor, out_dir: Path, tmp_dir: Path, gpu_id: int):
    if tensor.dtype != torch.bfloat16:
        return {
            "tensor": name,
            "status": "skipped",
            "reason": f"dtype={tensor.dtype} only bfloat16 supported by ZipServ path",
        }

    padded, rows, cols = pad_to_64_2d(tensor)
    arr = bf16_to_u16_numpy(padded)

    safe_name = name.replace("/", "_").replace(".", "_")
    input_bin = tmp_dir / f"{safe_name}.u16.bin"
    output_blob = out_dir / f"{safe_name}.zskv"
    arr.tofile(input_bin)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    out = run_cmd(
        [
            str(HELPER_BIN),
            str(input_bin),
            str(padded.shape[0]),
            str(padded.shape[1]),
            str(output_blob),
        ],
        cwd=BENCH_DIR,
        env=env,
    )

    stats = parse_helper_output(out)
    return {
        "tensor": name,
        "status": "ok",
        "orig_shape": list(tensor.shape),
        "packed_shape": [int(padded.shape[0]), int(padded.shape[1])],
        "orig_rows": rows,
        "orig_cols": cols,
        "dtype": str(tensor.dtype),
        "gpu": gpu_id,
        "blob": str(output_blob),
        "original_bytes": int(float(stats.get("original_bytes", "0"))),
        "compressed_bytes": int(float(stats.get("compressed_bytes", "0"))),
        "ratio": float(stats.get("ratio", "nan")),
        "compress_ms": float(stats.get("compress_ms", "nan")),
        "compress_speed_mib_s": float(stats.get("compress_speed_mib_s", "nan")),
        "high_freq_count": int(float(stats.get("high_freq_count", "0"))),
        "full_count": int(float(stats.get("full_count", "0"))),
    }


def main():
    parser = argparse.ArgumentParser(description="ZipServ KV cache GPU-parallel compression framework")
    parser.add_argument("--input", required=True, type=Path, help="input safetensors file")
    parser.add_argument("--out-dir", type=Path, default=FRAMEWORK_ROOT / "kvcache_data" / "zipserv_blobs")
    parser.add_argument("--tmp-dir", type=Path, default=FRAMEWORK_ROOT / ".zipserv_tmp_compress")
    parser.add_argument("--tensor-regex", type=str, default="")
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="1", help="comma separated gpu ids, e.g. 0,1")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No valid GPU ids")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    ensure_zipserv_built()
    ensure_helper_built()

    tensors = load_file(str(args.input))
    selected = choose_kv_tensors(tensors, args.tensor_regex, args.max_tensors)
    if not selected:
        raise ValueError("No KV tensors matched")

    print(f"selected_tensors={len(selected)}")
    futures = []
    results = []

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, name in enumerate(selected):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            futures.append(ex.submit(compress_one_tensor, name, tensors[name], args.out_dir, args.tmp_dir, gpu_id))

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res["status"] == "ok":
                print(
                    f"ok tensor={res['tensor']} gpu={res['gpu']} ratio={res['ratio']:.4f} "
                    f"compress_ms={res['compress_ms']:.3f} speed_mib_s={res['compress_speed_mib_s']:.2f}"
                )
            else:
                print(f"skip tensor={res['tensor']} reason={res.get('reason', '')}")
    wall_end = time.perf_counter()

    ok = [r for r in results if r["status"] == "ok"]
    total_original = sum(r["original_bytes"] for r in ok)
    total_compressed = sum(r["compressed_bytes"] for r in ok)
    total_ratio = (total_compressed / total_original) if total_original > 0 else float("nan")
    total_compress_ms_sum = sum(r.get("compress_ms", 0.0) for r in ok)
    avg_compress_ms = (total_compress_ms_sum / len(ok)) if ok else float("nan")
    avg_speed_mib_s = (
        sum(r.get("compress_speed_mib_s", 0.0) for r in ok) / len(ok)
        if ok
        else float("nan")
    )
    wall_ms = (wall_end - wall_start) * 1000.0
    wall_speed_mib_s = (
        (total_original / (1024.0 * 1024.0)) / ((wall_end - wall_start))
        if total_original > 0 and wall_end > wall_start
        else float("nan")
    )

    manifest = {
        "input": str(args.input),
        "num_selected": len(selected),
        "num_ok": len(ok),
        "num_skipped": len(results) - len(ok),
        "total_original_bytes": total_original,
        "total_compressed_bytes": total_compressed,
        "total_ratio": total_ratio,
        "total_compress_ms_sum": total_compress_ms_sum,
        "avg_compress_ms": avg_compress_ms,
        "avg_compress_speed_mib_s": avg_speed_mib_s,
        "wall_time_ms": wall_ms,
        "wall_speed_mib_s": wall_speed_mib_s,
        "results": sorted(results, key=lambda x: x["tensor"]),
    }

    manifest_path = args.out_dir / f"{args.input.stem}.zipserv_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print("=== summary ===")
    print(f"manifest={manifest_path}")
    print(f"num_ok={len(ok)}")
    print(f"total_ratio={total_ratio:.4f}")
    print(f"total_compress_ms_sum={total_compress_ms_sum:.3f}")
    print(f"avg_compress_ms={avg_compress_ms:.3f}")
    print(f"avg_compress_speed_mib_s={avg_speed_mib_s:.2f}")
    print(f"wall_time_ms={wall_ms:.3f}")
    print(f"wall_speed_mib_s={wall_speed_mib_s:.2f}")


if __name__ == "__main__":
    main()
