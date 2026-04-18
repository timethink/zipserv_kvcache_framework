import argparse
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

FRAMEWORK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRAMEWORK_ROOT.parent
ZIPSERV_ROOT = FRAMEWORK_ROOT / "ZipServ_ASPLOS26"
BUILD_DIR = ZIPSERV_ROOT / "build"
BENCH_DIR = ZIPSERV_ROOT / "kernel_benchmark"
HELPER_SRC = BENCH_DIR / "kv_roundtrip_from_bin.cu"
HELPER_BIN = BENCH_DIR / "kv_roundtrip_from_bin"


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
    if t.dtype != torch.bfloat16:
        raise TypeError(f"Expected bfloat16, got {t.dtype}")
    return t.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)


def u16_numpy_to_bf16_tensor(arr: np.ndarray, device="cpu") -> torch.Tensor:
    i16 = arr.view(np.int16)
    return torch.from_numpy(i16).to(device=device).view(torch.bfloat16)


def parse_helper_output(text: str):
    stats = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.strip().split("=", 1)
            stats[k] = v
    return stats


def roundtrip_one_tensor(name: str, tensor: torch.Tensor, tmp_dir: Path, cuda_device: int):
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Tensor {name} is {tensor.dtype}, ZipServ path requires bfloat16")

    padded, rows, cols = pad_to_64_2d(tensor)
    arr_in = bf16_to_u16_numpy(padded)

    input_bin = tmp_dir / f"{name.replace('/', '_').replace('.', '_')}.in.bin"
    output_bin = tmp_dir / f"{name.replace('/', '_').replace('.', '_')}.out.bin"
    arr_in.tofile(input_bin)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    out = run_cmd(
        [
            str(HELPER_BIN),
            str(input_bin),
            str(padded.shape[0]),
            str(padded.shape[1]),
            str(output_bin),
        ],
        cwd=BENCH_DIR,
        env=env,
    )

    stats = parse_helper_output(out)
    arr_out = np.fromfile(output_bin, dtype=np.uint16).reshape(padded.shape[0], padded.shape[1])

    same_bits = np.array_equal(arr_in, arr_out)
    out_tensor_padded = u16_numpy_to_bf16_tensor(arr_out.copy())
    out_tensor = out_tensor_padded[:rows, :cols].reshape(tensor.shape)
    same_tensor = torch.equal(out_tensor, tensor)

    ratio = float(stats.get("ratio", "nan"))
    helper_equal = stats.get("equal", "0") == "1"

    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "same_bits": same_bits,
        "same_tensor": same_tensor,
        "helper_equal": helper_equal,
        "ratio": ratio,
        "helper_output": out.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Use ZipServ for KV cache BF16 tensor roundtrip test")
    parser.add_argument("--input", required=True, type=Path, help="input safetensors")
    parser.add_argument("--tensor-regex", type=str, default="", help="regex to filter KV tensor names")
    parser.add_argument("--max-tensors", type=int, default=1, help="max number of KV tensors to test")
    parser.add_argument("--cuda-device", type=int, default=1, help="CUDA device index to use")
    parser.add_argument("--tmp-dir", type=Path, default=FRAMEWORK_ROOT / ".zipserv_tmp", help="temporary directory")
    args = parser.parse_args()

    tensors = load_file(str(args.input))
    selected = choose_kv_tensors(tensors, args.tensor_regex, args.max_tensors)
    if not selected:
        raise ValueError("No KV tensors matched")

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    ensure_zipserv_built()
    ensure_helper_built()

    all_ok = True
    print(f"selected_tensors={len(selected)}")
    for name in selected:
        result = roundtrip_one_tensor(name, tensors[name], args.tmp_dir, args.cuda_device)
        ok = result["same_bits"] and result["same_tensor"] and result["helper_equal"]
        all_ok = all_ok and ok
        print("---")
        print(f"tensor={result['name']}")
        print(f"shape={result['shape']}")
        print(f"ratio={result['ratio']:.4f}")
        print(f"helper_equal={result['helper_equal']}")
        print(f"same_bits={result['same_bits']}")
        print(f"same_tensor={result['same_tensor']}")

    print("=== summary ===")
    print(f"all_ok={all_ok}")
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
