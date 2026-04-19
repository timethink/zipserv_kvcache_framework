# ZipServ KVCache Framework

该目录用于集中管理 ZipServ KVCache 测评与压缩脚本。

## 内置依赖说明

本目录已内置 ZipServ 运行所需的最小源码依赖，位于：

- zipserv_kvcache_framework/ZipServ_ASPLOS26/

两个框架脚本只会使用该内置目录，不再引用外部同级目录 `../ZipServ_ASPLOS26`。

内置目录包含：

- build/Makefile, build/L_API.cuh
- csrc/ 下编译所需头文件和实现
- kernel_benchmark/kv_compress_to_blob.cu
- kernel_benchmark/kv_roundtrip_from_bin.cu
- kernel_benchmark/utils.h

注意：首次运行会在内置目录下触发 `make` 与 `nvcc` 编译，因此目标环境仍需具备 CUDA 工具链与相关库。

## 文件说明

- compress_framework.py
  - 对 safetensors 中 KV 张量做并行压缩，输出 zskv blob 和 manifest。
- decompress_framework.py
  - 从 zskv blob + manifest 解压恢复 KV 张量并回写 safetensors，可做 bit-exact 校验。
- roundtrip_framework.py
  - 对 KV 张量执行压缩/解压回环并验证 bit-exact。

## 详细指标文档

- docs/zipserv_kvcache_evaluation_guide.md

## 快速使用

在仓库根目录执行：

python zipserv_kvcache_framework/compress_framework.py \
  --input kvcache_data/narrativeqa_sample_0.safetensors \
  --max-tensors 4 \
  --gpus 1,2 \
  --workers 2

python zipserv_kvcache_framework/roundtrip_framework.py \
  --input kvcache_data/narrativeqa_sample_0.safetensors \
  --max-tensors 1 \
  --cuda-device 1

python zipserv_kvcache_framework/decompress_framework.py \
  --input kvcache_data/narrativeqa_sample_0.safetensors \
  --manifest zipserv_kvcache_framework/kvcache_data/zipserv_blobs/narrativeqa_sample_0.zipserv_manifest.json \
  --max-tensors 1 \
  --cuda-device 1

若将本文件夹单独复制到其他位置，也可直接在文件夹内执行：

python compress_framework.py --input /path/to/your.safetensors --max-tensors 4 --gpus 0
python roundtrip_framework.py --input /path/to/your.safetensors --max-tensors 1 --cuda-device 0
python decompress_framework.py --input /path/to/your.safetensors --manifest /path/to/your.zipserv_manifest.json --cuda-device 0
