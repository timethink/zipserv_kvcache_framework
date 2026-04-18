#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils.h"
#include "../build/L_API.cuh"

namespace {

size_t compressed_size_bytes(
    int sign_mantissa_bytes,
    int full_count,
    int num_tiles,
    int num_median_tiles,
    int num_global_tiles) {
    return static_cast<size_t>(sign_mantissa_bytes) +
           static_cast<size_t>(full_count) * sizeof(__nv_bfloat16) +
           static_cast<size_t>(num_tiles) * sizeof(uint64_t) * 3 +
           static_cast<size_t>(num_median_tiles) * 2 * sizeof(int) +
           static_cast<size_t>(num_global_tiles + 1) * 2 * sizeof(int);
}

bool read_u16_file(const std::string& path, std::vector<uint16_t>& data, size_t expected_elems) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "failed_to_open_input=" << path << "\n";
        return false;
    }

    data.resize(expected_elems);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_elems * sizeof(uint16_t)));
    if (!in || static_cast<size_t>(in.gcount()) != expected_elems * sizeof(uint16_t)) {
        std::cerr << "input_size_mismatch expected_bytes=" << (expected_elems * sizeof(uint16_t))
                  << " got_bytes=" << in.gcount() << "\n";
        return false;
    }
    return true;
}

bool write_u16_file(const std::string& path, const std::vector<uint16_t>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "failed_to_open_output=" << path << "\n";
        return false;
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(uint16_t)));
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " <input_u16_bin> <M> <K> <output_u16_bin>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const int M = std::stoi(argv[2]);
    const int K = std::stoi(argv[3]);
    const std::string output_path = argv[4];

    if (M <= 0 || K <= 0 || (M % 64) != 0 || (K % 64) != 0) {
        std::cerr << "invalid_shape_require_mk_multiple_of_64 M=" << M << " K=" << K << "\n";
        return 1;
    }

    const int num_tiles = (M / 8) * (K / 8);
    const int num_median_tiles = (M / 16) * (K / 64);
    const int num_global_tiles = (M / 64) * (K / 64);

    std::vector<uint16_t> input_u16;
    if (!read_u16_file(input_path, input_u16, static_cast<size_t>(M) * static_cast<size_t>(K))) {
        return 1;
    }

    std::vector<__nv_bfloat16> A_original(static_cast<size_t>(M) * static_cast<size_t>(K));
    for (size_t i = 0; i < input_u16.size(); ++i) {
        A_original[i] = __ushort_as_bfloat16(input_u16[i]);
    }

    __nv_bfloat16* top_exponents_cpu = nullptr;
    __nv_bfloat16* compressed_full_cpu = nullptr;
    uint8_t* sign_mantissa_cpu = nullptr;
    uint64_t* bitmap1_cpu = nullptr;
    uint64_t* bitmap2_cpu = nullptr;
    uint64_t* bitmap3_cpu = nullptr;
    int* TileOffsets_cpu = nullptr;
    int* TileOffsets_median_cpu = nullptr;
    int* TileOffsets_global_cpu = nullptr;
    int max_high_freq_count = 0;
    int max_full_count = 0;
    uint8_t start_exp = 0;

    const int num_global_tiles_ret = InitBF16MatrixTripleBitmap_Host(
        A_original.data(), M, K,
        8, 16, 64,
        8, 64, 64,
        &top_exponents_cpu,
        &compressed_full_cpu,
        &sign_mantissa_cpu,
        &bitmap1_cpu,
        &bitmap2_cpu,
        &bitmap3_cpu,
        &TileOffsets_cpu,
        &TileOffsets_median_cpu,
        &TileOffsets_global_cpu,
        max_high_freq_count,
        max_full_count,
        start_exp);

    if (num_global_tiles_ret <= 0) {
        std::cerr << "compression_failed\n";
        return 2;
    }

    const int high_freq_count = TileOffsets_global_cpu[num_global_tiles_ret * 2];
    const int full_count = TileOffsets_global_cpu[num_global_tiles_ret * 2 + 1];

    uint8_t* sign_mantissa_gpu = nullptr;
    __nv_bfloat16* compressed_full_gpu = nullptr;
    uint64_t* bitmap1_gpu = nullptr;
    uint64_t* bitmap2_gpu = nullptr;
    uint64_t* bitmap3_gpu = nullptr;
    int* TileOffsets_median_gpu = nullptr;
    int* TileOffsets_global_gpu = nullptr;
    __nv_bfloat16* output_gpu = nullptr;

    auto check_cuda = [](cudaError_t e, const char* step) {
        if (e != cudaSuccess) {
            std::cerr << "cuda_error step=" << step << " msg=" << cudaGetErrorString(e) << "\n";
            return false;
        }
        return true;
    };

    if (!check_cuda(cudaMalloc(&sign_mantissa_gpu, static_cast<size_t>(high_freq_count) * sizeof(uint8_t)), "cudaMalloc sign_mantissa")) return 3;
    if (!check_cuda(cudaMalloc(&compressed_full_gpu, static_cast<size_t>(full_count) * sizeof(__nv_bfloat16)), "cudaMalloc compressed_full")) return 3;
    if (!check_cuda(cudaMalloc(&bitmap1_gpu, static_cast<size_t>(num_tiles) * sizeof(uint64_t)), "cudaMalloc bitmap1")) return 3;
    if (!check_cuda(cudaMalloc(&bitmap2_gpu, static_cast<size_t>(num_tiles) * sizeof(uint64_t)), "cudaMalloc bitmap2")) return 3;
    if (!check_cuda(cudaMalloc(&bitmap3_gpu, static_cast<size_t>(num_tiles) * sizeof(uint64_t)), "cudaMalloc bitmap3")) return 3;
    if (!check_cuda(cudaMalloc(&TileOffsets_median_gpu, static_cast<size_t>(num_median_tiles) * 2 * sizeof(int)), "cudaMalloc TileOffsets_median")) return 3;
    if (!check_cuda(cudaMalloc(&TileOffsets_global_gpu, static_cast<size_t>(num_global_tiles_ret + 1) * 2 * sizeof(int)), "cudaMalloc TileOffsets_global")) return 3;
    if (!check_cuda(cudaMalloc(&output_gpu, static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16)), "cudaMalloc output")) return 3;

    if (!check_cuda(cudaMemcpy(sign_mantissa_gpu, sign_mantissa_cpu, static_cast<size_t>(high_freq_count) * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy sign_mantissa")) return 3;
    if (!check_cuda(cudaMemcpy(compressed_full_gpu, compressed_full_cpu, static_cast<size_t>(full_count) * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "cudaMemcpy compressed_full")) return 3;
    if (!check_cuda(cudaMemcpy(bitmap1_gpu, bitmap1_cpu, static_cast<size_t>(num_tiles) * sizeof(uint64_t), cudaMemcpyHostToDevice), "cudaMemcpy bitmap1")) return 3;
    if (!check_cuda(cudaMemcpy(bitmap2_gpu, bitmap2_cpu, static_cast<size_t>(num_tiles) * sizeof(uint64_t), cudaMemcpyHostToDevice), "cudaMemcpy bitmap2")) return 3;
    if (!check_cuda(cudaMemcpy(bitmap3_gpu, bitmap3_cpu, static_cast<size_t>(num_tiles) * sizeof(uint64_t), cudaMemcpyHostToDevice), "cudaMemcpy bitmap3")) return 3;
    if (!check_cuda(cudaMemcpy(TileOffsets_median_gpu, TileOffsets_median_cpu, static_cast<size_t>(num_median_tiles) * 2 * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy TileOffsets_median")) return 3;
    if (!check_cuda(cudaMemcpy(TileOffsets_global_gpu, TileOffsets_global_cpu, static_cast<size_t>(num_global_tiles_ret + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy TileOffsets_global")) return 3;
    if (!check_cuda(cudaMemset(output_gpu, 0, static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16)), "cudaMemset output")) return 3;

    cudaError_t err = BF16TripleBitmap_Decompress_API(
        0,
        sign_mantissa_gpu,
        compressed_full_gpu,
        bitmap1_gpu,
        bitmap2_gpu,
        bitmap3_gpu,
        TileOffsets_median_gpu,
        TileOffsets_global_gpu,
        max_high_freq_count,
        max_full_count,
        start_exp,
        output_gpu,
        M,
        K);

    if (err != cudaSuccess) {
        std::cerr << "decompress_failed=" << cudaGetErrorString(err) << "\n";
        return 4;
    }

    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after decompress")) return 4;

    std::vector<__nv_bfloat16> output(static_cast<size_t>(M) * static_cast<size_t>(K));
    if (!check_cuda(cudaMemcpy(output.data(), output_gpu, static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "cudaMemcpy output_to_host")) return 5;

    std::vector<uint16_t> output_u16(output.size());
    bool equal = true;
    for (size_t i = 0; i < output.size(); ++i) {
        output_u16[i] = __bfloat16_as_ushort(output[i]);
        if (output_u16[i] != input_u16[i]) {
            equal = false;
        }
    }

    if (!write_u16_file(output_path, output_u16)) {
        return 5;
    }

    const size_t original_bytes = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16);
    const size_t compressed_bytes = compressed_size_bytes(
        high_freq_count, full_count, num_tiles, num_median_tiles, num_global_tiles_ret);

    std::cout << "equal=" << (equal ? 1 : 0) << "\n";
    std::cout << "original_bytes=" << original_bytes << "\n";
    std::cout << "compressed_bytes=" << compressed_bytes << "\n";
    std::cout << "ratio=" << static_cast<double>(original_bytes) / static_cast<double>(compressed_bytes) << "\n";
    std::cout << "high_freq_count=" << high_freq_count << "\n";
    std::cout << "full_count=" << full_count << "\n";

    cudaFree(sign_mantissa_gpu);
    cudaFree(compressed_full_gpu);
    cudaFree(bitmap1_gpu);
    cudaFree(bitmap2_gpu);
    cudaFree(bitmap3_gpu);
    cudaFree(TileOffsets_median_gpu);
    cudaFree(TileOffsets_global_gpu);
    cudaFree(output_gpu);

    free(top_exponents_cpu);
    free(compressed_full_cpu);
    free(sign_mantissa_cpu);
    free(bitmap1_cpu);
    free(bitmap2_cpu);
    free(bitmap3_cpu);
    free(TileOffsets_cpu);
    free(TileOffsets_median_cpu);
    free(TileOffsets_global_cpu);

    return equal ? 0 : 6;
}
