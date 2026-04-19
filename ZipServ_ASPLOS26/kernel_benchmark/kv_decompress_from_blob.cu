#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils.h"
#include "../build/L_API.cuh"

namespace {

struct Header {
    uint32_t magic;
    uint32_t version;
    int32_t M;
    int32_t K;
    int32_t num_tiles;
    int32_t num_median_tiles;
    int32_t num_global_tiles;
    int32_t high_freq_count;
    int32_t full_count;
    int32_t max_high_freq_count;
    int32_t max_full_count;
    uint8_t start_exp;
    uint8_t reserved[3];
};

constexpr uint32_t kMagic = 0x56534B5A;  // "ZKSV" little-endian

bool read_header(std::ifstream& in, Header& h) {
    in.read(reinterpret_cast<char*>(&h), sizeof(Header));
    return static_cast<bool>(in);
}

template <typename T>
bool read_raw(std::ifstream& in, std::vector<T>& out, size_t n) {
    out.resize(n);
    if (n == 0) {
        return true;
    }
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(in);
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
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <input_blob> <output_u16_bin>\n";
        return 1;
    }

    const std::string input_blob = argv[1];
    const std::string output_bin = argv[2];

    std::ifstream in(input_blob, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "failed_to_open_input_blob=" << input_blob << "\n";
        return 1;
    }

    Header h{};
    if (!read_header(in, h)) {
        std::cerr << "failed_to_read_header\n";
        return 1;
    }

    if (h.magic != kMagic) {
        std::cerr << "invalid_magic got=" << h.magic << " expected=" << kMagic << "\n";
        return 1;
    }
    if (h.version != 1) {
        std::cerr << "unsupported_version=" << h.version << "\n";
        return 1;
    }
    if (h.M <= 0 || h.K <= 0 || (h.M % 64) != 0 || (h.K % 64) != 0) {
        std::cerr << "invalid_shape M=" << h.M << " K=" << h.K << "\n";
        return 1;
    }

    std::vector<__nv_bfloat16> top_exponents;
    std::vector<uint8_t> sign_mantissa;
    std::vector<__nv_bfloat16> compressed_full;
    std::vector<uint64_t> bitmap1;
    std::vector<uint64_t> bitmap2;
    std::vector<uint64_t> bitmap3;
    std::vector<int> tile_offsets_median;
    std::vector<int> tile_offsets_global;

    if (!read_raw(in, top_exponents, 7)) {
        std::cerr << "failed_to_read_top_exponents\n";
        return 1;
    }
    if (!read_raw(in, sign_mantissa, static_cast<size_t>(h.high_freq_count))) {
        std::cerr << "failed_to_read_sign_mantissa\n";
        return 1;
    }
    if (!read_raw(in, compressed_full, static_cast<size_t>(h.full_count))) {
        std::cerr << "failed_to_read_compressed_full\n";
        return 1;
    }
    if (!read_raw(in, bitmap1, static_cast<size_t>(h.num_tiles))) {
        std::cerr << "failed_to_read_bitmap1\n";
        return 1;
    }
    if (!read_raw(in, bitmap2, static_cast<size_t>(h.num_tiles))) {
        std::cerr << "failed_to_read_bitmap2\n";
        return 1;
    }
    if (!read_raw(in, bitmap3, static_cast<size_t>(h.num_tiles))) {
        std::cerr << "failed_to_read_bitmap3\n";
        return 1;
    }
    if (!read_raw(in, tile_offsets_median, static_cast<size_t>(h.num_median_tiles) * 2)) {
        std::cerr << "failed_to_read_tile_offsets_median\n";
        return 1;
    }
    if (!read_raw(in, tile_offsets_global, static_cast<size_t>(h.num_global_tiles + 1) * 2)) {
        std::cerr << "failed_to_read_tile_offsets_global\n";
        return 1;
    }

    // Ensure there is no trailing payload mismatch.
    in.peek();
    if (!in.eof()) {
        std::cerr << "blob_has_trailing_bytes\n";
        return 1;
    }

    uint8_t* sign_mantissa_gpu = nullptr;
    __nv_bfloat16* compressed_full_gpu = nullptr;
    uint64_t* bitmap1_gpu = nullptr;
    uint64_t* bitmap2_gpu = nullptr;
    uint64_t* bitmap3_gpu = nullptr;
    int* tile_offsets_median_gpu = nullptr;
    int* tile_offsets_global_gpu = nullptr;
    __nv_bfloat16* output_gpu = nullptr;

    auto check_cuda = [](cudaError_t e, const char* step) {
        if (e != cudaSuccess) {
            std::cerr << "cuda_error step=" << step << " msg=" << cudaGetErrorString(e) << "\n";
            return false;
        }
        return true;
    };

    if (!check_cuda(cudaMalloc(&sign_mantissa_gpu, sign_mantissa.size() * sizeof(uint8_t)), "cudaMalloc sign_mantissa")) return 2;
    if (!check_cuda(cudaMalloc(&compressed_full_gpu, compressed_full.size() * sizeof(__nv_bfloat16)), "cudaMalloc compressed_full")) return 2;
    if (!check_cuda(cudaMalloc(&bitmap1_gpu, bitmap1.size() * sizeof(uint64_t)), "cudaMalloc bitmap1")) return 2;
    if (!check_cuda(cudaMalloc(&bitmap2_gpu, bitmap2.size() * sizeof(uint64_t)), "cudaMalloc bitmap2")) return 2;
    if (!check_cuda(cudaMalloc(&bitmap3_gpu, bitmap3.size() * sizeof(uint64_t)), "cudaMalloc bitmap3")) return 2;
    if (!check_cuda(cudaMalloc(&tile_offsets_median_gpu, tile_offsets_median.size() * sizeof(int)), "cudaMalloc tile_offsets_median")) return 2;
    if (!check_cuda(cudaMalloc(&tile_offsets_global_gpu, tile_offsets_global.size() * sizeof(int)), "cudaMalloc tile_offsets_global")) return 2;
    if (!check_cuda(cudaMalloc(&output_gpu, static_cast<size_t>(h.M) * static_cast<size_t>(h.K) * sizeof(__nv_bfloat16)), "cudaMalloc output")) return 2;

    if (!check_cuda(cudaMemcpy(sign_mantissa_gpu, sign_mantissa.data(), sign_mantissa.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy sign_mantissa")) return 2;
    if (!check_cuda(cudaMemcpy(compressed_full_gpu, compressed_full.data(), compressed_full.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "cudaMemcpy compressed_full")) return 2;
    if (!check_cuda(cudaMemcpy(bitmap1_gpu, bitmap1.data(), bitmap1.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "cudaMemcpy bitmap1")) return 2;
    if (!check_cuda(cudaMemcpy(bitmap2_gpu, bitmap2.data(), bitmap2.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "cudaMemcpy bitmap2")) return 2;
    if (!check_cuda(cudaMemcpy(bitmap3_gpu, bitmap3.data(), bitmap3.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "cudaMemcpy bitmap3")) return 2;
    if (!check_cuda(cudaMemcpy(tile_offsets_median_gpu, tile_offsets_median.data(), tile_offsets_median.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy tile_offsets_median")) return 2;
    if (!check_cuda(cudaMemcpy(tile_offsets_global_gpu, tile_offsets_global.data(), tile_offsets_global.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy tile_offsets_global")) return 2;
    if (!check_cuda(cudaMemset(output_gpu, 0, static_cast<size_t>(h.M) * static_cast<size_t>(h.K) * sizeof(__nv_bfloat16)), "cudaMemset output")) return 2;

    auto decompress_start = std::chrono::high_resolution_clock::now();
    cudaError_t err = BF16TripleBitmap_Decompress_API(
        0,
        sign_mantissa_gpu,
        compressed_full_gpu,
        bitmap1_gpu,
        bitmap2_gpu,
        bitmap3_gpu,
        tile_offsets_median_gpu,
        tile_offsets_global_gpu,
        h.max_high_freq_count,
        h.max_full_count,
        h.start_exp,
        output_gpu,
        h.M,
        h.K);

    if (err != cudaSuccess) {
        std::cerr << "decompress_failed=" << cudaGetErrorString(err) << "\n";
        return 3;
    }

    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after decompress")) return 3;
    auto decompress_end = std::chrono::high_resolution_clock::now();

    std::vector<__nv_bfloat16> output(static_cast<size_t>(h.M) * static_cast<size_t>(h.K));
    if (!check_cuda(cudaMemcpy(output.data(), output_gpu, output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "cudaMemcpy output_to_host")) return 4;

    std::vector<uint16_t> output_u16(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        output_u16[i] = __bfloat16_as_ushort(output[i]);
    }

    if (!write_u16_file(output_bin, output_u16)) {
        std::cerr << "write_output_bin_failed\n";
        return 5;
    }

    const double decompress_ms = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    std::cout << "M=" << h.M << "\n";
    std::cout << "K=" << h.K << "\n";
    std::cout << "high_freq_count=" << h.high_freq_count << "\n";
    std::cout << "full_count=" << h.full_count << "\n";
    std::cout << "decompress_ms=" << decompress_ms << "\n";

    cudaFree(sign_mantissa_gpu);
    cudaFree(compressed_full_gpu);
    cudaFree(bitmap1_gpu);
    cudaFree(bitmap2_gpu);
    cudaFree(bitmap3_gpu);
    cudaFree(tile_offsets_median_gpu);
    cudaFree(tile_offsets_global_gpu);
    cudaFree(output_gpu);

    return 0;
}
