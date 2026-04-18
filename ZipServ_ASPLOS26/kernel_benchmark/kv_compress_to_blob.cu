#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

template <typename T>
bool write_raw(std::ofstream& out, const T* data, size_t n) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(out);
}

bool write_blob(
    const std::string& path,
    const Header& header,
    const __nv_bfloat16* top_exponents,
    const uint8_t* sign_mantissa,
    const __nv_bfloat16* compressed_full,
    const uint64_t* bitmap1,
    const uint64_t* bitmap2,
    const uint64_t* bitmap3,
    const int* tile_offsets_median,
    const int* tile_offsets_global) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "failed_to_open_output=" << path << "\n";
        return false;
    }

    out.write(reinterpret_cast<const char*>(&header), sizeof(Header));
    if (!out) return false;

    if (!write_raw(out, top_exponents, 7)) return false;
    if (!write_raw(out, sign_mantissa, static_cast<size_t>(header.high_freq_count))) return false;
    if (!write_raw(out, compressed_full, static_cast<size_t>(header.full_count))) return false;
    if (!write_raw(out, bitmap1, static_cast<size_t>(header.num_tiles))) return false;
    if (!write_raw(out, bitmap2, static_cast<size_t>(header.num_tiles))) return false;
    if (!write_raw(out, bitmap3, static_cast<size_t>(header.num_tiles))) return false;
    if (!write_raw(out, tile_offsets_median, static_cast<size_t>(header.num_median_tiles) * 2)) return false;
    if (!write_raw(out, tile_offsets_global, static_cast<size_t>(header.num_global_tiles + 1) * 2)) return false;

    return static_cast<bool>(out);
}

void choose_best_contiguous_top7(const __nv_bfloat16* data, int M, int K, int* top_exponents_out) {
    std::vector<uint64_t> hist(256, 0);
    const size_t total = static_cast<size_t>(M) * static_cast<size_t>(K);
    for (size_t i = 0; i < total; ++i) {
        const uint16_t bits = __bfloat16_as_ushort(data[i]);
        const uint8_t exp = static_cast<uint8_t>((bits >> 7) & 0xFF);
        hist[exp]++;
    }

    uint64_t best_sum = 0;
    int best_start = 0;
    uint64_t window_sum = 0;
    for (int i = 0; i < 7; ++i) {
        window_sum += hist[i];
    }
    best_sum = window_sum;

    for (int start = 1; start <= 249; ++start) {
        window_sum -= hist[start - 1];
        window_sum += hist[start + 6];
        if (window_sum > best_sum) {
            best_sum = window_sum;
            best_start = start;
        }
    }

    for (int i = 0; i < 7; ++i) {
        top_exponents_out[i] = best_start + i;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " <input_u16_bin> <M> <K> <output_blob>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const int M = std::stoi(argv[2]);
    const int K = std::stoi(argv[3]);
    const std::string output_blob = argv[4];

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

    std::vector<__nv_bfloat16> A(static_cast<size_t>(M) * static_cast<size_t>(K));
    for (size_t i = 0; i < input_u16.size(); ++i) {
        A[i] = __ushort_as_bfloat16(input_u16[i]);
    }

    int top_exponents_i32[7] = {0};
    choose_best_contiguous_top7(A.data(), M, K, top_exponents_i32);

    uint8_t* sign_mantissa = nullptr;
    __nv_bfloat16* compressed_full = nullptr;
    uint64_t* bitmap1 = nullptr;
    uint64_t* bitmap2 = nullptr;
    uint64_t* bitmap3 = nullptr;
    int* tile_offsets = nullptr;
    int* tile_offsets_median = nullptr;
    int* tile_offsets_global = nullptr;
    int max_high_freq_count = 0;
    int max_full_count = 0;

    auto compress_start = std::chrono::high_resolution_clock::now();
    const int num_global_tiles_ret = InitBF16MatrixTripleBitmap(
        A.data(),
        M,
        K,
        8,
        16,
        64,
        8,
        64,
        64,
        top_exponents_i32,
        &sign_mantissa,
        &compressed_full,
        &bitmap1,
        &bitmap2,
        &bitmap3,
        &tile_offsets,
        &tile_offsets_median,
        &tile_offsets_global,
        max_high_freq_count,
        max_full_count);
    auto compress_end = std::chrono::high_resolution_clock::now();

    if (num_global_tiles_ret <= 0) {
        std::cerr << "compression_failed\n";
        return 2;
    }

    std::vector<__nv_bfloat16> top_exponents_bf16(7);
    for (int i = 0; i < 7; ++i) {
        const uint16_t exp_bits = static_cast<uint16_t>((top_exponents_i32[i] & 0xFF) << 7);
        top_exponents_bf16[i] = __ushort_as_bfloat16(exp_bits);
    }

    const int high_freq_count = tile_offsets_global[num_global_tiles_ret * 2];
    const int full_count = tile_offsets_global[num_global_tiles_ret * 2 + 1];

    Header h{};
    h.magic = kMagic;
    h.version = 1;
    h.M = M;
    h.K = K;
    h.num_tiles = num_tiles;
    h.num_median_tiles = num_median_tiles;
    h.num_global_tiles = num_global_tiles_ret;
    h.high_freq_count = high_freq_count;
    h.full_count = full_count;
    h.max_high_freq_count = max_high_freq_count;
    h.max_full_count = max_full_count;
    h.start_exp = static_cast<uint8_t>(top_exponents_i32[0] - 1);

    if (!write_blob(
            output_blob,
            h,
            top_exponents_bf16.data(),
            sign_mantissa,
            compressed_full,
            bitmap1,
            bitmap2,
            bitmap3,
            tile_offsets_median,
            tile_offsets_global)) {
        std::cerr << "write_blob_failed path=" << output_blob << "\n";
        return 3;
    }

    const size_t original_bytes = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16);
    const size_t compressed_bytes = compressed_size_bytes(
        high_freq_count, full_count, num_tiles, num_median_tiles, num_global_tiles_ret);
    const double compress_ms = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
    const double compress_speed_mib_s =
        (compress_ms > 0.0)
            ? (static_cast<double>(original_bytes) / (1024.0 * 1024.0)) / (compress_ms / 1000.0)
            : 0.0;

    std::cout << "original_bytes=" << original_bytes << "\n";
    std::cout << "compressed_bytes=" << compressed_bytes << "\n";
    std::cout << "ratio=" << (static_cast<double>(compressed_bytes) / static_cast<double>(original_bytes)) << "\n";
    std::cout << "compress_ms=" << compress_ms << "\n";
    std::cout << "compress_speed_mib_s=" << compress_speed_mib_s << "\n";
    std::cout << "high_freq_count=" << high_freq_count << "\n";
    std::cout << "full_count=" << full_count << "\n";

    free(sign_mantissa);
    free(compressed_full);
    free(bitmap1);
    free(bitmap2);
    free(bitmap3);
    free(tile_offsets);
    free(tile_offsets_median);
    free(tile_offsets_global);

    return 0;
}
