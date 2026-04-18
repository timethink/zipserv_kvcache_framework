/***************************************************************************
 * Copyright 2025 The ZipServ Authors. All rights reserved.
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#ifndef TILINGCONFIG_H
#define TILINGCONFIG_H
// Fixed Parameters
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define WARP_SIZE 32
// Unchangable
#define WARP_ROW_TENSORS 4
#define BLOCK_K_TENSORS 4
#define BLOCK_K_TENSORS_HALF 2
#define TILE_K (MMA_K * BLOCK_K_TENSORS)  // 64
#define TILE_K_HALF (TILE_K/2)   // 32

// Unchangable
#define WARP_ROW_TENSORS_BITMAP_V3 1
#define TILE_BITMAP_K (TILE_K/8)   // 8

// Parameters for copying A_TILE & B_TILE & C_TILE
#define COPY_UNIT_FP16_ROWS 8
#define COPY_UNIT_FP16_ROWS_16 16
#define COPY_UNIT_FP16_COLS 64
#define HALF_PER_128B 8           // LDS.128 -> 8 * FP16
#define UINT32_PER_128B 4           // LDS.128 -> 4 * uint32_t
#define UINT32_PER_64B 2           // LDS.128 -> 4 * uint32_t
#define UINT64_PER_128B 2           // LDS.128 -> 2 * uint64_t
#define REG_PER_C_TENSOR_16_16 8  // 8 for FP32 Accumulation; 4 for FP16 Accumulation

#define PADDING_SHARED_MEM_FOR_C 4  // Padding 8/2 float each column to eliminating bank-conflict in C fragments
#define PADDING_SHARED_MEM_FOR_DECOMP 8
#define SIGN_MANTISSA_PADDING 16
template<int BLOCK_ROW_WARPS_, int BLOCK_COL_WARPS_, int WARP_COL_TENSORS_, int N8_ = 0>
struct TilingConfigBF16TripleBitmap {
    static constexpr int BLOCK_ROW_WARPS  = BLOCK_ROW_WARPS_;
    static constexpr int BLOCK_COL_WARPS  = BLOCK_COL_WARPS_;
    static constexpr int WARP_COL_TENSORS = WARP_COL_TENSORS_;

    static constexpr int TILE_M        = MMA_M * (WARP_ROW_TENSORS_BITMAP_V3 * BLOCK_ROW_WARPS);
    static constexpr int TILE_BITMAP_M_V3 = 1; 
    static constexpr int TILE_BITMAP_K_V3 = 64;

    static constexpr int TILE_N        = MMA_N * (WARP_COL_TENSORS * BLOCK_COL_WARPS);
    static constexpr int BLOCK_WARPS   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
    static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;

    static constexpr int N8      = N8_;
    static constexpr int TILE_N2 = N8 ? 8 : TILE_N;
};

#endif