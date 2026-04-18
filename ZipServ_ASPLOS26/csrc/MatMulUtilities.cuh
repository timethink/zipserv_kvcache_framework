/***************************************************************************
 * Copyright 2025 The ZipServ Authors. All rights reserved.
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
#ifndef MatMulUtilities_H
#define MatMulUtilities_H
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "AsyncCopy_PTX.cuh"
#include "MMA_PTX.cuh"
#include "TilingConfig.h"


// Load BF16 data from global memory to shared memory
template<int NumOfRowsToCopy, typename TilingConfig>  
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64_BF16(__nv_bfloat16* __restrict__ SharedPTR,
                                                                const __nv_bfloat16* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{
    if(Pred) {
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row1          = lane_id / 8;
    int row2          = lane_id / 8 + 4;
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_FP16_ROWS;
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
//
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  
        const __nv_bfloat16* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;
        __nv_bfloat16* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,
                     AsyncCopyPredictor);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor);
    }
    }
}


template<int NumOfRowsToCopy, typename TilingConfig>  
__device__ __forceinline__ void CopyTileFromGlobalToShared_X_64_BF16_Safe(__nv_bfloat16* __restrict__ SharedPTR,
                                                                const __nv_bfloat16* GlobalPTR,
                                                                const int   GlobalStride,
                                                                const int   N_global,      // New parameter
                                                                bool        Pred = true)
{
    if(Pred) {
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 8;
    int row1          = lane_id / 8;
    int row2          = lane_id / 8 + 4;
    int store_column1 = col ^ row1;
    int store_column2 = col ^ row2;
    
    int warp_id = threadIdx.x / 32;
    
    // === New: Calculate actual rows to copy for current block ===
    int block_start_col = blockIdx.x * TilingConfig::TILE_N2;
    int actual_rows_to_copy = min(NumOfRowsToCopy, N_global - block_start_col);
    
    // === Recalculate boundaries based on actual row count ===
    int TotalNumOfCopyUnit = actual_rows_to_copy / COPY_UNIT_FP16_ROWS;
    int RemainderRows = actual_rows_to_copy % COPY_UNIT_FP16_ROWS;
    int TotalUnitsToProcess = TotalNumOfCopyUnit + (RemainderRows > 0 ? 1 : 0);
    
    const int MaxIteration = (TotalUnitsToProcess - 1) / 
                            (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;

#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int COPY_UNIT_I = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalUnitsToProcess && Pred;
        
        // === Modified: Check based on actual boundaries ===
        bool isLastIncompleteUnit = (COPY_UNIT_I == TotalNumOfCopyUnit) && (RemainderRows > 0);
        bool row1_valid = true;
        bool row2_valid = true;
        
        if (isLastIncompleteUnit) {
            row1_valid = row1 < RemainderRows;
            row2_valid = row2 < RemainderRows;
        }
        
        // === New: Check if COPY_UNIT is within valid range ===
        int unit_start_row = COPY_UNIT_I * COPY_UNIT_FP16_ROWS;
        bool unit_valid = unit_start_row < actual_rows_to_copy;
        
        const __nv_bfloat16* GlobalPTR_Unit = GlobalPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * GlobalStride;
        __nv_bfloat16* __restrict__ SharedPTR_Unit = SharedPTR + COPY_UNIT_I * COPY_UNIT_FP16_ROWS * TILE_K;
        
        // === Modified: Combine all validity checks ===
        cp_async<16>(SharedPTR_Unit + store_column1 * HALF_PER_128B + row1 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row1 * GlobalStride,
                     AsyncCopyPredictor && row1_valid && unit_valid);
        cp_async<16>(SharedPTR_Unit + store_column2 * HALF_PER_128B + row2 * TILE_K,
                     GlobalPTR_Unit + col * HALF_PER_128B + row2 * GlobalStride,
                     AsyncCopyPredictor && row2_valid && unit_valid);
    }
    }
}

// Load three bitmaps to shared memory (64 bitmap)
template<int NumOfRowsToCopy, typename TilingConfig>
__device__ __forceinline__ void CopyTripleBitmapToShared(uint64_t* __restrict__ SharedBitmap1,
                                                         uint64_t* __restrict__ SharedBitmap2,
                                                         uint64_t* __restrict__ SharedBitmap3,
                                                         const uint64_t* GlobalBitmap1,
                                                         const uint64_t* GlobalBitmap2,
                                                         const uint64_t* GlobalBitmap3,
                                                         bool Pred = true)
{
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int TotalNumOfCopyUnit = NumOfRowsToCopy;

    bool AsyncCopyPredictor = warp_id < TotalNumOfCopyUnit && Pred;
    
    // Load three bitmaps
    cp_async<16>(SharedBitmap1 + lane_id * UINT64_PER_128B, 
                 GlobalBitmap1 + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
                 
    cp_async<16>(SharedBitmap2 + lane_id * UINT64_PER_128B, 
                 GlobalBitmap2 + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
                 
    cp_async<16>(SharedBitmap3 + lane_id * UINT64_PER_128B, 
                 GlobalBitmap3 + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
}


// Load three bitmaps to shared memory (128 bitmap)
template<int NumOfRowsToCopy, typename TilingConfig>
__device__ __forceinline__ void CopyTripleBitmapToShared_128(uint64_t* __restrict__ SharedBitmap1,
                                                         uint64_t* __restrict__ SharedBitmap2,
                                                         uint64_t* __restrict__ SharedBitmap3,
                                                         const uint64_t* GlobalBitmap1,
                                                         const uint64_t* GlobalBitmap2,
                                                         const uint64_t* GlobalBitmap3,
                                                         bool Pred = true)
{
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int TotalNumOfCopyUnit = NumOfRowsToCopy;

    bool AsyncCopyPredictor = warp_id < TotalNumOfCopyUnit && Pred;
    
    // Load three bitmaps
    cp_async<16>(SharedBitmap1 + (warp_id<<6) + lane_id * UINT64_PER_128B, 
                 GlobalBitmap1 + (warp_id<<6) + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
                 
    cp_async<16>(SharedBitmap2 + (warp_id<<6) + lane_id * UINT64_PER_128B, 
                 GlobalBitmap2 + (warp_id<<6) + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
                 
    cp_async<16>(SharedBitmap3 + (warp_id<<6) + lane_id * UINT64_PER_128B, 
                 GlobalBitmap3 + (warp_id<<6) + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
}

// Load three bitmaps into shared memory (128 bitmap)
template<int NumOfRowsToCopy, typename TilingConfig>
__device__ __forceinline__ void CopyTripleBitmapToShared_256(uint64_t* __restrict__ SharedBitmap1,
                                                         uint64_t* __restrict__ SharedBitmap2,
                                                         uint64_t* __restrict__ SharedBitmap3,
                                                         const uint64_t* GlobalBitmap1,
                                                         const uint64_t* GlobalBitmap2,
                                                         const uint64_t* GlobalBitmap3,
                                                         bool Pred = true)
{
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int TotalNumOfCopyUnit = NumOfRowsToCopy;

    bool AsyncCopyPredictor = warp_id < TotalNumOfCopyUnit && Pred;
    
    // Load three bitmaps
    cp_async<16>(SharedBitmap1 + (warp_id<<6) + lane_id * UINT64_PER_128B, 
                 GlobalBitmap1 + (warp_id<<6) + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
                 
    cp_async<16>(SharedBitmap2 + (warp_id<<6) + lane_id * UINT64_PER_128B, 
                 GlobalBitmap2 + (warp_id<<6) + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
                 
    cp_async<16>(SharedBitmap3 + (warp_id<<6) + lane_id * UINT64_PER_128B, 
                 GlobalBitmap3 + (warp_id<<6) + lane_id * UINT64_PER_128B,   
                 AsyncCopyPredictor);
}


// Load compressed data to shared memory
template<typename TilingConfig>
__device__ __forceinline__ void CopyCompressedDataToShared(uint8_t* __restrict__ SharedSignMantissa,
                                                           __nv_bfloat16* __restrict__ SharedFullValues,
                                                           const uint8_t* GlobalSignMantissa,
                                                           const __nv_bfloat16* GlobalFullValues,
                                                           const int HighFreqCount,
                                                           const int FullCount,
                                                           bool Pred = true)
{
    if(Pred) {
        int threadPerBlock = blockDim.x;
        // Load high-frequency elements (sign+mantissa), 16 elements per batch
        int HF_Batches = (HighFreqCount>>4);
        for(int i = threadIdx.x; i < HF_Batches; i += threadPerBlock) {
            // Complete 16 elements can be loaded with 64-bit load
            const uint8_t* GlobalPTR_Unit        =  GlobalSignMantissa + i * 16;  
            uint8_t* __restrict__ SharedPTR_Unit = SharedSignMantissa + i * 16; 
            cp_async<16>(SharedPTR_Unit, GlobalPTR_Unit, Pred);
        }
        
        // Load non-high-frequency elements (full values), 8 elements per batch
        int Full_Batches = (FullCount>>3);
        for(int i = threadIdx.x; i < Full_Batches; i += threadPerBlock) {
            // Complete 4 elements can be loaded with 64-bit load
            const __nv_bfloat16* GlobalPTR_Unit        =  GlobalFullValues + i * 8;  
            __nv_bfloat16* __restrict__ SharedPTR_Unit = SharedFullValues + i * 8; 
            cp_async<16>(SharedPTR_Unit, GlobalPTR_Unit, Pred);
        }
    }
}
template<typename TilingConfig>
__device__ __forceinline__ void
StoreToSharedMemoryFromRegisterBitmapV3(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                float c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V3);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V3;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

#endif