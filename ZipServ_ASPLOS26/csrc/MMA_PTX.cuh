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
#include "TilingConfig.h"

// BF16 version of the B matrix load function
template<int NumOfTensors, int N8>
__device__ __forceinline__ void B_FragLoadFromSharedToRegisters_BF16(uint32_t __restrict__ Registers[][4],
                                                                __nv_bfloat16* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{
    int      lane_id             = threadIdx.x % 32;
    int      i                   = lane_id % 8;
    uint32_t Mask_RowPermutation = i << 4;    
    if (lane_id > 15)
        i += 8;
    int j = (lane_id % 16) >= 8 ? 1 : 0;
    
    smem += TILE_K * (warp_start_row + i) + (k_offset + j * HALF_PER_128B);
    uint32_t __restrict__ smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    
    smem_local_ptr = smem_local_ptr ^ Mask_RowPermutation;

    #pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
                     : "r"(smem_local_ptr));

        smem_local_ptr += TILE_K * MMA_N * sizeof(__nv_bfloat16);
    }
}
// BF16 version of the MMA instruction
__device__ __forceinline__ void
MMA_BF16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(b[0]),
                  "r"(b[1]),  // For column-major B matrix
                   "r"(c[0]),
                   "r"(c[1]),
                   "r"(c[2]),
                   "r"(c[3]));
}