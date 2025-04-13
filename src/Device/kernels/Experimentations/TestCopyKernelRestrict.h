/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

 /**
  * Kernel for testing that creating and reading from a 3D texture happens correctly.
  * The 3D texture is just written to a linear buffer and the linear buffer is then
  * expected to contain the data of the texture, basically just a copy of it.
  */

#include "Device/includes/FixIntellisense.h"

#include <Orochi/Orochi.h>

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) TestCopyKernelRestrict(float* buffer_a, float* buffer_b, float* buffer_c, float* buffer_d, size_t buffer_size)
#else
GLOBAL_KERNEL_SIGNATURE(void) TestCopyKernelRestrict(float* __restrict__ buffer_a, const float* __restrict__ buffer_b, float* __restrict__ buffer_c, float* __restrict__ buffer_d, size_t buffer_size, int x)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (x >= buffer_size)
        return;

    buffer_a[x] = buffer_a[x] + buffer_b[x];
    buffer_d[x] = buffer_a[x] * buffer_b[x];
    buffer_d[x] *= buffer_c[x];
}
