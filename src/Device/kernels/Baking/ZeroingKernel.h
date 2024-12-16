/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Math.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ZeroingKernel(float* buffer, int3 dims)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ZeroingKernel(float* buffer, int3 dims, int x, int y, int z)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
#endif

    const uint32_t pixel_index = (x + y * dims.x + z * dims.x * dims.y);

    if (x >= dims.x || y >= dims.y || z >= dims.z)
        return;

    buffer[pixel_index] = 0.0f;
}
