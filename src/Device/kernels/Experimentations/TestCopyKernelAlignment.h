/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

 /**
  * Kernel for testing that creating and reading from a 3D texture happens correctly.
  * The 3D texture is just written to a linear buffer and the linear buffer is then
  * expected to contain the data of the texture, basically just a copy of it.
  */

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Color.h"

#include <Orochi/Orochi.h>

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) TestCopyKernelAlignment(ColorRGB32F* __restrict__ buffer_a, const ColorRGB32F* __restrict__ buffer_b, size_t buffer_size)
#else
GLOBAL_KERNEL_SIGNATURE(void) TestCopyKernelAlignment(ColorRGB32F* __restrict__ buffer_a, const ColorRGB32F* __restrict__ buffer_b, size_t buffer_size, int x)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    uint32_t offset = 13;
    uint32_t index = x + offset;

    if (index >= buffer_size)
        return;

    buffer_a[index] = buffer_b[index];
}
