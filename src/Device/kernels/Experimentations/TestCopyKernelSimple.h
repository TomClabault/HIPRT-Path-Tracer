/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"

#include "HostDeviceCommon/Color.h"

#include <Orochi/Orochi.h>

using TEST_COPY_KERNEL_SIMPLE_BUFFER_TYPE = DeviceUnpackedEffectiveMaterial;

struct TestCopyKernelSimpleInputData
{
    TEST_COPY_KERNEL_SIMPLE_BUFFER_TYPE* buffer_a;
    TEST_COPY_KERNEL_SIMPLE_BUFFER_TYPE* buffer_b;
};

HIPRT_HOST_DEVICE HIPRT_INLINE void copy_function(const TEST_COPY_KERNEL_SIMPLE_BUFFER_TYPE* __restrict__ input_buffer, TEST_COPY_KERNEL_SIMPLE_BUFFER_TYPE* __restrict__ output_buffer, uint32_t tIdx)
{
    output_buffer[tIdx] = input_buffer[tIdx];
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) TestCopyKernelSimple(TestCopyKernelSimpleInputData input, size_t buffer_size)
#else
GLOBAL_KERNEL_SIGNATURE(void) TestCopyKernelSimple(TestCopyKernelSimpleInputData input, size_t buffer_size, int x)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    uint32_t offset = 13;
    uint32_t index = x + offset;

    if (index >= buffer_size)
        return;

    copy_function(input.buffer_b, input.buffer_a, index);
}
