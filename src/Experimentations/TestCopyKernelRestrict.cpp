/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"

#include <memory>

void TestCopyKernelRestrict()
{
    std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx = std::make_shared<HIPRTOrochiCtx>(0);

    GPUKernel test_copy_kernel;
    test_copy_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Experimentations/TestCopyKernelRestrict.h");
    test_copy_kernel.set_kernel_function_name("TestCopyKernelRestrict");
    test_copy_kernel.compile(hiprt_orochi_ctx);

#define BUFFER_SIZE 1000000000
#define ITERATIONS 100
    OrochiBuffer<float> buffer_a(BUFFER_SIZE);
    OrochiBuffer<float> buffer_b(BUFFER_SIZE);
    OrochiBuffer<float> buffer_c(BUFFER_SIZE);
    OrochiBuffer<float> buffer_d(BUFFER_SIZE);

    buffer_a.memset_whole_buffer(1);
    buffer_b.memset_whole_buffer(1);
    buffer_c.memset_whole_buffer(1);
    buffer_d.memset_whole_buffer(1);

    size_t buffer_size = BUFFER_SIZE;
    void* launch_args[] = { buffer_a.get_device_pointer_address(), buffer_b.get_device_pointer_address(), buffer_c.get_device_pointer_address(), buffer_d.get_device_pointer_address(), &buffer_size };

    float min_exec_time = 1000000.0f;
    float max_exec_time = 0.0f;

    for (int i = 0; i < ITERATIONS; i++)
    {
        float execution_time;
        test_copy_kernel.launch_synchronous(256, 1, BUFFER_SIZE, 1, launch_args, &execution_time);

        min_exec_time = hippt::min(execution_time, min_exec_time);
        max_exec_time = hippt::max(execution_time, max_exec_time);
    }

    std::cout << "Min/max exec time:" << min_exec_time << " / " << max_exec_time << " ms" << std::endl;
}