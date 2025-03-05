/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernel.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/kernels/Experimentations/TestCopyKernelSimple.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HIPRT-Orochi/HIPRTOrochiCtx.h"

#include <memory>

void TestCopyKernelSimple()
{
    std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx = std::make_shared<HIPRTOrochiCtx>(0);

    GPUKernel test_copy_kernel;
    test_copy_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Experimentations/TestCopyKernelSimple.h");
    test_copy_kernel.set_kernel_function_name("TestCopyKernelSimple");
    test_copy_kernel.compile(hiprt_orochi_ctx);

#define BUFFER_SIZE 2560*1440
#define ITERATIONS 4000

    OrochiBuffer<TEST_COPY_KERNEL_SIMPLE_BUFFER_TYPE> buffer_a(BUFFER_SIZE);
    OrochiBuffer<TEST_COPY_KERNEL_SIMPLE_BUFFER_TYPE> buffer_b(BUFFER_SIZE);

    buffer_a.memset_whole_buffer(1);
    buffer_b.memset_whole_buffer(1);

    TestCopyKernelSimpleInputData input_data;
    input_data.buffer_a = buffer_a.get_device_pointer();
    input_data.buffer_b = buffer_b.get_device_pointer();

    size_t buffer_size = BUFFER_SIZE;
    void* launch_args[] = { &input_data, &buffer_size};

    float average_sum = 0.0f;
    float min_exec_time = 1000000.0f;
    float max_exec_time = 0.0f;

    for (int i = 0; i < ITERATIONS; i++)
    {
        float execution_time;
        test_copy_kernel.launch_synchronous(256, 1, BUFFER_SIZE, 1, launch_args, &execution_time);

        min_exec_time = hippt::min(execution_time, min_exec_time);
        max_exec_time = hippt::max(execution_time, max_exec_time);
        average_sum += execution_time;
    }

    std::cout << "Min/max/average exec time:" << min_exec_time << " / " << max_exec_time << " / " << average_sum / ITERATIONS << " ms" << std::endl;
}