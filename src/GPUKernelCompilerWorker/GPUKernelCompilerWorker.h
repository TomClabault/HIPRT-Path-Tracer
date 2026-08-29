/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_COMPILER_WORKER_H
#define GPU_KERNEL_COMPILER_WORKER_H

#include "Compiler/GPUKernelCompilerWindowProcessCompilationRequest.h"

#include <Orochi/Orochi.h>
#include <hiprt/hiprt.h>

#include <filesystem>

class GPUKernelCompilerWorker
{
public:
	static int run(const std::filesystem::path& request_file_path);

private:
	static bool initialize_worker_context(int device_index, hiprtContext& hiprt_context_out, oroCtx& orochi_context_out);
	static void destroy_worker_context(hiprtContext hiprt_context, oroCtx orochi_context);
	static bool compile_in_worker(const GPUKernelCompilerWindowProcessCompilationRequest& request);
};

#endif // #ifndef GPU_KERNEL_COMPILER_WORKER_H
