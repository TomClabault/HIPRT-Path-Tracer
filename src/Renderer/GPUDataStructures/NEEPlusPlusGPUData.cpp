/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPUDataStructures/NEEPlusPlusGPUData.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

NEEPlusPlusGPUData::NEEPlusPlusGPUData()
{
	finalize_accumulation_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/NEE++/NEEPlusPlusFinalizeAccumulation.h");
	finalize_accumulation_kernel.set_kernel_function_name("NEEPlusPlusFinalizeAccumulation");
}

void NEEPlusPlusGPUData::compile_finalize_accumulation_kernel(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx)
{
	ThreadManager::start_thread(ThreadManager::COMPILE_NEE_PLUS_PLUS_FINALIZE_ACCUMULATION_KERNEL_KEY, ThreadFunctions::compile_kernel_no_func_sets, std::ref(finalize_accumulation_kernel), hiprt_orochi_ctx);
}

void NEEPlusPlusGPUData::recompile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, bool silent, bool use_cache)
{
	if (silent)
		finalize_accumulation_kernel.compile_silent(hiprt_orochi_ctx, {}, use_cache);
	else
		finalize_accumulation_kernel.compile(hiprt_orochi_ctx, {}, use_cache);
}