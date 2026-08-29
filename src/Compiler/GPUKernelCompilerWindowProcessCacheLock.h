/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_CACHE_LOCK_H
#define GPU_KERNEL_COMPILER_WINDOW_PROCESS_CACHE_LOCK_H

#include "Compiler/GPUKernelCompilerWindowProcessCompilationRequest.h"

#include <cstddef>

class GPUKernelCompilerWindowProcessCacheLock
{
public:
	explicit GPUKernelCompilerWindowProcessCacheLock(const GPUKernelCompilerWindowProcessCompilationRequest& request);
	~GPUKernelCompilerWindowProcessCacheLock();

	bool acquired() const;

private:
#ifdef _WIN32
	static void update_request_hash(unsigned long long& request_hash, const void* data, size_t data_size);
	static void update_request_hash(unsigned long long& request_hash, const std::string& value);
	static void update_request_hash(unsigned long long& request_hash, int value);
	static void update_request_hash(unsigned long long& request_hash, unsigned int value);
	static void update_request_hash(unsigned long long& request_hash, bool value);
	static unsigned long long get_request_hash(const GPUKernelCompilerWindowProcessCompilationRequest& request);
#endif // _WIN32 // #ifdef _WIN32

	void* mutex_handle	= nullptr;
	bool mutex_acquired = false;
};

#endif // #ifndef GPU_KERNEL_COMPILER_WINDOW_PROCESS_CACHE_LOCK_H
