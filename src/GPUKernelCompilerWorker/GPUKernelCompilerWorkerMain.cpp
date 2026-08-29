/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GPUKernelCompilerWorker/GPUKernelCompilerWorker.h"

#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
{
	if (argc != 3 || std::wstring(argv[1]) != L"--kernel-compiler-worker")
	{
		std::cout << "Usage: GPUKernelCompilerWorker.exe --kernel-compiler-worker <request-file>" << std::endl;
		return 1;
	}

	return GPUKernelCompilerWorker::run(std::filesystem::path(argv[2]));
}
#endif // _WIN32 // #ifdef _WIN32
