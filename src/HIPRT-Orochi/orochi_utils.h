/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_UTILS_H
#define OROCHI_UTILS_H

#include "hiprt/hiprt.h"
#include "Orochi/Orochi.h"

#include <filesystem>
#include <optional>

#define OROCHI_CHECK_ERROR( error ) ( orochi_check_error( error, __FILE__, __LINE__ ) )
#define OROCHI_RTC_CHECK_ERROR( error ) ( orochi_rtc_check_error( error, __FILE__, __LINE__ ) )
#define HIPRT_CHECK_ERROR( error ) ( hiprt_check_error( error, __FILE__, __LINE__ ) )

bool readSourceCode(const std::filesystem::path& path, std::string& sourceCode, std::vector<std::filesystem::path>* includes = nullptr);

void buildTraceKernelFromBitcode(
	hiprtContext				   ctxt,
	const char* path,
	const char* functionName,
	oroFunction& functionOut,
	std::vector<std::string> include_paths,
	std::vector<std::pair<std::string, std::string>> precompiler_defines = {},
	std::vector<const char*>* opts = nullptr,
	std::vector<hiprtFuncNameSet>* funcNameSets = nullptr,
	uint32_t					   numGeomTypes = 0,
	uint32_t					   numRayTypes = 1);

void orochi_check_error(oroError res, const char* file, uint32_t line);
void orochi_rtc_check_error(orortcResult res, const char* file, uint32_t line);
void hiprt_check_error(hiprtError res, const char* file, uint32_t line);

#endif
