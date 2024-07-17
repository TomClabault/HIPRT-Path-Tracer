/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "HostDeviceCommon/KernelOptions.h"

const std::string GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY = "InteriorStackStrategy";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY = "DirectLightSamplingStrategy";
const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY = "EnvmapSamplingStrategy";
const std::string GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION = "RISUseVisiblityTargetFunction";

GPUKernelCompilerOptions::GPUKernelCompilerOptions()
{
	// Mandatory options that every kernel must have so we're
	// adding them here with their default values
	set_macro(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY, InteriorStackStrategy);
	set_macro(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, DirectLightSamplingStrategy);
	set_macro(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, EnvmapSamplingStrategy);
	set_macro(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, RISUseVisiblityTargetFunction);
}

std::vector<std::string> GPUKernelCompilerOptions::get_options_as_std_vector_string()
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	return macros;
}

const std::vector<std::string>& GPUKernelCompilerOptions::get_additional_include_directories() const
{
	return m_additional_include_directories;
}

std::vector<std::string> GPUKernelCompilerOptions::get_additional_include_directories()
{
	return m_additional_include_directories;
}

void GPUKernelCompilerOptions::set_additional_include_directories(const std::vector<std::string>& additional_include_directories)
{
	m_additional_include_directories = additional_include_directories;
}

void GPUKernelCompilerOptions::set_macro(const std::string& name, int value)
{
	m_macro_map[name] = value;
}

void GPUKernelCompilerOptions::remove_macro(const std::string& name)
{
	m_macro_map.erase(name);
}

bool GPUKernelCompilerOptions::has_macro(const std::string& name)
{
	return m_macro_map.find(name) != m_macro_map.end();
}

int GPUKernelCompilerOptions::get_macro_value(const std::string& name)
{
	auto find = m_macro_map.find(name);

	if (find == m_macro_map.end())
		return -1;
	else
		return find->second;
}

int* GPUKernelCompilerOptions::get_pointer_to_macro_value(const std::string& name)
{
	auto find = m_macro_map.find(name);

	if (find == m_macro_map.end())
		return nullptr;
	else
		return &find->second;
}
