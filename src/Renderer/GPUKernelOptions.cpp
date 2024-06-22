/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/KernelOptions.h"
#include "Renderer/GPUKernelOptions.h"

const std::string GPUKernelOptions::INTERIOR_STACK_STRATEGY = "InteriorStackStrategy";
const std::string GPUKernelOptions::DIRECT_LIGHT_SAMPLING_STRATEGY = "DirectLightSamplingStrategy";
const std::string GPUKernelOptions::ENVMAP_SAMPLING_STRATEGY = "EnvmapSamplingStrategy";
const std::string GPUKernelOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION = "RISUseVisiblityTargetFunction";

GPUKernelOptions::GPUKernelOptions()
{
	// Adding options with their default values
	set_option(GPUKernelOptions::INTERIOR_STACK_STRATEGY, InteriorStackStrategy);
	set_option(GPUKernelOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, DirectLightSamplingStrategy);
	set_option(GPUKernelOptions::ENVMAP_SAMPLING_STRATEGY, EnvmapSamplingStrategy);
	set_option(GPUKernelOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, RISUseVisiblityTargetFunction);
}

std::vector<std::string> GPUKernelOptions::get_compiler_options()
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_options_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	return macros;
}

void GPUKernelOptions::set_option(const std::string& name, int value)
{
	m_options_map[name] = value;
}

int GPUKernelOptions::get_option_value(const std::string& name)
{
	auto find = m_options_map.find(name);

	if (find == m_options_map.end())
		return -1;
	else
		return find->second;
}

int* GPUKernelOptions::get_pointer_to_option_value(const std::string& name)
{
	auto find = m_options_map.find(name);

	if (find == m_options_map.end())
		return nullptr;
	else
		return &find->second;
}
