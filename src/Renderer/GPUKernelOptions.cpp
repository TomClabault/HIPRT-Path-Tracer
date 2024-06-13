/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/KernelOptions.h"
#include "Renderer/GPUKernelOptions.h"

const std::string GPUKernelOptions::INTERIOR_STACK_STRATEGY = "InteriorStackStrategy";
const std::string GPUKernelOptions::DIRECT_LIGHT_SAMPLING_STRATEGY = "DirectLightSamplingStrategy";

GPUKernelOptions::GPUKernelOptions()
{
	// Adding options with their default values
	set_option(GPUKernelOptions::INTERIOR_STACK_STRATEGY, InteriorStackStrategy);
	set_option(GPUKernelOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, DirectLightSamplingStrategy);
}

std::vector<std::string> GPUKernelOptions::get_compiler_options()
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_macros)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	return macros;
}

void GPUKernelOptions::set_option(const std::string& name, int value)
{
	m_macros[name] = value;
}
