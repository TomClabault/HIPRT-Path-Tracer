/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "HostDeviceCommon/KernelOptions.h"

#include <cassert>

const std::string GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY = "InteriorStackStrategy";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY = "DirectLightSamplingStrategy";
const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY = "EnvmapSamplingStrategy";
const std::string GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION = "RISUseVisiblityTargetFunction";
const std::string GPUKernelCompilerOptions::GGX_SAMPLE_FUNCTION = "GGXAnisotropicSampleFunction";
const std::string GPUKernelCompilerOptions::RESTIR_DI_TARGET_FUNCTION_VISIBILITY = "ReSTIR_DI_TargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE = "ReSTIR_DI_DoVisibilityReuse";
const std::string GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_REUSE_BIAS_CORRECTION_USE_VISIBILITY = "ReSTIR_DI_BiasCorrectionUseVisiblity";
const std::string GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS = "ReSTIR_DI_BiasCorrectionWeights";
const std::string GPUKernelCompilerOptions::BSDF_OVERRIDE = "BSDFOverride";

const std::vector<std::string> GPUKernelCompilerOptions::ALL_MACROS_NAMES = {
	GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY,
	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION,
	GPUKernelCompilerOptions::GGX_SAMPLE_FUNCTION,
	GPUKernelCompilerOptions::RESTIR_DI_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE,
	GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_REUSE_BIAS_CORRECTION_USE_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS,
	GPUKernelCompilerOptions::BSDF_OVERRIDE
};

GPUKernelCompilerOptions::GPUKernelCompilerOptions()
{
	// Mandatory options that every kernel must have so we're
	// adding them here with their default values
	m_options_macro_map[GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY] = InteriorStackStrategy;
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY] = DirectLightSamplingStrategy;
	m_options_macro_map[GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY] = EnvmapSamplingStrategy;
	m_options_macro_map[GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION] = RISUseVisiblityTargetFunction;
	m_options_macro_map[GPUKernelCompilerOptions::GGX_SAMPLE_FUNCTION] = GGXAnisotropicSampleFunction;
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_TARGET_FUNCTION_VISIBILITY] = ReSTIR_DI_TargetFunctionVisibility;
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE] = ReSTIR_DI_DoVisibilityReuse;
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_REUSE_BIAS_CORRECTION_USE_VISIBILITY] = ReSTIR_DI_BiasCorrectionUseVisiblity;
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS] = ReSTIR_DI_BiasCorrectionWeights;
	m_options_macro_map[GPUKernelCompilerOptions::BSDF_OVERRIDE] = BSDFOverride;

	// Making sure we didn't forget to fill the ALL_MACROS_NAMES vector with all the options that exist
	assert(GPUKernelCompilerOptions::ALL_MACROS_NAMES.size() == m_options_macro_map.size());
}

std::vector<std::string> GPUKernelCompilerOptions::get_all_macros_as_std_vector_string()
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_options_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	for (auto macro_key_value : m_custom_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	return macros;
}

std::vector<std::string> GPUKernelCompilerOptions::get_relevant_macros_as_std_vector_string(const GPUKernel& kernel)
{
	std::vector<std::string> macros;

	// Looping on all the options macros and checking if the kernel uses that option macro,
	// only adding the macro to the returned vector if the kernel uses that option macro
	for (auto macro_key_value : m_options_macro_map)
		if (kernel.uses_macro(macro_key_value.first))
			macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	// Adding all the custom macros without conditions
	for (auto macro_key_value : m_custom_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	std::vector<std::string> additional_macros = kernel.get_additional_compiler_macros();
	for (const std::string& additional_macro : additional_macros)
		macros.push_back(additional_macro);

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
	if (m_options_macro_map.find(name) != m_options_macro_map.end())
		// If you could find the name in the options-macro, settings its value
		m_options_macro_map[name] = value;
	else
		// Otherwise, this is a user defined macro, putting it in the custom macro map
		m_custom_macro_map[name] = value;
}

void GPUKernelCompilerOptions::remove_macro(const std::string& name)
{
	// Only removing from the custom macro map because we cannot remove the options-macro
	m_custom_macro_map.erase(name);
}

bool GPUKernelCompilerOptions::has_macro(const std::string& name)
{
	// Only checking the custom macro map because we cannot remove the options-macro so it makes
	// no sense to check whether this instance has the macro "InteriorStackStrategy"
	// for example, it will always be yes
	return m_custom_macro_map.find(name) != m_custom_macro_map.end();
}

int GPUKernelCompilerOptions::get_macro_value(const std::string& name)
{
	auto find = m_options_macro_map.find(name);

	if (find == m_options_macro_map.end())
	{
		// Wasn't found in the options-macro, trying in the custom macros
		auto find_custom = m_custom_macro_map.find(name);
		if (find_custom == m_custom_macro_map.end())
			return std::numeric_limits<int>::min();
		else
			return find_custom->second;
	}
	else
		return find->second;
}

int* GPUKernelCompilerOptions::get_pointer_to_macro_value(const std::string& name)
{
	auto find = m_options_macro_map.find(name);

	if (find == m_options_macro_map.end())
	{
		// Wasn't found in the options-macro, trying in the custom macros
		auto find_custom = m_custom_macro_map.find(name);
		if (find_custom == m_custom_macro_map.end())
			return nullptr;
		else
			return &find_custom->second;
	}
	else
		return &find->second;
}
