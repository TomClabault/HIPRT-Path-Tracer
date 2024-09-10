/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_OPTIONS_H
#define GPU_KERNEL_OPTIONS_H

#include "Compiler/GPUKernel.h"

#include <string>
#include <unordered_map>
#include <vector>

class GPUKernelCompilerOptions
{
public:
	static const std::string INTERIOR_STACK_STRATEGY;
	static const std::string DIRECT_LIGHT_SAMPLING_STRATEGY;
	static const std::string ENVMAP_SAMPLING_STRATEGY;
	static const std::string RIS_USE_VISIBILITY_TARGET_FUNCTION;
	static const std::string GGX_SAMPLE_FUNCTION;
	static const std::string RESTIR_DI_TARGET_FUNCTION_VISIBILITY;
	static const std::string RESTIR_DI_DO_VISIBILITY_REUSE;
	static const std::string RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY;
	static const std::string RESTIR_DI_RAYTRACE_SPATIAL_REUSE_RESERVOIR;
	static const std::string RESTIR_DI_BIAS_CORRECTION_WEIGHTS;
	static const std::string BSDF_OVERRIDE;

	static const std::vector<std::string> ALL_MACROS_NAMES;

	GPUKernelCompilerOptions();

	/**
	 * Gets a list of all the compiler options of the form { "-D InteriorStackStrategy=1", ... }
	 * that can directly be passed to the kernel compiler.
	 * 
	 * The returned options do not contain additional include directories.
	 * Additional include directories are not considered options.
	 */
	std::vector<std::string> get_all_macros_as_std_vector_string();

	/**
	 * Same as get_all_macros_as_std_vector_string() but the returned vector doesn't contain
	 * the macros that do not apply to the kernel given in parameter.
	 * 
	 * For example, the camera rays kernel doesn't care about whether our direct lighting
	 * strategy is MIS, RIS, ReSTIR DI, ... so if a camera ray kernel is given in parameter
	 * the returned vector will not contain the macro for the direct lighting strategy.
	 * Same logic for the other macros defined in KernelOptions.h
	 * 
	 * The returned vector always contain all the "custom" macros manually defined through
	 * a call to 'set_macro()' (unless the macro changed through 'set_macro()' is an
	 * option macro defined in KernelOptions.h as defined above)
	 * 
	 * The returned vector also contains all the additional compiler macro that were added
	 * to the kernel by calling 'kernel.add_additional_macro_for_compilation()'
	 */
	std::vector<std::string> get_relevant_macros_as_std_vector_string(const GPUKernel& kernel);

	///@{
	/**
	 * Returns the list of additional include directories of these CompilerOptions
	 */
	const std::vector<std::string>& get_additional_include_directories() const;
	std::vector<std::string> get_additional_include_directories();
	///@}

	/**
	 * Replaces the current list of include directories with the one given
	 */
	void set_additional_include_directories(const std::vector<std::string>& additional_include_directories);

	/**
	 * Replace the value of the macro if it has already been added previous to this call
	 * 
	 * The @name parameter is expected to be given without the '-D' macro prefix.
	 * For example, if you want to define a macro "MyMacro" equal to 1, you simply
	 * call set_macro("MyMacro", 1).
	 * The addition of the -D prefix will be added internally.
	 */
	void set_macro(const std::string& name, int value);

	/**
	 * Removes a macro from the list given to the compiler
	 */
	void remove_macro(const std::string& name);

	/**
	 * Returns true if the given macro is defined. False otherwise
	 */
	bool has_macro(const std::string& name);

	/** 
	 * Gets the value of a macro or -1 if the macro isn't set
	 */
	int get_macro_value(const std::string& name);

	/**
	 * Returns a pointer to the value of a macro given its name.
	 * 
	 * Useful for use with ImGui for example.
	 * 
	 * nullptr is returned if the option doesn't exist (set_macro() wasn't called yet)
	 */
	int* get_pointer_to_macro_value(const std::string& name);

private:
	// Maps the name of the macro to its value. 
	// Example: ["InteriorStackStrategy", 1]
	// 
	// This "options macro" map only contains the macro as defined in KernelOptions.h
	// Those are the macros that control the compilation of the kernels to enable / disable
	// certain behavior of the path tracer by recompilation (to save registers by eliminating code)
	std::unordered_map<std::string, int> m_options_macro_map;

	// This "custom macro" map contains the macros given by the user with set_macro()
	std::unordered_map<std::string, int> m_custom_macro_map;

	// Additional include directories. Does not include the "-I".
	// Example: "../"
	std::vector<std::string> m_additional_include_directories;
};


#endif