/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_OPTIONS_H
#define GPU_KERNEL_OPTIONS_H

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class GPUKernel;

class GPUKernelCompilerOptions
{
public:
	static const std::string USE_SHARED_STACK_BVH_TRAVERSAL;
	static const std::string SHARED_STACK_BVH_TRAVERSAL_BLOCK_SIZE;
	static const std::string SHARED_STACK_BVH_TRAVERSAL_SIZE;

	static const std::string BSDF_OVERRIDE;
	static const std::string PRINCIPLED_BSDF_DIFFUSE_LOBE;
	static const std::string PRINCIPLED_BSDF_ENFORCE_ENERGY_CONSERVATION;
	static const std::string PRINCIPLED_BSDF_GGX_MULTIPLE_SCATTERING;
	static const std::string PRINCIPLED_BSDF_GGX_MULTIPLE_SCATTERING_DO_FRESNEL;
	static const std::string GGX_SAMPLE_FUNCTION;
	static const std::string INTERIOR_STACK_STRATEGY;
	static const std::string NESTED_DIELETRCICS_STACK_SIZE_OPTION;

	static const std::string DIRECT_LIGHT_SAMPLING_STRATEGY;
	static const std::string RIS_USE_VISIBILITY_TARGET_FUNCTION;
	static const std::string ENVMAP_SAMPLING_STRATEGY;
	static const std::string ENVMAP_SAMPLING_DO_BSDF_MIS;

	static const std::string RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY;
	static const std::string RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY;
	static const std::string RESTIR_DI_DO_VISIBILITY_REUSE;
	static const std::string RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY;
	static const std::string RESTIR_DI_BIAS_CORRECTION_WEIGHTS;
	static const std::string RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY;
	static const std::string RESTIR_DI_DO_LIGHTS_PRESAMPLING;

	static const std::unordered_set<std::string> ALL_MACROS_NAMES;

	GPUKernelCompilerOptions();
	GPUKernelCompilerOptions(const GPUKernelCompilerOptions& other);

	GPUKernelCompilerOptions operator=(const GPUKernelCompilerOptions& other);

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
	 * a call to 'set_macro_value()' (unless the macro changed through 'set_macro_value()' is an
	 * option macro defined in KernelOptions.h as defined above)
	 * 
	 * The returned vector also contains all the additional compiler macro that were added
	 * to the kernel by calling 'kernel.add_additional_macro_for_compilation()'
	 */
	std::vector<std::string> get_relevant_macros_as_std_vector_string(const GPUKernel* kernel) const;

	/**
	 * Replace the value of the macro if it has already been added previous to this call.
	 * If the macro doesn't exist in these compiler options, it it added to the custom
	 * options map.
	 * 
	 * The 'name' parameter is expected to be given without the '-D' macro prefix commonly 
	 * given to compilers.
	 * For example, if you want to define a macro "MyMacro" equal to 1, you simply
	 * call set_macro_value("MyMacro", 1).
	 * The addition of the -D prefix will be added internally.
	 */
	void set_macro_value(const std::string& name, int value);

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
	int get_macro_value(const std::string& name) const;

	/**
	 * Returns a pointer to the value of a macro given its name.
	 * 
	 * Useful for use with ImGui for example.
	 * 
	 * nullptr is returned if the option doesn't exist (set_macro_value() wasn't called yet)
	 */
	const std::shared_ptr<int> get_pointer_to_macro_value(const std::string& name) const;
	int* get_raw_pointer_to_macro_value(const std::string& name);

	/**
	 * Links the value of the macro 'name' with the given pointer such that if the value at the given
	 * 'pointer_to_value' is modified, the value of the same macro in this instance of GPUKernelCompilerOptions
	 * will also be modified to the same value
	 */
	void set_pointer_to_macro(const std::string& name, std::shared_ptr<int> pointer_to_value);

	/**
	 * Returns the map that stores the macro names with their associated values
	 */
	const std::unordered_map<std::string, std::shared_ptr<int>>& get_options_macro_map() const;

	/**
	 * Returns the map that stores the custom macro names with their associated values
	 */
	const std::unordered_map<std::string, std::shared_ptr<int>>& get_custom_macro_map() const;

	/**
	 * Removes all options from this instance
	 */
	void clear();

	/**
	 * Overrides any option value of 'other' with the value of the corresponding option of this instance
	 * If the option doesn't exist in other, it is added
	 */
	void apply_onto(GPUKernelCompilerOptions& other);

private:
	// Maps the name of the macro to its value. 
	// Example: ["InteriorStackStrategy", 1]
	// 
	// This "options macro" map only contains the macro as defined in KernelOptions.h
	// Those are the macros that control the compilation of the kernels to enable / disable
	// certain behavior of the path tracer by recompilation (to save registers by eliminating code)
	//
	// This macro map and the 'custom_macro_map' contain pointers to int for their values
	// because we want to be able to synchronize the value of the options with
	// another instance of GPUKernelCompilerOptions. This requires having the value
	// of our macro point to the value of the other GPUKernelCompilerOptions instance
	// and we need pointers for that
	std::unordered_map<std::string, std::shared_ptr<int>> m_options_macro_map;

	// This "custom macro" map contains the macros given by the user with set_macro_value().
	// Any macro that isn't defined in KernelOptions.h will be found in this custom macro map
	std::unordered_map<std::string, std::shared_ptr<int>> m_custom_macro_map;

	// Additional include directories. Does not include the "-I".
	// Example: "../"
	std::vector<std::string> m_additional_include_directories;
};


#endif