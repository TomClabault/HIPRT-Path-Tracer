/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
	static const std::string REUSE_BSDF_MIS_RAY;
	static const std::string DO_FIRST_BOUNCE_WARP_DIRECTION_REUSE;

	static const std::string BSDF_OVERRIDE;
	static const std::string PRINCIPLED_BSDF_DIFFUSE_LOBE;
	static const std::string PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION;
	static const std::string PRINCIPLED_BSDF_ENFORCE_ENERGY_CONSERVATION;
	static const std::string PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION;
	static const std::string PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION;
	static const std::string PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION;
	static const std::string PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION;
	static const std::string PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION;
	static const std::string PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION;
	static const std::string PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL;
	static const std::string PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL;
	static const std::string PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION;
	static const std::string PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION;
	static const std::string PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC;
	static const std::string GGX_SAMPLE_FUNCTION;
	static const std::string NESTED_DIELETRCICS_STACK_SIZE_OPTION;

	static const std::string TRIANGLE_POINT_SAMPLING_STRATEGY;

	static const std::string REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY;
	static const std::string REGIR_GRID_FILL_TARGET_FUNCTION_VISIBILITY;
	static const std::string REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM;
	static const std::string REGIR_SHADING_RESAMPLING_TARGET_FUNCTION_VISIBILITY;
	static const std::string REGIR_SHADING_RESAMPLING_INCLUDE_BSDF;
	static const std::string REGIR_DO_VISIBILITY_REUSE;
	static const std::string REGIR_FALLBACK_LIGHT_SAMPLING_STRATEGY;
	static const std::string REGIR_DO_DISPATCH_COMPACTION;
	static const std::string REGIR_DEBUG_MODE;

	static const std::string DIRECT_LIGHT_SAMPLING_STRATEGY;
	static const std::string DIRECT_LIGHT_SAMPLING_BASE_STRATEGY;
	static const std::string DIRECT_LIGHT_USE_NEE_PLUS_PLUS;
	static const std::string DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE;
	static const std::string DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED;
	static const std::string DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED_BOUNCE;
	static const std::string DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION;
	static const std::string RIS_USE_VISIBILITY_TARGET_FUNCTION;

	static const std::string ENVMAP_SAMPLING_STRATEGY;
	static const std::string ENVMAP_SAMPLING_DO_BSDF_MIS;
	static const std::string ENVMAP_SAMPLING_DO_BILINEAR_FILTERING;

	static const std::string PATH_SAMPLING_STRATEGY;

	static const std::string RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY;
	static const std::string RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY;
	static const std::string RESTIR_DI_DO_VISIBILITY_REUSE;
	static const std::string RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY;
	static const std::string RESTIR_DI_BIAS_CORRECTION_WEIGHTS;
	static const std::string RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY;
	static const std::string RESTIR_DI_DO_LIGHT_PRESAMPLING;
	static const std::string RESTIR_DI_LIGHT_PRESAMPLING_STRATEGY;
	static const std::string RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT;
	static const std::string RESTIR_DI_DO_OPTIMAL_VISIBILITY_SAMPLING;

	static const std::string RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY;
	static const std::string RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT;
	static const std::string RESTIR_GI_DOUBLE_BSDF_TARGET_FUNCTION;
	static const std::string RESTIR_GI_BIAS_CORRECTION_USE_VISIBILITY;
	static const std::string RESTIR_GI_BIAS_CORRECTION_WEIGHTS;
	static const std::string RESTIR_GI_DO_OPTIMAL_VISIBILITY_SAMPLING;
	
	static const std::string GMON_M_SETS_COUNT;

	static const std::unordered_set<std::string> ALL_MACROS_NAMES;

	GPUKernelCompilerOptions();
	GPUKernelCompilerOptions(const GPUKernelCompilerOptions& other);

	/**
	 * Shallow copy of the options of 'other' into 'this'
	 * 
	 * The shared_ptr of the options of 'other' will be shared with 'this': 
	 * this means that if changing the value of "OPTION_1" in 'other', 
	 * the value of "OPTION_1" will also change in 'this'.
	 * 
	 * If this is exactly the behavior that you don't want, have a look at 'deep_copy'
	 */
	GPUKernelCompilerOptions& operator=(const GPUKernelCompilerOptions& other);
	/**
	 * Returns a new GPUKernelCompilerOptions object that has the same option values as 'this'
	 * but with different shared_ptr. This means that if changing the value of "OPTION_1" in 'other', 
	 * the value of "OPTION_1" will not change in the new object returned by this function.
	 */
	GPUKernelCompilerOptions deep_copy() const;

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
};


#endif