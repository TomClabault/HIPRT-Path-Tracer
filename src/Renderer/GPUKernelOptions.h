/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_KERNEL_OPTIONS_H
#define GPU_KERNEL_OPTIONS_H

#include <string>
#include <unordered_map>
#include <vector>

class GPUKernelOptions
{
public:
	static const std::string INTERIOR_STACK_STRATEGY;
	static const std::string DIRECT_LIGHT_SAMPLING_STRATEGY;
	static const std::string RIS_USE_VISIBILITY_TARGET_FUNCTION;

	GPUKernelOptions();

	/**
	 * Gets a list of compiler options of the form { "-D InteriorStackStrategy=1", ... }
	 * that can directly be passed to the kernel compiler
	 */
	std::vector<std::string> get_compiler_options();

	/**
	 * Replace the value of the macro if it has already been added previous to this call
	 */
	void set_option(const std::string& name, int value);

	/** 
	 * Gets the value of an option or -1 if the option isn't set
	 */
	int get_option_value(const std::string& name);

	/**
	 * Returns a pointer to the value of an option given its name.
	 * 
	 * Useful for use with ImGui for example.
	 * 
	 * nullptr is returned if the option doesn't exist (set_option() wasn't called yet)
	 */
	int* get_pointer_to_option_value(const std::string& name);

private:
	std::unordered_map<std::string, int> m_options_map;
};


#endif