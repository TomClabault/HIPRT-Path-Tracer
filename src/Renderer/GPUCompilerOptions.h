/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_COMPILER_OPTIONS_H
#define GPU_COMPILER_OPTIONS_H

#include <string>
#include <unordered_map>
#include <vector>

class GPUCompilerMacros
{
public:
	/**
	 * Gets a list of compiler options of the form { "-DInteriorStackStrategy=1", ... }
	 * that can directly be passed to the kernel compiler
	 */
	std::vector<std::string> get_compiler_options();

	/**
	 * Replace the value of the macro if it has already been added previous to this call
	 */
	void add_macro(const std::string& name, int value);

private:
	std::unordered_map<std::string, int> m_macros;
};


#endif