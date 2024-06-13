/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPUCompilerOptions.h"

std::vector<std::string> GPUCompilerMacros::get_compiler_options()
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_macros)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(macro_key_value.second));

	return macros;
}

void GPUCompilerMacros::add_macro(const std::string& name, int value)
{
	m_macros[name] = value;
}
