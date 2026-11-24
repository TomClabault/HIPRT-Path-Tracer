/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef COMMANDLINE_ARGUMENTS_H
#define COMMANDLINE_ARGUMENTS_H

#include <iostream>
#include <vector>

struct CommandlineArguments
{
	static const std::string DEFAULT_SCENE;
	static const std::string DEFAULT_SKYSPHERE;

	static CommandlineArguments process_command_line_args(int argc, char** argv);
	// Parses a comma separated list of sample counts, "2,4,16" for example
	static std::vector<int> parse_sample_checkpoints(const std::string& comma_separated_sample_counts);

	int render_width = 1280, render_height = 720;

	// Default scene and skysphere paths as expected if running the application from a build
	// directory inside the repo root folder
	std::string scene_file_path		= DEFAULT_SCENE;
	std::string skysphere_file_path = "";

	int render_samples = 64;
	int bounces		   = 8;

	// When specified, render on the GPU until render_samples is reached, write this PNG, and exit.
	std::string output_filepath = "";
	// With 'output_filepath': sample counts below 'render_samples' at which the render is also written, to '<output stem>_<count>spp<extension>',
	// in the same process
	std::vector<int> sample_checkpoints;

	bool only_compile_shaders = false;
	bool no_shader_cache	  = false;
};

#endif // #ifndef COMMANDLINE_ARGUMENTS_H
