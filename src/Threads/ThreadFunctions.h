/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef THREAD_FUNCTIONS_H
#define THREAD_FUNCTIONS_H

#include "Renderer/GPURenderer.h"

class ThreadFunctions
{
public:
	static void compile_kernel(std::shared_ptr<GPURenderer> renderer, std::string kernel_file, std::string kernel_function);

	static void load_texture(Scene& parsed_scene, std::string scene_path, const std::vector<std::pair<aiTextureType, std::string>>& tex_paths, int thread_index, int nb_threads);
};

#endif
