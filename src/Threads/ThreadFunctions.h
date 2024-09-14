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
	static void compile_kernel(GPUKernel& kernel, std::shared_ptr<GPUKernelCompilerOptions> kernel_compiler_options, HIPRTOrochiCtx& hiprt_ctx);

	static void load_scene_texture(Scene& parsed_scene, std::string scene_path, const std::vector<std::pair<aiTextureType, std::string>>& tex_paths, const std::vector<int>& material_indices, int thread_index, int nb_threads);

	/**
	 * Reads 'wanted_channel_count' channels of a 32 bit HDR image from 'filepath' and stores it in 'hdr_image_out'.
	 * If flip_y is true, the image will be postprocessed such that its origin is in the bottom left corner
	 * (as used by OpenGL or CUDA for example)
	 */
	static void read_image_hdr(Image32Bit& hdr_image_out, const std::string& filepath, int wanted_channel_count, bool flip_Y);
};

#endif
