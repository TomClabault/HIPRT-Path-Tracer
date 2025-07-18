/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef THREAD_FUNCTIONS_H
#define THREAD_FUNCTIONS_H

#include "Renderer/GPURenderer.h"

class ThreadFunctions
{
public:
	static void compile_kernel(std::shared_ptr<GPUKernel> kernel, std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets);
	static void compile_kernel_silent(std::shared_ptr<GPUKernel>, std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets);
	static void precompile_kernel(const std::string& kernel_function_name, const std::string& kernel_filepath, GPUKernelCompilerOptions options, std::shared_ptr<HIPRTOrochiCtx> hiprt_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets);

	static void load_scene_texture(Scene& parsed_scene, std::string scene_path, const std::vector<std::pair<aiTextureType, std::string>>& tex_paths, const std::vector<int>& material_indices, int thread_index, int nb_threads);

	/**
	 * Scans through the emissive meshes of the scene and adds the triangle of those emissive meshes
	 * to the parsed_scene.emissive_triangles_indices field of the scene
	 * 
	 * This function all precomputes the AB and AC edges of the triangles of the scenes for light sampling
	 */
	static void load_scene_parse_emissive_triangles(const aiScene* scene, Scene& parsed_scene);
	/**
	 * Computes the area of each triangle in the scene and stores it in the triangle_areas buffer
	 */
	static void load_scene_compute_triangle_areas(Scene& parsed_scene);

	/**
	 * Reads 'wanted_channel_count' channels of a 32 bit HDR image from 'filepath' and stores it in 'hdr_image_out'.
	 * 
	 * If flip_y is true, the image will be postprocessed such that its origin is in the bottom left corner
	 * (as used by OpenGL or CUDA for example)
	 */
	static void read_envmap(Image32Bit& hdr_image_out, const std::string& filepath, int wanted_channel_count, bool flip_Y);
};

#endif
