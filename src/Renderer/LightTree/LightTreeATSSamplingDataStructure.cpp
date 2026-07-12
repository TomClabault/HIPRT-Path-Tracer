/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/LightTree/LightTreeATSSamplingDataStructure.h"
#include "Threads/ThreadManager.h"

void LightTreeATSSamplingDataStructure::compute_from_scene(const Scene& scene, std::shared_ptr<GPUKernelCompilerOptions> compiler_options)
{
	if (!is_needed(scene.emissive_triangles_primitive_indices.size(), compiler_options))
	{
		free();

		return;
	}

	compute(compiler_options, scene.emissive_triangles_primitive_indices, scene.triangles_average_emissive_power_luminance, scene.vertices_positions,
			scene.triangles_vertex_indices);
}

void LightTreeATSSamplingDataStructure::compute(std::shared_ptr<GPUKernelCompilerOptions> compiler_options,
												const std::vector<int>& emissive_triangles_primitive_indices,
												const std::vector<float>& triangles_average_emissive_power_luminance,
												const std::vector<float3_t>& vertices_positions,
												const std::vector<int>& triangles_vertex_indices)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE_ATS, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE_ATS,
								[this, compiler_options,

								 &emissive_triangles_primitive_indices, &triangles_average_emissive_power_luminance, &triangles_vertex_indices,
								 &vertices_positions]()
								{
									OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

									if (!is_needed(emissive_triangles_primitive_indices.size(), compiler_options))
									{
										free();

										return;
									}

									m_light_tree_builder.build_light_tree(emissive_triangles_primitive_indices, triangles_average_emissive_power_luminance,
																		  triangles_vertex_indices, vertices_positions);
									m_light_tree_ats_device_data = m_light_tree_builder.compute_device_data<OrochiBuffer>();
									m_light_tree_builder.to_device(m_renderer->get_render_data(), emissive_triangles_primitive_indices,
																   triangles_vertex_indices.size() / 3, m_light_tree_ats_device_data);
									m_light_tree_builder.cleanup();
								});
}

void LightTreeATSSamplingDataStructure::recompute_if_needed_or_free(std::shared_ptr<GPUKernelCompilerOptions> compiler_options, bool skip_if_already_computed)
{
	if (skip_if_already_computed && m_light_tree_ats_device_data.m_device_nodes_buffer.get_byte_size() > 0)
		// Already computed
		return;

	if (!is_needed(m_renderer->get_render_data().buffers.emissive_triangles_count, compiler_options))
	{
		free();

		return;
	}

	m_renderer->synchronize_all_kernels();

	HIPRTScene& hiprt_scene = m_renderer->get_hiprt_scene();

	std::vector<int> emissive_triangle_indices					  = hiprt_scene.emissive_triangles_primitive_indices.download_data();
	std::vector<float> triangles_average_emissive_power_luminance = hiprt_scene.triangle_average_emissive_power_luminance.download_data();
	std::vector<float3_t> vertices_positions					  = hiprt_scene.whole_scene_BLAS.download_vertices_positions();
	std::vector<int> triangles_indices							  = hiprt_scene.whole_scene_BLAS.download_triangle_indices();

	free();
	compute(compiler_options, emissive_triangle_indices, triangles_average_emissive_power_luminance, vertices_positions, triangles_indices);

	ThreadManager::join_threads(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE_ATS);
}

void LightTreeATSSamplingDataStructure::free()
{
	m_light_tree_ats_device_data.free();
}

bool LightTreeATSSamplingDataStructure::is_needed(unsigned int emissive_count, std::shared_ptr<GPUKernelCompilerOptions> compiler_options)
{
	bool directly_using_light_tree = compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_LIGHT_TREE_ATS;
	bool using_regir_light_tree	   = compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR;
	bool nee_plus_plus_using_light_tree =
		compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_GRID_PREPOPULATE_LIGHT_SAMPLING_STRATEGY) ==
			LSS_BASE_LIGHT_TREE_ATS &&
		compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE;

	return (directly_using_light_tree || using_regir_light_tree || nee_plus_plus_using_light_tree) && emissive_count > 0;
}

size_t LightTreeATSSamplingDataStructure::get_VRAM_usage_bytes() const
{
	return m_light_tree_ats_device_data.get_VRAM_usage_bytes();
}

LightTreeATSBuilderOptions& LightTreeATSSamplingDataStructure::get_builder_options()
{
	return m_light_tree_builder.get_build_options();
}
