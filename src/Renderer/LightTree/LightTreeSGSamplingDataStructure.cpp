#include "LightTreeSGSamplingDataStructure.h"
/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/LightTree/LightTreeSGSamplingDataStructure.h"
#include "Renderer/GPURenderer.h"
#include "Threads/ThreadManager.h"

void LightTreeSGSamplingDataStructure::compute_from_scene(const Scene& scene, std::shared_ptr<GPUKernelCompilerOptions> compiler_options)
{
	compute(compiler_options,

			scene.emissive_triangles_primitive_indices, scene.vertices_positions, scene.triangles_vertex_indices, scene.material_indices, scene.materials);
}

void LightTreeSGSamplingDataStructure::compute(std::shared_ptr<GPUKernelCompilerOptions> compiler_options,
											   const std::vector<int>& emissive_triangles_primitive_indices,
											   const std::vector<float3_t>& vertices_positions,
											   const std::vector<int>& triangles_vertex_indices,
											   const std::vector<int>& material_indices,
											   const std::vector<CPUMaterial>& materials)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE_SG, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE_SG,
								[this, compiler_options,

								 &emissive_triangles_primitive_indices, &triangles_vertex_indices, &vertices_positions, &material_indices, &materials]()
								{
									OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

									if (!is_needed(emissive_triangles_primitive_indices.size(), compiler_options))
									{
										free();

										return;
									}

									m_light_tree_builder_sg.build_light_tree(emissive_triangles_primitive_indices, triangles_vertex_indices, vertices_positions,
																			 material_indices, materials);
									m_light_tree_sg_device_data = m_light_tree_builder_sg.compute_device_data<OrochiBuffer>();
									m_light_tree_builder_sg.to_device(m_renderer->get_render_data(), emissive_triangles_primitive_indices,
																	  triangles_vertex_indices.size() / 3, m_light_tree_sg_device_data);
									m_light_tree_builder_sg.cleanup();
								});
}

void LightTreeSGSamplingDataStructure::recompute_if_needed_or_free(std::shared_ptr<GPUKernelCompilerOptions> compiler_options, bool skip_if_already_computed)
{
	if (skip_if_already_computed && m_light_tree_sg_device_data.m_device_nodes_buffer.get_byte_size() > 0)
		// Already computed
		return;

	if (!is_needed(m_renderer->get_render_data().buffers.emissive_triangles_count, compiler_options))
		return;

	m_renderer->synchronize_all_kernels();

	HIPRTScene& hiprt_scene = m_renderer->get_hiprt_scene();

	std::vector<int> emissive_triangle_indices = hiprt_scene.emissive_triangles_primitive_indices.download_data();
	std::vector<float3_t> vertices_positions   = hiprt_scene.whole_scene_BLAS.download_vertices_positions();
	std::vector<int> triangles_indices		   = hiprt_scene.whole_scene_BLAS.download_triangle_indices();
	std::vector<int> material_indices		   = hiprt_scene.material_indices.download_data();

	free();
	compute(compiler_options,

			emissive_triangle_indices, vertices_positions, triangles_indices, material_indices, m_renderer->get_current_materials());

	ThreadManager::join_threads(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE_SG);
}

void LightTreeSGSamplingDataStructure::free()
{
	m_light_tree_sg_device_data.free();
}

bool LightTreeSGSamplingDataStructure::is_needed(unsigned int emissive_count, std::shared_ptr<GPUKernelCompilerOptions> compiler_options)
{
	bool directly_using_light_tree = compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_LIGHT_TREE_SG;
	bool using_regir_light_tree	   = compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR;
	bool nee_plus_plus_using_light_tree =
							compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_GRID_PREPOPULATE_LIGHT_SAMPLING_STRATEGY) ==
													LSS_BASE_LIGHT_TREE_SG &&
							compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE;

	return (directly_using_light_tree || using_regir_light_tree || nee_plus_plus_using_light_tree) && emissive_count > 0;
}

size_t LightTreeSGSamplingDataStructure::get_VRAM_usage_bytes() const
{
	return m_light_tree_sg_device_data.get_VRAM_usage_bytes();
}

LightTreeATSBuilderOptions& LightTreeSGSamplingDataStructure::get_builder_options()
{
	return m_light_tree_builder_sg.get_build_options();
}
