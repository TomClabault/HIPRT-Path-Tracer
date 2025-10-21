/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "LightTreeSamplingDataStructure.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/LightTreeSamplingDataStructure.h"
#include "Threads/ThreadManager.h"

void LightTreeSamplingDataStructure::compute_from_scene(const Scene& scene)
{
	compute(
		scene.emissive_triangles_primitive_indices,
		scene.vertices_positions,
		scene.triangles_vertex_indices,
		scene.material_indices,
		scene.materials);
}

void LightTreeSamplingDataStructure::compute(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& triangles_vertex_indices, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE, 
		[this,
		&emissive_triangles_primitive_indices,
		&triangles_vertex_indices,
		&vertices_positions,
		&material_indices,
		&materials] () 
	{
		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

		if (!is_needed(emissive_triangles_primitive_indices.size()))
		{
			free();

			return;
		}

		m_light_tree_builder.build_light_tree(
			emissive_triangles_primitive_indices,
			triangles_vertex_indices,
			vertices_positions,
			material_indices,
			materials);
		m_light_tree_device_data = m_light_tree_builder.compute_device_data<OrochiBuffer>();
		m_light_tree_builder.to_device(m_renderer->get_render_data(), emissive_triangles_primitive_indices, triangles_vertex_indices.size() / 3, m_light_tree_device_data);
		m_light_tree_builder.cleanup();
	});
}

void LightTreeSamplingDataStructure::recompute_if_needed(bool skip_if_already_computed)
{
	if (skip_if_already_computed && m_light_tree_device_data.m_device_nodes_buffer.get_byte_size() > 0)
		// Already computed
		return;

	if (!is_needed(m_renderer->get_render_data().buffers.emissive_triangles_count))
		return;

	m_renderer->synchronize_all_kernels();

	HIPRTScene& hiprt_scene = m_renderer->get_hiprt_scene();

	std::vector<int> emissive_triangle_indices = hiprt_scene.emissive_triangles_primitive_indices.download_data();
	std::vector<float3> vertices_positions = hiprt_scene.whole_scene_BLAS.download_vertices_positions();
	std::vector<int> triangles_indices = hiprt_scene.whole_scene_BLAS.download_triangle_indices();
	std::vector<int> material_indices = hiprt_scene.material_indices.download_data();

	free();
	compute(
		emissive_triangle_indices,
		vertices_positions,
		triangles_indices,
		material_indices,
		m_renderer->get_current_materials());

	ThreadManager::join_threads(ThreadManager::RENDERER_COMPUTE_LIGHT_TREE);
}

void LightTreeSamplingDataStructure::free()
{
	m_light_tree_device_data.free();
}

bool LightTreeSamplingDataStructure::is_needed(unsigned int emissive_count)
{
	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();
	bool directly_using_light_tree = global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) == LSS_BASE_LIGHT_TREE_ATS;
	bool using_regir_light_tree = global_compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) == LSS_BASE_REGIR;

	return (directly_using_light_tree || using_regir_light_tree) && emissive_count > 0;
}

LightTreeBuilderOptions& LightTreeSamplingDataStructure::get_builder_options()
{
	return m_light_tree_builder.get_options();
}
