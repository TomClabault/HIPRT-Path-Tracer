#include "Renderer/GPURenderer.h"
#include "Renderer/PowerSamplingDataStructure.h"
#include "Threads/ThreadManager.h"

PowerSamplingDataStructure::PowerSamplingDataStructure(GPURenderer* renderer) : m_renderer(renderer) {}

void PowerSamplingDataStructure::compute_from_scene(const Scene& scene, std::shared_ptr<GPUKernelCompilerOptions> compiler_options)
{
	HIPRTScene& hiprt_scene = m_renderer->get_hiprt_scene();

	compute(compiler_options, scene.emissive_triangles_primitive_indices, scene.triangles_average_emissive_power_luminance,
			m_renderer->get_render_data().buffers.emissive_triangles_power_alias_table);

	// Not joining the thread that does the computation here because it will
	// be joined before starting the render since this method is called during
	// the initialization of the renderer
}

void PowerSamplingDataStructure::recompute_if_needed_or_free(std::shared_ptr<GPUKernelCompilerOptions> compiler_options, bool skip_if_already_computed)
{
	if (skip_if_already_computed && m_alias_table_aliases.get_byte_size() > 0)
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

	compute(compiler_options, emissive_triangle_indices, triangles_average_emissive_power_luminance,
			m_renderer->get_render_data().buffers.emissive_triangles_power_alias_table);

	ThreadManager::join_threads(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE);
}

void PowerSamplingDataStructure::compute(std::shared_ptr<GPUKernelCompilerOptions> compiler_options,
										 const std::vector<int>& emissive_triangle_indices,
										 const std::vector<float>& triangles_average_emissive_power_luminance,
										 AliasTableDevice& power_alias_table)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE,
								[this, compiler_options, &emissive_triangle_indices, &triangles_average_emissive_power_luminance, &power_alias_table]()
								{
									OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

									if (!is_needed(emissive_triangle_indices.size(), compiler_options))
									{
										free();

										return;
									}

									std::vector<float> power_list(emissive_triangle_indices.size());
									float power_sum = 0.0f;

									for (int i = 0; i < emissive_triangle_indices.size(); i++)
									{
										int emissive_triangle_global_index = emissive_triangle_indices[i];

										float power = triangles_average_emissive_power_luminance[emissive_triangle_global_index];

										power_list[i] = power;
										power_sum += power;
									}

									std::vector<float> alias_probas;
									std::vector<int> alias_aliases;
									Utils::compute_alias_table(power_list, power_sum, alias_probas, alias_aliases);

									m_alias_table_probas.resize(emissive_triangle_indices.size());
									m_alias_table_aliases.resize(emissive_triangle_indices.size());

									m_alias_table_probas.upload_data(alias_probas);
									m_alias_table_aliases.upload_data(alias_aliases);

									power_alias_table.alias_table_probas = m_alias_table_probas.get_device_pointer();
									power_alias_table.alias_table_alias	 = m_alias_table_aliases.get_device_pointer();
									power_alias_table.size				 = emissive_triangle_indices.size();
									power_alias_table.sum_elements		 = power_sum;
								});
}

void PowerSamplingDataStructure::free()
{
	if (m_alias_table_probas.size() > 0)
		m_alias_table_probas.free();

	if (m_alias_table_aliases.size() > 0)
		m_alias_table_aliases.free();

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.buffers.emissive_triangles_power_alias_table.alias_table_alias	= nullptr;
	render_data.buffers.emissive_triangles_power_alias_table.alias_table_probas = nullptr;
	render_data.buffers.emissive_triangles_power_alias_table.size				= 0;
	render_data.buffers.emissive_triangles_power_alias_table.sum_elements		= 0;
}

bool PowerSamplingDataStructure::is_needed(unsigned int emissive_count, std::shared_ptr<GPUKernelCompilerOptions> compiler_options)
{
	bool directly_using_power = compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_POWER;
	bool using_regir_power =
		compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR &&
		(compiler_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY_NON_CANONICAL) == LSS_BASE_POWER ||
		 compiler_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY_CANONICAL) == LSS_BASE_POWER);
	bool regir_using_light_distributions_using_power_sampling =
		compiler_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR &&
		compiler_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_USE_PER_CELL_LIGHT_DISTRIBUTIONS) == KERNEL_OPTION_TRUE &&
		compiler_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTIONS_CANONICAL_SAMPLING_TECHNIQUE) == LSS_BASE_POWER;

	return (directly_using_power || using_regir_power || regir_using_light_distributions_using_power_sampling) && emissive_count > 0;
}
