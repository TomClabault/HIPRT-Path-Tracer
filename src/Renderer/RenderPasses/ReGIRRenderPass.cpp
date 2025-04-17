/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

const std::string ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID = "ReGIR Grid fill & temporal reuse";
const std::string ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID = "ReGIR Spatial reuse";

const std::string ReGIRRenderPass::REGIR_RENDER_PASS_NAME = "ReGIR Render Pass";

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, "ReGIR_Grid_Fill_Temporal_Reuse" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, "ReGIR_Spatial_Reuse" },
};

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FILES =
{
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/ReGIRGridFillTemporalReuse.h" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/ReGIRSpatialReuse.h" },
};

ReGIRRenderPass::ReGIRRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReGIRRenderPass::REGIR_RENDER_PASS_NAME)
{
	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);

	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
}

bool ReGIRRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool grid_fill_compiled = m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	if (!grid_fill_compiled)
		// Spatiotemporal is needed but hasn't been compiled yet
		m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool spatial_reuse_compiled = m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled();
	if (!spatial_reuse_compiled)
		// Spatiotemporal is needed but hasn't been compiled yet
		m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return !grid_fill_compiled || !spatial_reuse_compiled;
}

bool ReGIRRenderPass::pre_render_update(float delta_time)
{
	bool updated = false;

	if (is_render_pass_used())
	{
		// Resizing the grid if it is not the right size
		if (m_grid_buffers.get_element_count() != m_render_data->render_settings.regir_settings.get_total_number_of_reservoirs_ReGIR())
		{
			m_grid_buffers.resize(m_render_data->render_settings.regir_settings.get_total_number_of_reservoirs_ReGIR());

			updated = true;
		}

		if (m_render_data->render_settings.regir_settings.spatial_reuse.do_spatial_reuse && m_spatial_reuse_output_grid_buffer.get_element_count() != m_render_data->render_settings.regir_settings.get_number_of_reservoirs_per_grid())
		{
			m_spatial_reuse_output_grid_buffer.resize(m_render_data->render_settings.regir_settings.get_number_of_reservoirs_per_grid());

			updated = true;
		}
	}
	else
	{
		if (m_grid_buffers.get_element_count() > 0)
		{
			m_grid_buffers.free();

			updated = true;
		}

		if (m_spatial_reuse_output_grid_buffer.get_element_count() > 0)
		{
			m_spatial_reuse_output_grid_buffer.free();

			updated = true;
		}
	}

	return updated;
}

bool ReGIRRenderPass::launch()
{
	if (!is_render_pass_used())
		return false;

	launch_grid_fill_temporal_reuse();
	launch_spatial_reuse();

	return true;
}

void ReGIRRenderPass::launch_grid_fill_temporal_reuse()
{
	void* launch_args[] = { m_render_data };

	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(64, 1, m_render_data->render_settings.regir_settings.get_number_of_reservoirs_per_grid(), 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::launch_spatial_reuse()
{
	if (!m_render_data->render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
		return;

	void* launch_args[] = { m_render_data };

	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(64, 1, m_render_data->render_settings.regir_settings.get_number_of_reservoirs_per_grid(), 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::post_render_update()
{
	if (m_render_data->render_settings.regir_settings.temporal_reuse.do_temporal_reuse)
	{
		m_render_data->render_settings.regir_settings.temporal_reuse.current_grid_index++;
		m_render_data->render_settings.regir_settings.temporal_reuse.current_grid_index %= m_render_data->render_settings.regir_settings.temporal_reuse.temporal_history_length;
	}
}

void ReGIRRenderPass::update_render_data()
{
	if (is_render_pass_used())
	{
		m_render_data->render_settings.regir_settings.grid_fill.grid_buffers = m_grid_buffers.get_device_pointer();
		m_render_data->render_settings.regir_settings.grid.grid_origin = m_renderer->get_scene_metadata().scene_bounding_box.mini;
		m_render_data->render_settings.regir_settings.grid.extents = m_renderer->get_scene_metadata().scene_bounding_box.get_extents();

		m_render_data->render_settings.regir_settings.spatial_reuse.output_grid = m_spatial_reuse_output_grid_buffer.get_device_pointer();
	}
	else
	{
		m_render_data->render_settings.regir_settings.grid_fill.grid_buffers = nullptr;
	}
}

void ReGIRRenderPass::reset()
{
	m_render_data->render_settings.regir_settings.temporal_reuse.current_grid_index = 0;
}

bool ReGIRRenderPass::is_render_pass_used() const
{
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) == LSS_BASE_REGIR;
}

float ReGIRRenderPass::get_VRAM_usage() const
{
	return (m_grid_buffers.get_byte_size() + m_spatial_reuse_output_grid_buffer.get_byte_size()) / 1000000.0f;
}
