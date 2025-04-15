/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

const std::string ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID = "ReGIR Grid fill";

const std::string ReGIRRenderPass::REGIR_RENDER_PASS_NAME = "ReGIR Render Pass";

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ REGIR_GRID_FILL_KERNEL_ID, "ReGIR_Grid_Fill" },
};

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FILES =
{
	{ REGIR_GRID_FILL_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/ReGIRGridFill.h" },
};

ReGIRRenderPass::ReGIRRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReGIRRenderPass::REGIR_RENDER_PASS_NAME)
{
	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);
}

bool ReGIRRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (is_render_pass_used())
	{
		bool compiled = m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->has_been_compiled();
		if (!compiled)
		{
			// Spatiotemporal is needed but hasn't been compiled yet
			m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

			return true;
		}
	}

	return false;
}

bool ReGIRRenderPass::pre_render_update(float delta_time)
{
	if (is_render_pass_used())
	{
		// Resizing the grid if it is not the right size
		if (m_regir_data.grid_buffer.get_element_count() != m_regir_data.get_number_of_reservoirs_in_grid(*m_render_data))
		{
			m_regir_data.grid_buffer.resize(m_regir_data.get_number_of_reservoirs_in_grid(*m_render_data));

			return true;
		}
	}
	else
	{
		if (m_regir_data.grid_buffer.get_element_count() > 0)
		{
			m_regir_data.grid_buffer.free();

			return true;
		}
	}

	return false;
}

bool ReGIRRenderPass::launch()
{
	if (!is_render_pass_used())
		return false;

	void* launch_args[] = { m_render_data };

	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_KERNEL_ID]->launch_asynchronous(64, 1, m_regir_data.get_number_of_reservoirs_in_grid(*m_render_data), 1, launch_args, m_renderer->get_main_stream());

	return true;
}

void ReGIRRenderPass::update_render_data()
{
	if (is_render_pass_used())
	{
		m_render_data->render_settings.regir_settings.grid_buffer = m_regir_data.grid_buffer.get_device_pointer();
		m_render_data->render_settings.regir_settings.grid_origin = m_renderer->get_scene_metadata().scene_bounding_box.mini;
		m_render_data->render_settings.regir_settings.extents = m_renderer->get_scene_metadata().scene_bounding_box.get_extents();
	}
	else
	{
		m_render_data->render_settings.regir_settings.grid_buffer = nullptr;
	}
}

void ReGIRRenderPass::reset()
{
}

bool ReGIRRenderPass::is_render_pass_used() const
{
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) == LSS_BASE_REGIR;
}

float ReGIRRenderPass::get_VRAM_usage() const
{
	return m_regir_data.grid_buffer.get_byte_size() / 1000000.0f;
}

ReGIRGPUData& ReGIRRenderPass::get_ReGIR_data()
{
	return m_regir_data;
}
