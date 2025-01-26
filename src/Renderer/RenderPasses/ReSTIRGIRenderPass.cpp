/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReSTIRGIRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

const std::string ReSTIRGIRenderPass::RESTIR_GI_RENDER_PASS_NAME = "ReSTIR GI Render Pass";
const std::string ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID = "ReSTIR GI Spatial Reuse";

const std::unordered_map<std::string, std::string> ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ RESTIR_GI_SPATIAL_REUSE_KERNEL_ID, "ReSTIR_GI_SpatialReuse" },
};

const std::unordered_map<std::string, std::string> ReSTIRGIRenderPass::KERNEL_FILES =
{
	{ RESTIR_GI_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/GI/SpatialReuse.h" },

};

ReSTIRGIRenderPass::ReSTIRGIRenderPass() : ReSTIRGIRenderPass(nullptr) {}
ReSTIRGIRenderPass::ReSTIRGIRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReSTIRGIRenderPass::RESTIR_GI_RENDER_PASS_NAME) 
{
	OROCHI_CHECK_ERROR(oroEventCreate(&m_spatial_reuse_time_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_spatial_reuse_time_stop));

	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReSTIRGIRenderPass::KERNEL_FILES.at(ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
}

void ReSTIRGIRenderPass::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	if (!is_render_pass_used())
		return;

	ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel, m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID], hiprt_orochi_ctx, std::ref(func_name_sets));
}

void ReSTIRGIRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used())
		return;

	m_initial_candidates_buffer.resize(new_width * new_height);
}

bool ReSTIRGIRenderPass::pre_render_update(float delta_time)
{
	bool render_data_invalidated = false;

	int2 render_resolution = m_renderer->m_render_resolution;

	if (is_render_pass_used())
	{
		// ReSTIR GI enabled
		bool initial_candidates_reservoir_needs_resize = m_initial_candidates_buffer.get_element_count() == 0;

		if (initial_candidates_reservoir_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			render_data_invalidated = true;

		if (initial_candidates_reservoir_needs_resize)
			m_initial_candidates_buffer.resize(render_resolution.x * render_resolution.y);
	}
	else
	{
		// ReSTIR DI disabled, we're going to free the buffers if that's not already done
		if (m_initial_candidates_buffer.get_element_count() > 0)
		{
			m_initial_candidates_buffer.free();

			render_data_invalidated = true;
		}
	}

	return render_data_invalidated;
}

bool ReSTIRGIRenderPass::launch()
{
	return false;
}

void ReSTIRGIRenderPass::post_render_update()
{
}

void ReSTIRGIRenderPass::update_render_data()
{
	if (!is_render_pass_used)
		return;

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer = m_initial_candidates_buffer.get_device_pointer();
}

void ReSTIRGIRenderPass::reset()
{
}

void ReSTIRGIRenderPass::compute_render_times()
{
	if (!is_render_pass_used())
		return;

	std::unordered_map<std::string, float>& render_pass_times = m_renderer->get_render_pass_times();
	render_pass_times[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID] = m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->get_last_execution_time();
}

std::map<std::string, std::shared_ptr<GPUKernel>> ReSTIRGIRenderPass::get_tracing_kernels()
{
	return get_all_kernels();
}

bool ReSTIRGIRenderPass::is_render_pass_used() const
{
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PSS_RESTIR_GI;
}
