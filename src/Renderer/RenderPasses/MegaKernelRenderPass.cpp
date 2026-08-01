/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

const std::string MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME = "Megakernel Render Pass";
const std::string MegaKernelRenderPass::MEGAKERNEL_KERNEL			= "Megakernel (1 SPP)";

MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: MegaKernelRenderPass(MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME, renderer, options)
{
}
MegaKernelRenderPass::MegaKernelRenderPass(const std::string& name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(name, renderer, options)
{
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + MegaKernelRenderPass::MEGAKERNEL_KERNEL);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Megakernel.h");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_function_name("MegaKernel");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
																							 KERNEL_OPTION_TRUE);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
}

bool MegaKernelRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
														const std::vector<hiprtFuncNameSet>& func_name_sets,
														bool silent,
														bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool updated = false;

	if (!m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->has_been_compiled())
	{
		updated = true;
		m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	return updated;
}

void MegaKernelRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool MegaKernelRenderPass::pre_sample_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
		return false;

	// Resetting this flag as this is a new frame
	render_data.render_settings.do_update_status_buffers = false;

	if (!render_data.render_settings.accumulate)
		render_data.render_settings.sample_number = 0;

	return false;
}

bool MegaKernelRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	void* launch_args[] = { &render_data };

	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x,
																			m_render_resolution.y, launch_args, m_renderer->get_main_stream());
	OROCHI_CHECK_ERROR(oroStreamSynchronize(m_renderer->get_main_stream()));

	return true;
}

void MegaKernelRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
		return;

	if (render_data.render_settings.accumulate)
		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			render_data.render_settings.samples_per_frame = 1;

	render_data.render_settings.denoiser_AOV_accumulation_counter = 0;

	render_data.render_settings.sample_number = 0;
}

bool MegaKernelRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	// Only active if we're not using ReSTIR GI/PT because if we are using ReSTIR, the path tracing is done in
	// the initial candidates kernel
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) != PATH_SAMPLING_RESTIR_GI &&
		   compiler_options.get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) != PATH_SAMPLING_RESTIR_PT;
}
