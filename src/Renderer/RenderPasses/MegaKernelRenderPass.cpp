/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"

const std::string MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME = "Megakernel Render Pass";
const std::string MegaKernelRenderPass::MEGAKERNEL_KERNEL = "Megakernel (1 SPP)";

MegaKernelRenderPass::MegaKernelRenderPass() : MegaKernelRenderPass(nullptr) {}
MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer) : MegaKernelRenderPass(renderer, MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME) {}
MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer, const std::string& name) : RenderPass(renderer, name) 
{
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL] = std::make_shared<GPUKernel>();
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Megakernel.h");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_function_name("MegaKernel");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->synchronize_options_with(m_renderer->get_global_compiler_options(), GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
}

void MegaKernelRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool MegaKernelRenderPass::pre_render_update(float delta_time)
{
	if (!is_render_pass_used())
		return false;

	// Resetting this flag as this is a new frame
	m_render_data->render_settings.do_update_status_buffers = false;

	if (!m_render_data->render_settings.accumulate)
		m_render_data->render_settings.sample_number = 0;

	return false;
}

bool MegaKernelRenderPass::launch()
{
	if (!is_render_pass_used())
		return false;

	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();

	void* launch_args[] = { m_render_data };

	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_renderer->get_main_stream());

	return true;
}

void MegaKernelRenderPass::post_render_update()
{
	if (!is_render_pass_used())
		return;
}

void MegaKernelRenderPass::reset()
{
	if (!is_render_pass_used())
		return;

	if (m_render_data->render_settings.accumulate)
		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			m_render_data->render_settings.samples_per_frame = 1;

	m_render_data->render_settings.denoiser_AOV_accumulation_counter = 0;

	m_render_data->render_settings.sample_number = 0;
}

bool MegaKernelRenderPass::is_render_pass_used() const
{
	// Only active if we're not using ReSTIR GI because if we are using ReSTIR, the path tracing is done in
	// the initial candidates kernel
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) != PSS_RESTIR_GI;
}
