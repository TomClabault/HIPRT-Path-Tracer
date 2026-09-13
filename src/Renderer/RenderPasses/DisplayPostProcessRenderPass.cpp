/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/DisplayPostProcessRenderPass.h"
#include "UI/DisplayView/DisplayViewSystem.h"
#include "UI/RenderWindow.h"

const std::string DisplayPostProcessRenderPass::RENDER_PASS_NAME			= "Display Post Process Render Pass";
const std::string DisplayPostProcessRenderPass::DISPLAY_POST_PROCESS_KERNEL = "Display Post Process";

DisplayPostProcessRenderPass::DisplayPostProcessRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(DisplayPostProcessRenderPass::RENDER_PASS_NAME, renderer, options)
{
	m_render_data_host_pinned.resize_host_pinned_mem(1);

	std::shared_ptr<GPUKernel> kernel = std::make_shared<GPUKernel>(this->get_name() + "::" + DISPLAY_POST_PROCESS_KERNEL);
	kernel->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/DisplayPostProcess.h");
	kernel->set_kernel_function_name("DisplayPostProcess");
	kernel->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);

	m_kernels[DISPLAY_POST_PROCESS_KERNEL] = kernel;
}

void DisplayPostProcessRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool DisplayPostProcessRenderPass::pre_frame_render_update(float delta_time)
{
	DisplaySettings& display_settings				  = m_render_window->get_display_view_system()->get_display_settings();
	DisplayPostProcessSettings& post_process_settings = m_renderer->get_render_data().display_post_process_settings;

	post_process_settings.do_tonemapping = display_settings.do_tonemapping;
	post_process_settings.gamma			 = display_settings.tone_mapping_gamma;
	post_process_settings.exposure		 = display_settings.tone_mapping_exposure;

	return false;
}

bool DisplayPostProcessRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;

	std::shared_ptr<GPUKernel> kernel = m_kernels[DISPLAY_POST_PROCESS_KERNEL];
	kernel->upload_to_module_global("DISPLAY_POST_PROCESS_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData), m_renderer->get_main_stream());

	kernel->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, nullptr,
								m_renderer->get_main_stream());

	return true;
}
