/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/DisplayPostProcessRenderPass.h"
#include "UI/DisplayView/DisplayViewSystem.h"
#include "UI/RenderWindow.h"

#include <algorithm>

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
	update_display_post_process_settings();

	return false;
}

bool DisplayPostProcessRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	// The display post-process output is only needed once per frame, after the final path-tracing sample.
	if (!render_data.render_settings.do_update_status_buffers)
		return false;

	// launch_async() runs before the render thread increments sample_number for the sample just accumulated.
	render_data.display_post_process_settings.denoised_blend_noisy_sample_count = render_data.render_settings.sample_number + 1;

	return launch_kernel(render_data);
}

void DisplayPostProcessRenderPass::update_display_post_process_settings()
{
	DisplaySettings& display_settings				  = m_render_window->get_display_view_system()->get_display_settings();
	DisplayPostProcessSettings& post_process_settings = m_renderer->get_render_data().display_post_process_settings;

	post_process_settings.do_tonemapping				   = display_settings.do_tonemapping;
	post_process_settings.gamma							   = display_settings.tone_mapping_gamma;
	post_process_settings.exposure						   = display_settings.tone_mapping_exposure;
	post_process_settings.white_furnace_use_low_threshold  = display_settings.white_furnace_display_use_low_threshold;
	post_process_settings.white_furnace_use_high_threshold = display_settings.white_furnace_display_use_high_threshold;

	std::shared_ptr<ApplicationSettings> application_settings = m_renderer->get_application_settings();
	post_process_settings.denoised_blend_factor = display_settings.blend_override != -1.0f ? display_settings.blend_override : display_settings.denoiser_blend;
	post_process_settings.denoised_blend_noisy_sample_count = std::max(1, static_cast<int>(m_renderer->get_render_settings().sample_number));
	post_process_settings.denoised_blend_sample_count		= std::max(1, application_settings->last_denoised_sample_count);

	DisplayViewType display_view_type = m_render_window->get_display_view_system()->get_current_display_view_type();
	switch (display_view_type)
	{
	case DisplayViewType::DISPLAY_DENOISER_ALBEDO:
		post_process_settings.display_view = DISPLAY_POST_PROCESS_DENOISER_ALBEDO;
		break;
	case DisplayViewType::DISPLAY_DENOISER_NORMALS:
		post_process_settings.display_view = DISPLAY_POST_PROCESS_DENOISER_NORMALS;
		break;
	case DisplayViewType::WHITE_FURNACE_THRESHOLD:
		post_process_settings.display_view = DISPLAY_POST_PROCESS_WHITE_FURNACE_THRESHOLD;
		break;
	case DisplayViewType::DENOISED_BLEND:
		post_process_settings.display_view = DISPLAY_POST_PROCESS_DENOISED_BLEND;
		break;
	case DisplayViewType::DEFAULT:
	default:
		post_process_settings.display_view = DISPLAY_POST_PROCESS_DEFAULT;
		break;
	}

	int white_furnace_sample_count = static_cast<int>(m_renderer->get_render_settings().sample_number);
	if (application_settings->enable_denoising && application_settings->last_denoised_sample_count != -1)
		white_furnace_sample_count = application_settings->last_denoised_sample_count;
	post_process_settings.white_furnace_sample_count = std::max(1, white_furnace_sample_count);
}

bool DisplayPostProcessRenderPass::launch_display_only(HIPRTRenderData& render_data)
{
	update_display_post_process_settings();
	render_data.display_post_process_settings = m_renderer->get_render_data().display_post_process_settings;
	render_data.display_post_process_settings.denoised_blend_noisy_sample_count =
		std::max(1, static_cast<int>(m_renderer->get_render_settings().sample_number));
	return launch_kernel(render_data);
}

bool DisplayPostProcessRenderPass::launch_kernel(HIPRTRenderData& render_data)
{
	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;

	std::shared_ptr<GPUKernel> kernel = m_kernels[DISPLAY_POST_PROCESS_KERNEL];
	kernel->upload_to_module_global("DISPLAY_POST_PROCESS_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData), m_renderer->get_main_stream());

	kernel->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, nullptr,
								m_renderer->get_main_stream());

	return true;
}
