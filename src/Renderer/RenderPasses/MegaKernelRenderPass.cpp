/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"

const std::string MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME = "Megakernel Render Pass";
const std::string MegaKernelRenderPass::MEGAKERNEL_KERNEL = "Megakernel (1 SPP)";

MegaKernelRenderPass::MegaKernelRenderPass() : MegaKernelRenderPass(nullptr) {}
MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer) : RenderPass(renderer, MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME) {}

void MegaKernelRenderPass::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL] = std::make_shared<GPUKernel>();
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/FullPathTracer.h");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_function_name("FullPathTracer");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->synchronize_options_with(m_renderer->get_global_compiler_options(), GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel, m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL], hiprt_orochi_ctx, std::ref(func_name_sets));
}

void MegaKernelRenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, true);
}

void MegaKernelRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool MegaKernelRenderPass::pre_render_update(float delta_time)
{
	// Resetting this flag as this is a new frame
	m_render_data->render_settings.do_update_status_buffers = false;

	if (!m_render_data->render_settings.accumulate)
		m_render_data->render_settings.sample_number = 0;

	return false;
}

bool MegaKernelRenderPass::launch()
{
	m_render_data->random_seed = m_renderer->rng().xorshift32();

	void* launch_args[] = { m_render_data };

	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_renderer->get_main_stream());

	return true;
}

void MegaKernelRenderPass::post_render_update()
{
	m_render_data->render_settings.sample_number++;
	m_render_data->render_settings.denoiser_AOV_accumulation_counter++;

	// We only reset once so after rendering a frame, we're sure that we don't need to reset anymore 
	// so we're setting the flag to false (it will be set to true again if we need to reset the render
	// again)
	m_render_data->render_settings.need_to_reset = false;
	m_render_data->nee_plus_plus.reset_visibility_map = false;
	// If we had requested a temporal buffers clear, this has be done by this frame so we can
	// now reset the flag
	m_render_data->render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested = false;

	// Saving the current frame camera to be the previous camera of the next frame
	m_renderer->get_previous_frame_camera()  = m_renderer->get_camera();
}

void MegaKernelRenderPass::reset()
{
	if (m_render_data->render_settings.accumulate)
	{
		// Only resetting the seed for deterministic rendering if we're accumulating.
		// If we're not accumulating, we want each frame of the render to be different
		// so we don't get into that if block and we don't reset the seed
		m_renderer->rng().m_state.seed = 42;

		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			m_render_data->render_settings.samples_per_frame = 1;
	}

	m_render_data->render_settings.denoiser_AOV_accumulation_counter = 0;
	m_render_data->render_settings.sample_number = 0;
	m_render_data->render_settings.need_to_reset = true;
}

void MegaKernelRenderPass::compute_render_times()
{
	std::unordered_map<std::string, float>& render_times = m_renderer->get_render_pass_times();

	render_times[MegaKernelRenderPass::MEGAKERNEL_KERNEL] = m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_last_execution_time();
}

void MegaKernelRenderPass::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	perf_metrics->add_value(MegaKernelRenderPass::MEGAKERNEL_KERNEL, m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_last_execution_time());
}

std::map<std::string, std::shared_ptr<GPUKernel>> MegaKernelRenderPass::get_all_kernels()
{
	return m_kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> MegaKernelRenderPass::get_tracing_kernels()
{
	return m_kernels;
}
