/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GMoNRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

GMoNRenderPass::GMoNRenderPass()
{
	m_compute_gmon_kernel.set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/GMoN/GMoNComputeMedianOfMeans.h");
	m_compute_gmon_kernel.set_kernel_function_name("GMoNComputeMedianOfMeans");
}

GMoNRenderPass::GMoNRenderPass(GPURenderer* renderer) : GMoNRenderPass() 
{
	m_renderer = renderer;

	m_compute_gmon_kernel.synchronize_options_with(*renderer->get_global_compiler_options(), {});
}

void GMoNRenderPass::compile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx)
{
	if (!use_gmon())
		return;

	ThreadManager::start_thread(ThreadManager::COMPILE_KERNELS_THREAD_KEY, ThreadFunctions::compile_kernel_no_func_sets, std::ref(m_compute_gmon_kernel), hiprt_orochi_ctx);
}

void GMoNRenderPass::recompile(std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, bool silent, bool use_cache)
{
	if (!use_gmon())
		return;

	if (silent)
		m_compute_gmon_kernel.compile_silent(hiprt_orochi_ctx, {}, use_cache);
	else
		m_compute_gmon_kernel.compile(hiprt_orochi_ctx, {}, use_cache);
}

void GMoNRenderPass::launch()
{
	if (!use_gmon())
		return;

	unsigned int number_of_sets = m_compute_gmon_kernel.get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT);

	// Adding +1 to sample_number here because this launch() function is called after the renderer has accumulated
	// one more sample but before render_settings.sample_number is incremented
	//
	// We also want to update the viewport at sample 0 just so that we don't get a black viewport
	// (that update at sample 0 isn't going to be a full GMoN computation, it's just going to be
	// a copy of the current pixel color (which is only 1 sample accumuluated) to the framebuffer)
	bool enough_samples_accumulated = (m_renderer->get_render_settings().sample_number + 1) % number_of_sets == 0;
	bool sample_0 = m_renderer->get_render_settings().sample_number == 0;
	if (enough_samples_accumulated || sample_0)
	{
		// If we have rendered enough samples that one more sample has been accumulated in each of the
		// GMoN sets
		int2 render_resolution = m_renderer->m_render_resolution;
		void* launch_args[] = { &m_renderer->get_render_data() };

		m_compute_gmon_kernel.launch_synchronous(
			GMoNComputeMeansKernelThreadBlockSize, GMoNComputeMeansKernelThreadBlockSize, render_resolution.x, render_resolution.y,
			launch_args);/* ,
			m_renderer->get_main_stream());*/
		std::cout << m_compute_gmon_kernel.get_last_execution_time() << "ms" << std::endl;
	}
}

bool GMoNRenderPass::pre_render_update(HIPRTRenderData& render_data)
{
	int2 render_resolution = render_data.render_settings.render_resolution;

	if (use_gmon())
	{
		if (m_gmon.current_resolution.x != render_resolution.x || m_gmon.current_resolution.y != render_resolution.y)
		{
			// Resizing the buffers because the resolution has changed
			m_gmon.resize_sets(render_resolution.x, render_resolution.y);
			m_gmon.resize_interop(render_resolution.x, render_resolution.y);

			render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;

			// Returning true to indicate that the render data buffers have been invalidated
			return true;
		}
	}
	else
	{
		m_gmon.free();

		// Returning true to indicate that the render data buffers have been invalidated
		return true;
	}

	return false;
}

void GMoNRenderPass::post_render_update(HIPRTRenderData& render_data)
{
	if (use_gmon())
	{
		// Else, if we didn't resize the buffers meaning that GMoN isn't in a fresh state, we're going to increment the
		// counter that indicates in which sets of GMoN to accumulate
		render_data.buffers.gmon_estimator.next_set_to_accumulate++;
		if (render_data.buffers.gmon_estimator.next_set_to_accumulate == m_compute_gmon_kernel.get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT))
			render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;
	}
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GMoNRenderPass::get_result_framebuffer()
{
	return m_gmon.result_framebuffer;
}

ColorRGB32F* GMoNRenderPass::get_sets_buffers_device_pointer()
{
	return m_gmon.sets.get_device_pointer();
}

void GMoNRenderPass::resize_interop_buffers(unsigned int new_width, unsigned int new_height)
{
	if (use_gmon())
		m_gmon.result_framebuffer->resize(new_width * new_height);
}

void GMoNRenderPass::resize_non_interop_buffers(unsigned int new_width, unsigned int new_height)
{
	if (use_gmon())
		m_gmon.resize_sets(new_width, new_height);
}

ColorRGB32F* GMoNRenderPass::map_result_framebuffer()
{
	return m_gmon.map_result_framebuffer();
}

void GMoNRenderPass::unmap_result_framebuffer()
{
	if (use_gmon())
		m_gmon.result_framebuffer->unmap();
}

bool GMoNRenderPass::use_gmon()
{
	return m_gmon.use_gmon && m_renderer->get_render_settings().accumulate;
}

GMoNGPUData& GMoNRenderPass::get_gmon_data()
{
	return m_gmon;
}