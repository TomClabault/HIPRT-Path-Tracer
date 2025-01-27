/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GMoNRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

const std::string GMoNRenderPass::GMON_RENDER_PASS_NAME = "GMoN Render Pass";
const std::string GMoNRenderPass::COMPUTE_GMON_KERNEL = "Compute G-MoN";

GMoNRenderPass::GMoNRenderPass() : GMoNRenderPass(nullptr) {}

GMoNRenderPass::GMoNRenderPass(GPURenderer* renderer) : RenderPass(renderer, GMoNRenderPass::GMON_RENDER_PASS_NAME)
{
	m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL] = std::make_shared<GPUKernel>();
	m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/GMoN/GMoNComputeMedianOfMeans.h");
	m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->set_kernel_function_name("GMoNComputeMedianOfMeans");
	m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->synchronize_options_with(renderer->get_global_compiler_options(), {});
}

bool GMoNRenderPass::pre_render_update(float delta_time)
{
	int2 render_resolution = m_renderer->get_render_data().render_settings.render_resolution;

	if (is_render_pass_used())
	{
		HIPRTRenderData& render_data = m_renderer->get_render_data();
		unsigned int number_of_sets = m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT);
		if (m_gmon.current_resolution.x != render_resolution.x || m_gmon.current_resolution.y != render_resolution.y)
		{
			// Resizing the buffers because the resolution has changed
			m_gmon.resize_sets(render_resolution.x, render_resolution.y, get_number_of_sets_used());
			m_gmon.resize_interop(render_resolution.x, render_resolution.y);

			render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;

			// Returning true to indicate that the render data buffers have been invalidated
			return true;
		}
		else if (number_of_sets != m_gmon.current_number_of_sets)
		{
			// If the number of sets changed...

			m_gmon.resize_sets(render_resolution.x, render_resolution.y, get_number_of_sets_used());
			render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;

			return true;
		}

		if (m_gmon.gmon_auto_blend_factor)
			// Auto adjusting the GMoN blend factor
			// 
			// Choosing the blending factor based on how many samples we've accumulated so far
			// 
			// This is just a linear ramp.
			//
			// 0 blend factor at sample number 0
			// 1 blend factor at sample number (2 * number_of_sets^2)
			m_gmon.gmon_blend_factor = hippt::clamp(0.0f, 1.0f, render_data.render_settings.sample_number / (2.0f * hippt::square(number_of_sets)));

		// Resetting the flag because we're now rendering a new frame
		m_gmon.m_gmon_recomputed = false;
	}
	else
	{
		if (!m_gmon.is_freed())
		{
			m_gmon.free();

			// Returning true to indicate that the render data buffers have been invalidated
			return true;
		}
	}

	return false;
}

bool GMoNRenderPass::launch()
{
	if (!is_render_pass_used())
		return false;

	std::shared_ptr<ApplicationSettings> application_settings = m_renderer->get_application_settings();

	unsigned int number_of_sets = m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT);

	// Adding +1 to sample_number here because this launch() function is called after the renderer has accumulated
	// one more sample but before render_settings.sample_number is incremented
	//
	// We also want to update the viewport at sample 0 just so that we don't get a black viewport
	// (that update at sample 0 isn't going to be a full GMoN computation, it's just going to be
	// a copy of the current pixel color (which is only 1 sample accumuluated) to the framebuffer)
	bool enough_samples_accumulated = (m_renderer->get_render_settings().sample_number + 1) % number_of_sets == 0;
	bool sample_0 = m_renderer->get_render_settings().sample_number == 0;
	bool last_sample_of_render = m_renderer->get_render_settings().sample_number == (application_settings->max_sample_count - 1);
	bool recomputation_necessary = m_gmon.m_gmon_recomputation_requested || last_sample_of_render;
	if ((enough_samples_accumulated || sample_0) && recomputation_necessary)
	{
		// If we have rendered enough samples that one more sample has been accumulated in each of the
		// GMoN sets
		int2 render_resolution = m_renderer->m_render_resolution;
		void* launch_args[] = { m_render_data };

		m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->launch_asynchronous(
			GMoNComputeMeansKernelThreadBlockSize, GMoNComputeMeansKernelThreadBlockSize, render_resolution.x, render_resolution.y,
			launch_args,
			m_renderer->get_main_stream());

		m_gmon.m_gmon_recomputed = true;
		m_gmon.m_gmon_recomputation_requested = false;
		m_gmon.last_recomputed_sample_count = m_renderer->get_render_settings().sample_number + 1;

		return true;
	}

	return false;
}

void GMoNRenderPass::post_render_update()
{
	if (is_render_pass_used())
	{
		HIPRTRenderData& render_data = m_renderer->get_render_data();

		// Else, if we didn't resize the buffers meaning that GMoN isn't in a fresh state, we're going to increment the
		// counter that indicates in which sets of GMoN to accumulate
		render_data.buffers.gmon_estimator.next_set_to_accumulate++;
		if (render_data.buffers.gmon_estimator.next_set_to_accumulate == m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT))
			render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;
	}
}

void GMoNRenderPass::request_recomputation()
{
	m_gmon.m_gmon_recomputation_requested = true;
}

bool GMoNRenderPass::recomputation_completed()
{
	return m_gmon.m_gmon_recomputed;
}

bool GMoNRenderPass::recomputation_requested()
{
	return m_gmon.m_gmon_recomputation_requested;
}

unsigned int GMoNRenderPass::get_last_recomputed_sample_count()
{
	return m_gmon.last_recomputed_sample_count;
}

void GMoNRenderPass::reset()
{
	if (is_render_pass_used())
	{
		m_renderer->get_render_data().buffers.gmon_estimator.next_set_to_accumulate = 0;

		if (buffers_allocated())
			m_gmon.sets.memset_whole_buffer(0);

		// Requesting a computation on reset just so that we copy the very
		// first sample to the framebuffer to avoid having a black viewport
		// until the next GMoN recomputation
		m_gmon.m_gmon_recomputation_requested = true;
	}
}

void GMoNRenderPass::update_render_data()
{
	m_renderer->get_render_data().buffers.gmon_estimator.sets = m_gmon.sets.get_device_pointer();
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GMoNRenderPass::get_result_framebuffer()
{
	return m_gmon.result_framebuffer;
}

unsigned int GMoNRenderPass::get_number_of_sets_used()
{
	return m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT);
}

void GMoNRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (is_render_pass_used())
	{
		m_gmon.resize_sets(new_width, new_height, get_number_of_sets_used());

		m_gmon.result_framebuffer->resize(new_width * new_height);
	}
}

ColorRGB32F* GMoNRenderPass::map_result_framebuffer()
{
	if (is_render_pass_used())
		return m_gmon.map_result_framebuffer();

	return nullptr;
}

void GMoNRenderPass::unmap_result_framebuffer()
{
	if (is_render_pass_used())
		m_gmon.result_framebuffer->unmap();
}

bool GMoNRenderPass::buffers_allocated()
{
	return m_gmon.sets.get_device_pointer() != nullptr;
}

bool GMoNRenderPass::is_render_pass_used() const
{
	bool gmon_enabled = m_gmon.using_gmon;
	bool accumulation_enabled = m_renderer->get_render_settings().accumulate;

	return gmon_enabled && accumulation_enabled;
}

GMoNGPUData& GMoNRenderPass::get_gmon_data()
{
	return m_gmon;
}

unsigned int GMoNRenderPass::get_VRAM_usage_bytes() const
{
	if (!is_render_pass_used())
		return 0;

	return m_gmon.get_VRAM_usage_bytes();
}

std::map<std::string, std::shared_ptr<GPUKernel>> GMoNRenderPass::get_tracing_kernels()
{
	return {};
}
