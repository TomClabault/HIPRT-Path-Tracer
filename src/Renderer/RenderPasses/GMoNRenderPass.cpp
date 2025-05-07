/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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

bool GMoNRenderPass::pre_render_update_async(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	int2 render_resolution = render_data.render_settings.render_resolution;

	if (is_render_pass_used())
	{
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
			if (HIPRTRenderSettings::DEBUG_DEV_GMON_BLEND_WEIGHTS)
				m_gmon.gmon_blend_factor = 1.0f;// -m_darkening_factor;
			else
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

	render_data.buffers.gmon_estimator.next_set_to_accumulate = m_next_set_to_accumulate;

	return false;
}

bool GMoNRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return false;

	std::shared_ptr<ApplicationSettings> application_settings = m_renderer->get_application_settings();

	unsigned int number_of_sets = m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT);

	// Adding +1 to sample_number here because this launch() function is called after the renderer has accumulated
	// one more sample but before render_settings.sample_number is incremented
	//
	// We also want to update the viewport at sample 0 just so that we don't get a black viewport
	// (that update at sample 0 isn't going to be a full GMoN computation, it's just going to be
	// a copy of the current pixel color (which is only 1 sample accumuluated) to the framebuffer)
	bool enough_samples_accumulated = (render_data.render_settings.sample_number + 1) % number_of_sets == 0;
	bool sample_0 = render_data.render_settings.sample_number == 0;
	bool last_sample_of_render = render_data.render_settings.sample_number == (application_settings->max_sample_count - 1);
	bool recomputation_necessary = m_gmon.m_gmon_recomputation_requested || last_sample_of_render;
	if ((enough_samples_accumulated || sample_0) && recomputation_necessary)
	{
		// If we have rendered enough samples that one more sample has been accumulated in each of the
		// GMoN sets
		int2 render_resolution = m_renderer->m_render_resolution;

		render_data.buffers.gmon_estimator.next_set_to_accumulate = m_next_set_to_accumulate;

		void* launch_args[] = { &render_data };

		m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->launch_asynchronous(
			GMoNComputeMeansKernelThreadBlockSize, GMoNComputeMeansKernelThreadBlockSize, render_resolution.x, render_resolution.y,
			launch_args,
			m_renderer->get_main_stream());

		m_gmon.m_gmon_recomputed = true;
		m_gmon.m_gmon_recomputation_requested = false;
		m_gmon.last_recomputed_sample_count = render_data.render_settings.sample_number + 1;

		m_darkening_factor = compute_gmon_darkening(render_data);

		return true;
	}

	return false;
}

float GMoNRenderPass::compute_gmon_darkening(HIPRTRenderData& render_data)
{
	if (!render_data.render_settings.DEBUG_gmon_auto_blending_weights || !HIPRTRenderSettings::DEBUG_DEV_GMON_BLEND_WEIGHTS)
		return 0.0f;

	std::vector<ColorRGB32F> result = OrochiBuffer<ColorRGB32F>::download_data(m_gmon.result_framebuffer->map(), m_gmon.result_framebuffer->size());
	std::vector<ColorRGB32F> reference = OrochiBuffer<ColorRGB32F>::download_data(m_renderer->get_default_interop_framebuffer()->map(), m_gmon.result_framebuffer->size());
	std::vector<float> blend_weights_framebuffer(reference.size());

	int debug_x_1 = 589; //51
	int debug_y_1 = 24;
	int debug_x_2 = 596;
	int debug_y_2 = 34;

	for (int y = 0; y < m_renderer->m_render_resolution.y; y++)
	{
		for (int x = 0; x < m_renderer->m_render_resolution.x; x++)
		{
			int index = x + y * m_renderer->m_render_resolution.x;

			ColorRGB32F ref_color = reference[index] / render_data.render_settings.sample_number;
			ColorRGB32F result_color = result[index] / render_data.render_settings.sample_number;

			ref_color = ColorRGB32F(1.0f) - exp(-ref_color * 1.8f);
			ref_color = pow(ref_color, 1.0f / 2.2f);

			result_color = ColorRGB32F(1.0f) - exp(-result_color * 1.8f);
			result_color = pow(result_color, 1.0f / 2.2f);

			float ref_luminance = ref_color.luminance();
			float result_luminance = result_color.luminance();

			if (x == debug_x_1 && m_renderer->m_render_resolution.y - 1 - y == debug_y_1)
				std::cout << std::endl;

			if (ref_luminance - result_luminance > ref_luminance / 2.0f)
			{
				// If the pixel has lost a lot of luminance i.e. darkening, determining if this is a firefly or not

				int window_size = render_data.render_settings.DEBUG_GMON_WINDOW_SIZE;
				int valid_neighbors = 0;
				float neighbor_luminance_sum = 0.0f;
				float neighbor_luminance_average = 0.0f;

				// Computing the average of neighbors
				for (int j = -window_size / 2; j <= window_size / 2; j++)
				{
					for (int i = -window_size / 2; i <= window_size / 2; i++)
					{
						int neighbor_index_x = x + i;
						int neighbor_index_y = y + j;
						if (neighbor_index_x < 0 || neighbor_index_x >= m_renderer->m_render_resolution.x || neighbor_index_y < 0 || neighbor_index_y >= m_renderer->m_render_resolution.y)
							continue;
						else if (i == 0 && j == 0)
							// Not counting the center pixel
							continue;

						int neighbor_index = neighbor_index_x + neighbor_index_y * m_renderer->m_render_resolution.x;

						ColorRGB32F current_color = reference[neighbor_index] / render_data.render_settings.sample_number;
						current_color = ColorRGB32F(1.0f) - exp(-current_color * 1.8f);
						current_color = pow(current_color, 1.0f / 2.2f);

						valid_neighbors++;
						neighbor_luminance_sum += current_color.luminance();
					}
				}

				if (x == debug_x_1 && m_renderer->m_render_resolution.y - 1 - y == debug_y_1)
					std::cout << std::endl;

				neighbor_luminance_average = neighbor_luminance_sum / valid_neighbors;

				float brighter = ref_luminance / neighbor_luminance_average;

				blend_weights_framebuffer[index] = hippt::inverse_lerp(brighter, 1.0f, 3.0f);
			}
			else
				blend_weights_framebuffer[index] = 1.0f;














			//int window_size = 5;
			//int valid_neighbors = 0;
			//float neighbor_luminance_sum = 0.0f;
			//float neighbor_luminance_average = 0.0f;
			//std::vector<float> neighbors_luminance;
			//neighbors_luminance.reserve(window_size * window_size);

			//// Computing the average of neighbors
			//for (int i = -window_size / 2; i <= window_size / 2; i++)
			//{
			//	for (int j = -window_size / 2; j <= window_size / 2; j++)
			//	{
			//		int neighbor_index_x = x + j;
			//		int neighbor_index_y = y + i;
			//		if (neighbor_index_x < 0 || neighbor_index_x >= m_renderer->m_render_resolution.x || neighbor_index_y < 0 || neighbor_index_y >= m_renderer->m_render_resolution.y)
			//			continue;

			//		int neighbor_index = neighbor_index_x + neighbor_index_y * m_renderer->m_render_resolution.x;

			//		float current_luminance = reference[neighbor_index].luminance() / render_data.render_settings.sample_number;
			//		valid_neighbors++;
			//		neighbor_luminance_sum += current_luminance;
			//		neighbors_luminance.push_back(current_luminance);
			//	}
			//}
			//neighbor_luminance_average = neighbor_luminance_sum / valid_neighbors;

			//// Computing the variance
			//float neighbor_luminance_variance = 0.0f;
			//for (int i = -window_size / 2; i <= window_size / 2; i++)
			//{
			//	for (int j = -window_size / 2; j <= window_size / 2; j++)
			//	{
			//		int neighbor_index_x = x + j;
			//		int neighbor_index_y = y + i;
			//		if (neighbor_index_x < 0 || neighbor_index_x >= m_renderer->m_render_resolution.x || neighbor_index_y < 0 || neighbor_index_y >= m_renderer->m_render_resolution.y)
			//			continue;

			//		int neighbor_index = neighbor_index_x + neighbor_index_y * m_renderer->m_render_resolution.x;

			//		float current_luminance = reference[neighbor_index].luminance() / render_data.render_settings.sample_number;
			//		neighbor_luminance_variance += hippt::square(current_luminance - neighbor_luminance_average) / valid_neighbors;
			//	}
			//}

			//if (x == debug_x_1 && m_renderer->m_render_resolution.y - 1 - y == debug_y_1)
			//	m_DEBUG_LUMINANCE_VARIANCE1 = neighbor_luminance_variance;
			//if (x == debug_x_2 && m_renderer->m_render_resolution.y - 1 - y == debug_y_2)
			//	m_DEBUG_LUMINANCE_VARIANCE2 = neighbor_luminance_variance;

			//std::sort(neighbors_luminance.begin(), neighbors_luminance.end());

			/*if (hippt::abs(x - debug_x_1) <= 5 && hippt::abs(m_renderer->m_render_resolution.y - y - debug_y_1 - 1) <= 5)
				result[index] = ColorRGB32F(1.0e10f, 0.0f, 0.0f);*/

				//blend_weights_framebuffer[index] = 1.0f;
		}
	}

	for (int i = 0; i < result.size(); i++)
		result[i] = hippt::lerp(reference[i], result[i], blend_weights_framebuffer[i]);
	OrochiBuffer<ColorRGB32F>::upload_data(m_gmon.result_framebuffer->map(), result, m_gmon.result_framebuffer->size());

	return 0.0f;// 1.0f - result_luminance_sum / ref_luminance_sum;
}

float GMoNRenderPass::get_gmon_darkening()
{
	return m_darkening_factor;
}

float GMoNRenderPass::get_lumi()
{
	return m_DEBUG_LUMINANCE_VARIANCE1;
}

void GMoNRenderPass::post_sample_update(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (m_render_pass_used_this_frame)
	{
		// We're going to increment the counter that indicates in which sets of GMoN to accumulate
		m_next_set_to_accumulate++;
		if (m_next_set_to_accumulate == m_kernels[GMoNRenderPass::COMPUTE_GMON_KERNEL]->get_kernel_options().get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT))
			// Going back to 0 if we've reached the end of the sets, round robin style
			m_next_set_to_accumulate = 0;
	}

	render_data.buffers.gmon_estimator.next_set_to_accumulate = m_next_set_to_accumulate;
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

void GMoNRenderPass::reset(bool reset_by_camera_movement)
{
	if (is_render_pass_used())
	{
		m_next_set_to_accumulate = 0;

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
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (m_gmon.sets.is_allocated())
		render_data.buffers.gmon_estimator.sets = m_gmon.sets.get_device_pointer();
	else
		render_data.buffers.gmon_estimator.sets = nullptr;
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
	return m_gmon.sets.size() > 0;
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
