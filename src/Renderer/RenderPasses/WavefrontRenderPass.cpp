/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/WavefrontRenderPass.h"

#include <algorithm>
#include <utility>

const std::string WavefrontRenderPass::WAVEFRONT_RENDER_PASS_NAME = "Wavefront Render Pass";
const std::string WavefrontRenderPass::INITIALIZE_PATHS_KERNEL	  = "Wavefront - Initialize Paths";
const std::string WavefrontRenderPass::ADVANCE_PATHS_KERNEL		  = "Wavefront - Advance Paths";

WavefrontRenderPass::WavefrontRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(WAVEFRONT_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[INITIALIZE_PATHS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + INITIALIZE_PATHS_KERNEL);
	m_kernels[INITIALIZE_PATHS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/InitializePaths.h");
	m_kernels[INITIALIZE_PATHS_KERNEL]->set_kernel_function_name("WavefrontInitializePaths");

	m_kernels[ADVANCE_PATHS_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + ADVANCE_PATHS_KERNEL);
	m_kernels[ADVANCE_PATHS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Wavefront/AdvancePaths.h");
	m_kernels[ADVANCE_PATHS_KERNEL]->set_kernel_function_name("WavefrontAdvancePaths");

	for (std::pair<const std::string, std::shared_ptr<GPUKernel>>& name_to_kernel : m_kernels)
	{
		name_to_kernel.second->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
		name_to_kernel.second->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
		name_to_kernel.second->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
	}
}

bool WavefrontRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
													   const std::vector<hiprtFuncNameSet>& func_name_sets,
													   bool silent,
													   bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool updated = false;
	for (std::pair<const std::string, std::shared_ptr<GPUKernel>>& name_to_kernel : m_kernels)
	{
		if (!name_to_kernel.second->has_been_compiled())
		{
			updated = true;
			name_to_kernel.second->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
		}
	}

	return updated;
}

void WavefrontRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool WavefrontRenderPass::pre_frame_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
	{
		bool had_buffers = m_staging_buffers_allocated;
		free_staging_buffers();
		return had_buffers;
	}

	bool resized = resize_staging_buffers();

	// Resetting this flag as this is a new frame
	render_data.render_settings.do_update_status_buffers = false;

	if (!render_data.render_settings.accumulate)
		render_data.render_settings.sample_number = 0;

	return resized;
}

bool WavefrontRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	render_data.wavefront_data = m_wavefront_data.to_device();

	unsigned int path_capacity = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	unsigned int bounce_count  = static_cast<unsigned int>(render_data.render_settings.nb_bounces);
	if (render_data.render_settings.do_render_low_resolution())
		bounce_count = std::min(3u, bounce_count);

	oroStream_t main_stream = m_renderer->get_main_stream();

	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;

	m_kernels[INITIALIZE_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_INITIALIZE_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData),
																main_stream);
	m_kernels[ADVANCE_PATHS_KERNEL]->upload_to_module_global("WAVEFRONT_ADVANCE_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData), main_stream);

	unsigned int* zero = m_zero_host_pinned.get_host_pinned_pointer();
	zero[0]			   = 0;
	m_wavefront_data.get_queue_count_buffer(0).upload_data_async(zero, main_stream);

	m_kernels[INITIALIZE_PATHS_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y,
															nullptr, main_stream);

	unsigned int input_queue_index	= 0;
	unsigned int output_queue_index = 1;
	unsigned int queue_block_size	= KernelBlockWidthHeight * KernelBlockWidthHeight;

	for (unsigned int stage_index = 0; stage_index <= bounce_count; stage_index++)
	{
		m_wavefront_data.get_queue_count_buffer(output_queue_index).upload_data_async(zero, main_stream);

		void* advance_launch_args[] = { &input_queue_index, &output_queue_index, &bounce_count };
		m_kernels[ADVANCE_PATHS_KERNEL]->launch_asynchronous(queue_block_size, 1, path_capacity, 1, advance_launch_args, main_stream);

		std::swap(input_queue_index, output_queue_index);
	}

	return true;
}

void WavefrontRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used(*m_compiler_options) && m_staging_buffers_allocated)
		render_data.wavefront_data = m_wavefront_data.to_device();
	else
		render_data.wavefront_data = WavefrontDataDevice();
}

void WavefrontRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
		return;

	if (render_data.render_settings.accumulate)
		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			render_data.render_settings.samples_per_frame = 1;

	render_data.render_settings.denoiser_AOV_accumulation_counter = 0;
	render_data.render_settings.sample_number					  = 0;
}

bool WavefrontRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	int nee_estimator	  = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	int sampling_strategy = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY);
	bool nisml_enabled	  = ILLUMINATION_AWARE_KD_TREE_IS_NISML(nee_estimator, sampling_strategy);

	return !nisml_enabled && compiler_options.get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_BSDF_WAVEFRONT;
}

bool WavefrontRenderPass::resize_staging_buffers()
{
	unsigned int path_capacity			   = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	std::size_t ray_volume_state_byte_size = m_renderer->get_render_data().ray_volume_state_byte_size;
	if (ray_volume_state_byte_size == 0)
		ray_volume_state_byte_size = sizeof(RayVolumeState);

	bool needs_resize = !m_staging_buffers_allocated || m_wavefront_data.path_capacity() != path_capacity ||
						m_allocated_ray_volume_state_byte_size != ray_volume_state_byte_size || m_render_data_host_pinned.size() != 1 ||
						m_zero_host_pinned.size() != 1;
	if (!needs_resize)
		return false;

	free_staging_buffers();

	m_wavefront_data.resize(path_capacity, ray_volume_state_byte_size);
	m_render_data_host_pinned.resize_host_pinned_mem(1);
	m_zero_host_pinned.resize_host_pinned_mem(1);

	m_allocated_ray_volume_state_byte_size = ray_volume_state_byte_size;
	m_staging_buffers_allocated			   = true;
	return true;
}

void WavefrontRenderPass::free_staging_buffers()
{
	m_wavefront_data.free();
	m_render_data_host_pinned.free_no_error();
	m_zero_host_pinned.free_no_error();

	m_allocated_ray_volume_state_byte_size = 0;
	m_staging_buffers_allocated			   = false;
}
