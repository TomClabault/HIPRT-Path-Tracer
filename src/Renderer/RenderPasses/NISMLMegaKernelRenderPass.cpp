/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NISMLMegaKernelRenderPass.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

#include <algorithm>

const std::string NISMLMegaKernelRenderPass::NISML_MEGA_KERNEL_RENDER_PASS_NAME = "NISML Megakernel Render Pass";
const std::string NISMLMegaKernelRenderPass::GENERATE_QUERIES_KERNEL			= "NISML Megakernel - Generate Queries";
const std::string NISMLMegaKernelRenderPass::INFERENCE_KERNEL					= "NISML Megakernel - Inference";
const std::string NISMLMegaKernelRenderPass::RESUME_KERNEL						= "NISML Megakernel - Resume";

NISMLMegaKernelRenderPass::NISMLMegaKernelRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(NISML_MEGA_KERNEL_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[GENERATE_QUERIES_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + GENERATE_QUERIES_KERNEL);
	m_kernels[GENERATE_QUERIES_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLMegaKernelGenerateQueries.h");
	m_kernels[GENERATE_QUERIES_KERNEL]->set_kernel_function_name("NISMLMegaKernelGenerateQueries");

	m_kernels[INFERENCE_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + INFERENCE_KERNEL);
	m_kernels[INFERENCE_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLMegaKernelInference.h");
	m_kernels[INFERENCE_KERNEL]->set_kernel_function_name("NISMLMegaKernelInference");

	m_kernels[RESUME_KERNEL] = std::make_shared<GPUKernel>(this->get_name() + "::" + RESUME_KERNEL);
	m_kernels[RESUME_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Neural/NISMLMegaKernelResume.h");
	m_kernels[RESUME_KERNEL]->set_kernel_function_name("NISMLMegaKernelResume");

	for (std::pair<const std::string, std::shared_ptr<GPUKernel>>& name_to_kernel : m_kernels)
	{
		name_to_kernel.second->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
		name_to_kernel.second->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
		name_to_kernel.second->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
	}
}

bool NISMLMegaKernelRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
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

void NISMLMegaKernelRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool NISMLMegaKernelRenderPass::pre_sample_update(float delta_time)
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

bool NISMLMegaKernelRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	render_data.nisml_mega_kernel = get_device_data();

	unsigned int bounce_count = static_cast<unsigned int>(render_data.render_settings.nb_bounces);
	if (render_data.render_settings.do_render_low_resolution())
		bounce_count = std::min(3u, bounce_count);

	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;

	m_kernels[GENERATE_QUERIES_KERNEL]->upload_to_module_global("NISML_MEGAKERNEL_GENERATE_QUERIES_RENDER_DATA", host_pinned_render_data,
																sizeof(HIPRTRenderData), m_renderer->get_main_stream());
	m_kernels[INFERENCE_KERNEL]->upload_to_module_global("NISML_MEGAKERNEL_INFERENCE_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData),
														 m_renderer->get_main_stream());
	m_kernels[RESUME_KERNEL]->upload_to_module_global("NISML_MEGAKERNEL_RESUME_RENDER_DATA", host_pinned_render_data, sizeof(HIPRTRenderData),
													  m_renderer->get_main_stream());

	for (unsigned int stage_index = 0; stage_index <= bounce_count; stage_index++)
	{
		m_query_count.memset_whole_buffer(0u, m_renderer->get_main_stream());

		void* generate_queries_launch_args[] = { &stage_index };
		m_kernels[GENERATE_QUERIES_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y,
																generate_queries_launch_args, m_renderer->get_main_stream());

		m_kernels[INFERENCE_KERNEL]->launch_asynchronous(NeuralImportanceSamplingMLP::BLOCK_SIZE, 1, render_data.nisml_mega_kernel.query_capacity, 1, nullptr,
														 m_renderer->get_main_stream());

		void* resume_launch_args[] = { &stage_index };
		m_kernels[RESUME_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y,
													  resume_launch_args, m_renderer->get_main_stream());
	}

	return true;
}

void NISMLMegaKernelRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void NISMLMegaKernelRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used(*m_compiler_options) && m_staging_buffers_allocated)
		render_data.nisml_mega_kernel = get_device_data();
	else
		render_data.nisml_mega_kernel = NISMLMegaKernelDevice();
}

void NISMLMegaKernelRenderPass::reset(bool reset_by_camera_movement)
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

bool NISMLMegaKernelRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	int nee_estimator		   = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR);
	int sampling_strategy	   = compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY);
	int path_sampling_strategy = compiler_options.get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY);

	return ILLUMINATION_AWARE_KD_TREE_IS_NISML(nee_estimator, sampling_strategy) && path_sampling_strategy != PATH_SAMPLING_RESTIR_GI &&
		   path_sampling_strategy != PATH_SAMPLING_RESTIR_PT;
}

std::map<std::string, std::shared_ptr<GPUKernel>> NISMLMegaKernelRenderPass::get_tracing_kernels()
{
	if (!is_render_pass_used(*m_compiler_options))
		return {};

	std::map<std::string, std::shared_ptr<GPUKernel>> tracing_kernels;
	tracing_kernels[GENERATE_QUERIES_KERNEL] = m_kernels[GENERATE_QUERIES_KERNEL];
	tracing_kernels[INFERENCE_KERNEL]		 = m_kernels[INFERENCE_KERNEL];
	tracing_kernels[RESUME_KERNEL]			 = m_kernels[RESUME_KERNEL];
	return tracing_kernels;
}

bool NISMLMegaKernelRenderPass::resize_staging_buffers()
{
	unsigned int path_count			  = static_cast<unsigned int>(m_render_resolution.x * m_render_resolution.y);
	size_t ray_volume_state_byte_size = m_renderer->get_render_data().ray_volume_state_byte_size;
	if (ray_volume_state_byte_size == 0)
		ray_volume_state_byte_size = sizeof(RayVolumeState);

	bool needs_resize = !m_staging_buffers_allocated || m_path_data.size() != path_count || m_path_states.size() != path_count ||
						m_path_volume_states.size() != path_count || m_queries.size() != path_count ||
						m_residuals.size() != path_count * NISML_MAX_CLUSTER_COUNT || m_results.size() != path_count || m_query_count.size() != 1 ||
						m_allocated_ray_volume_state_byte_size != ray_volume_state_byte_size || m_render_data_host_pinned.size() != 1;
	if (!needs_resize)
		return false;

	free_staging_buffers();

	m_path_data.resize(path_count);
	m_path_states.resize(path_count);
	m_path_volume_states.resize(path_count, ray_volume_state_byte_size);
	m_queries.resize(path_count);
	m_residuals.resize(path_count * NISML_MAX_CLUSTER_COUNT);
	m_results.resize(path_count);
	m_query_count.resize(1);
	m_render_data_host_pinned.resize_host_pinned_mem(1);

	m_allocated_ray_volume_state_byte_size = ray_volume_state_byte_size;
	m_staging_buffers_allocated			   = true;
	return true;
}

void NISMLMegaKernelRenderPass::free_staging_buffers()
{
	m_path_data.free_no_error();
	m_path_states.free_no_error();
	m_path_volume_states.free_no_error();
	m_queries.free_no_error();
	m_residuals.free_no_error();
	m_results.free_no_error();
	m_query_count.free_no_error();
	m_render_data_host_pinned.free_no_error();

	m_allocated_ray_volume_state_byte_size = 0;
	m_staging_buffers_allocated			   = false;
}

NISMLMegaKernelDevice NISMLMegaKernelRenderPass::get_device_data()
{
	NISMLMegaKernelDevice device_data;
	device_data.path_data		   = m_path_data.get_device_pointer();
	device_data.path_states		   = m_path_states.get_device_pointer();
	device_data.path_volume_states = m_path_volume_states.get_device_pointer();
	device_data.queries			   = m_queries.get_device_pointer();
	device_data.residuals		   = m_residuals.get_device_pointer();
	device_data.results			   = m_results.get_device_pointer();
	device_data.query_count		   = m_query_count.get_atomic_device_pointer();
	device_data.path_count		   = static_cast<unsigned int>(m_path_data.size());
	device_data.query_capacity	   = static_cast<unsigned int>(m_queries.size());
	device_data.residual_stride	   = NISML_MAX_CLUSTER_COUNT;
	return device_data;
}
