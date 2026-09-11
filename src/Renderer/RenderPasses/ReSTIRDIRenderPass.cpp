/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/KernelOptions/ReSTIRCommonOptions.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReSTIRDIRenderPass.h"
#include "Renderer/RenderPasses/ReSTIRRenderPassCommon.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

const std::string ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID		= "ReSTIR DI Initial candidates";
const std::string ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID			= "ReSTIR DI Temporal reuse";
const std::string ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID				= "ReSTIR DI Spatial reuse";
const std::string ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID = "ReSTIR DI Directional reuse compute";

const std::string ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME = "ReSTIR DI Render Pass";

const std::unordered_map<std::string, std::string> ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES = {
	{ RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID, "ReSTIR_DI_InitialCandidates" },
	{ RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID, "ReSTIR_DI_TemporalReuse" },
	{ RESTIR_DI_SPATIAL_REUSE_KERNEL_ID, "ReSTIR_DI_SpatialReuse" },
	{ RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FUNCTION_NAME },
};

const std::unordered_map<std::string, std::string> ReSTIRDIRenderPass::KERNEL_FILES = {
	{ RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/InitialCandidates.h" },
	{ RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/TemporalReuse.h" },
	{ RESTIR_DI_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/SpatialReuse.h" },
	{ RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FILE },
};

ReSTIRDIRenderPass::ReSTIRDIRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME, renderer, options)
{
	m_render_data_host_pinned.resize_host_pinned_mem(1);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_file_path(
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_function_name(
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																									GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																								GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(m_compiler_options,
																							   GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(
		GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_file_path(
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_function_name(
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->synchronize_options_with(m_compiler_options);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->get_kernel_options().set_macro_value(
		ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_RESTIR_VARIANT_COMPILE_OPTION_NAME, ReSTIR_VARIANT_DI);
}

bool ReSTIRDIRenderPass::pre_frame_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	bool render_data_invalidated = false;

	int2_t render_resolution = m_renderer->m_render_resolution;

	if (is_render_pass_used(*m_compiler_options))
	{
		// ReSTIR DI enabled
		bool initial_candidates_reservoir_needs_resize = m_initial_candidates_reservoirs.size() == 0;
		bool spatial_output_1_needs_resize			   = m_spatial_output_reservoirs_1.size() == 0;
		bool spatial_output_2_needs_resize			   = m_spatial_output_reservoirs_2.size() == 0;

		if (initial_candidates_reservoir_needs_resize || spatial_output_1_needs_resize || spatial_output_2_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			render_data_invalidated = true;

		if (initial_candidates_reservoir_needs_resize)
			m_initial_candidates_reservoirs.resize(render_resolution.x * render_resolution.y);

		if (spatial_output_1_needs_resize)
			m_spatial_output_reservoirs_1.resize(render_resolution.x * render_resolution.y);

		if (spatial_output_2_needs_resize)
			m_spatial_output_reservoirs_2.resize(render_resolution.x * render_resolution.y);

		render_data_invalidated |= ReSTIRRenderPassCommon::pre_render_update_common_buffers<ReSTIR_VARIANT_DI>(render_data, m_directional_spatial_reuse_data);
	}
	else
	{
		// ReSTIR DI disabled, we're going to free the buffers if that's not already done
		if (m_initial_candidates_reservoirs.size() > 0)
		{
			m_initial_candidates_reservoirs.free();

			render_data_invalidated = true;
		}

		if (m_spatial_output_reservoirs_1.size() > 0)
		{
			m_spatial_output_reservoirs_1.free();

			render_data_invalidated = true;
		}

		if (m_spatial_output_reservoirs_2.size() > 0)
		{
			m_spatial_output_reservoirs_2.free();

			render_data_invalidated = true;
		}

		render_data_invalidated |= ReSTIRRenderPassCommon::free_common_buffers<ReSTIR_VARIANT_DI>(m_directional_spatial_reuse_data);
	}

	if (render_data.render_settings.restir_di_settings.common_spatial_pass.auto_reuse_radius)
		// A percentage of the maximum render resolution extent for automatic spatial reuse radius
		render_data.render_settings.restir_di_settings.common_spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) *
			ReSTIRRenderPassCommon::AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE;

	return render_data_invalidated;
}

void ReSTIRDIRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	// Setting the pointers for use in reset_render() in the camera rays kernel
	if (is_render_pass_used(*m_compiler_options))
		ReSTIRRenderPassCommon::update_render_data_common_buffers<ReSTIR_VARIANT_DI>(render_data, m_directional_spatial_reuse_data);
	else
	{
		render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = nullptr;
		render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_radius			   = nullptr;
	}
}

void ReSTIRDIRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used(*m_compiler_options))
		return;

	m_initial_candidates_reservoirs.resize(new_width * new_height);
	m_spatial_output_reservoirs_2.resize(new_width * new_height);
	m_spatial_output_reservoirs_1.resize(new_width * new_height);

	ReSTIRRenderPassCommon::resize_common_buffers<ReSTIR_VARIANT_DI>(m_renderer, new_width, new_height, m_directional_spatial_reuse_data);
}

bool ReSTIRDIRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
													  const std::vector<hiprtFuncNameSet>& func_name_sets,
													  bool silent,
													  bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	bool recompiled = false;

	bool need_temporal = m_renderer->get_render_settings().restir_di_settings.common_temporal_pass.do_temporal_reuse_pass &&
						 !m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_temporal;
	if (need_temporal)
		// Temporal needed
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_spatial = m_renderer->get_render_settings().restir_di_settings.common_spatial_pass.do_spatial_reuse_pass &&
						!m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_spatial;
	if (need_spatial)
		// Spatial needed
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_directional_spatial_reuse = !m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->has_been_compiled() &&
										  render_data.render_settings.restir_di_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;
	recompiled |= need_directional_spatial_reuse;
	if (need_directional_spatial_reuse)
		// Directional spatial reuse needed
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return recompiled;
}

void ReSTIRDIRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used(*m_compiler_options))
		return;

	if (render_data.render_settings.need_to_reset)
	{
		std::vector<ReSTIRDIReservoir> empty_reservoirs(m_initial_candidates_reservoirs.size());

		if (m_initial_candidates_reservoirs.size() > 0)
			m_initial_candidates_reservoirs.upload_data(empty_reservoirs);

		if (m_spatial_output_reservoirs_1.size() > 0)
			m_spatial_output_reservoirs_1.upload_data(empty_reservoirs);

		if (m_spatial_output_reservoirs_2.size() > 0)
			m_spatial_output_reservoirs_2.upload_data(empty_reservoirs);

		if (m_spatial_output_reservoirs_1.size() > 0)
			// If we just got ReSTIR enabled back, setting this one arbitrarily and resetting its content
			m_last_restir_output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
		else
			m_last_restir_output_reservoirs = nullptr;
	}

	odd_frame = false;
}

bool ReSTIRDIRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	ReSTIRDISettings& restir_di_settings = m_renderer->get_render_data().render_settings.restir_di_settings;

	// GPUKernel records every kernel launch automatically and accumulates its execution time for the frame.

	// If ReSTIR DI is enabled

	compute_optimal_spatial_reuse_radii(render_data);

	launch_initial_candidates_pass(render_data);

	if (restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
		launch_temporal_reuse_pass(render_data);

	if (restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
		launch_spatial_reuse_passes(render_data);

	configure_output_buffer(render_data);

	odd_frame = !odd_frame;

	return true;
}

void ReSTIRDIRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	// If we had requested a temporal buffers clear, this has be done by this frame so we can
	// now reset the flag
	m_temporal_buffer_clear_requested = false;
}

void ReSTIRDIRenderPass::compute_optimal_spatial_reuse_radii(HIPRTRenderData& render_data)
{
	bool accumulating							  = render_data.render_settings.accumulate;
	bool first_frame							  = render_data.render_settings.sample_number == 0;
	bool not_interacting						  = render_data.render_settings.wants_render_low_resolution == false;
	bool using_adaptive_directional_spatial_reuse = render_data.render_settings.restir_di_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;

	if (accumulating && first_frame && not_interacting && using_adaptive_directional_spatial_reuse)
	{
		// If we're not accumulating, we have no guarantee that the camera isn't moving and so
		// there isn't really an "optimal" reuse radius per pixel to find
		//
		// But if the camera isn't moving, then the neighborhood of a pixel is fixed and we can optimize
		// the best spatial reuse radius
		//
		// Also, we're only doing this as a "prepass" at sample 0: we only need this once for the whole rendering

		unsigned long long int* per_pixel_spatial_reuse_direction_mask_ull =
			m_directional_spatial_reuse_data.m_spatial_reuse_data
				.get_buffer_data_ptr<ReSTIRDirectionalSpatialReuseDataHostBuffers::RESTIR_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL>();
		unsigned char* per_pixel_spatial_reuse_radius =
			m_directional_spatial_reuse_data.m_spatial_reuse_data
				.get_buffer_data_ptr<ReSTIRDirectionalSpatialReuseDataHostBuffers::RESTIR_DIRECTIONAL_SPATIAL_REUSE_RADIUS>();
		upload_render_data(ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, render_data);
		void* launch_args[] = { &per_pixel_spatial_reuse_direction_mask_ull, &per_pixel_spatial_reuse_radius };

		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->launch_asynchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args,
			m_renderer->get_main_stream());
	}
}

void ReSTIRDIRenderPass::configure_initial_pass(HIPRTRenderData& render_data)
{
	render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_initial_candidates_reservoirs.get_device_pointer();
}

void ReSTIRDIRenderPass::launch_initial_candidates_pass(HIPRTRenderData& render_data)
{
	configure_initial_pass(render_data);

	upload_render_data(ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID, render_data);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->launch_asynchronous(
		KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, nullptr,
		m_renderer->get_main_stream());
}

void ReSTIRDIRenderPass::configure_temporal_pass(HIPRTRenderData& render_data)
{
	render_data.render_settings.restir_di_settings.common_temporal_pass.temporal_buffer_clear_requested = m_temporal_buffer_clear_requested;

	if (m_last_restir_output_reservoirs == nullptr)
		m_last_restir_output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
	// The input of the temporal pass is the output of last frame's
	// ReSTIR (and also the initial candidates but this is implicit
	// and hardcoded in the shader)
	render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_last_restir_output_reservoirs;

	ReSTIRDIReservoir* temporal_output_reservoirs;
	if (render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
		// If we're going to do spatial reuse, reuse the initial
		// candidate reservoirs to store the output of the temporal pass.
		// The spatial reuse pass will read form that buffer.
		//
		// Reusing the initial candidates buffer (which is an input
		// to the temporal pass) as the output is legal and does not
		// cause a race condition because a given pixel only read and
		// writes to its own pixel in the initial candidates buffer.
		// We're not risking another pixel reading in someone else's
		// pixel in the initial candidates buffer while we write into
		// it (that would be a race condition)
		temporal_output_reservoirs = m_initial_candidates_reservoirs.get_device_pointer();
	else
	{
		// Else, no spatial reuse, the output of the temporal pass is going to be in its own buffer (because otherwise,
		// if we output in the initial candidates buffer, then it's going to be overriden by the initial candidates pass of the next frame).
		// Alternatively using m_spatial_output_reservoirs_1 and m_spatial_output_reservoirs_2 to avoid race conditions
		if (odd_frame)
			temporal_output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
		else
			temporal_output_reservoirs = m_spatial_output_reservoirs_2.get_device_pointer();
	}

	render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs = temporal_output_reservoirs;
}

void ReSTIRDIRenderPass::launch_temporal_reuse_pass(HIPRTRenderData& render_data)
{
	configure_temporal_pass(render_data);

	upload_render_data(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID, render_data);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight,
																						   m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y,
																						   nullptr, m_renderer->get_main_stream());
}

void ReSTIRDIRenderPass::configure_spatial_pass(HIPRTRenderData& render_data, int spatial_pass_index)
{
	render_data.render_settings.restir_di_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

	ReSTIRDIReservoir* spatial_pass_input_reservoirs  = nullptr;
	ReSTIRDIReservoir* spatial_pass_output_reservoirs = nullptr;

	if (spatial_pass_index == 0)
	{
		if (render_data.render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
			// For the first spatial reuse pass, we hardcode reading from the output of the temporal pass and storing into 'm_spatial_output_reservoirs_1'
			spatial_pass_input_reservoirs = render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs;
		else
			// If there is no temporal reuse pass, using the initial candidates as the input to the spatial reuse pass
			spatial_pass_input_reservoirs = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;

		spatial_pass_output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
	}
	else
	{
		// And then, starting at the second spatial reuse pass, we read from the output of the previous spatial pass and store
		// in either m_spatial_output_reservoirs_1 or m_spatial_output_reservoirs_2, depending on which one isn't the input (we don't
		// want to store in the same buffers that is used for output because that's a race condition so
		// we're ping-ponging between the two outputs of the spatial reuse pass)

		if ((spatial_pass_index & 1) == 0)
		{
			spatial_pass_input_reservoirs  = m_spatial_output_reservoirs_2.get_device_pointer();
			spatial_pass_output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
		}
		else
		{
			spatial_pass_input_reservoirs  = m_spatial_output_reservoirs_1.get_device_pointer();
			spatial_pass_output_reservoirs = m_spatial_output_reservoirs_2.get_device_pointer();
		}
	}

	render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs  = spatial_pass_input_reservoirs;
	render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs = spatial_pass_output_reservoirs;
}

void ReSTIRDIRenderPass::launch_spatial_reuse_passes(HIPRTRenderData& render_data)
{
	for (int spatial_reuse_pass = 0; spatial_reuse_pass < render_data.render_settings.restir_di_settings.common_spatial_pass.number_of_passes;
		 spatial_reuse_pass++)
	{
		configure_spatial_pass(render_data, spatial_reuse_pass);
		upload_render_data(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID, render_data);
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, nullptr,
			m_renderer->get_main_stream());
	}
}

void ReSTIRDIRenderPass::upload_render_data(const std::string& kernel_id, HIPRTRenderData& render_data)
{
	HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
	*host_pinned_render_data				 = render_data;

	std::string render_data_global_name =
		kernel_id == ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID ? "RESTIR_DIRECTIONAL_REUSE_RENDER_DATA" : "RESTIR_DI_RENDER_DATA";
	m_kernels[kernel_id]->upload_to_module_global(render_data_global_name.c_str(), host_pinned_render_data, sizeof(HIPRTRenderData), m_renderer->get_main_stream());
}

void ReSTIRDIRenderPass::configure_output_buffer(HIPRTRenderData& render_data)
{
	ReSTIRDISettings& restir_di_settings = render_data.render_settings.restir_di_settings;

	// Keeping in mind which was the buffer used last for the output of the spatial reuse pass as this is the buffer that
	// we're going to use as the input to the temporal reuse pass of the next frame
	if (restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
		// If there was spatial reuse, using the output of the spatial reuse pass as the input of the temporal
		// pass of next frame
		restir_di_settings.restir_output_reservoirs = restir_di_settings.spatial_pass.output_reservoirs;
	else if (restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
		// If there was a temporal reuse pass, using that output as the input of the next temporal reuse pass
		restir_di_settings.restir_output_reservoirs = restir_di_settings.temporal_pass.output_reservoirs;
	else
		// No spatial or temporal, the output of ReSTIR is just the output of the initial candidates pass
		restir_di_settings.restir_output_reservoirs = restir_di_settings.initial_candidates.output_reservoirs;

	m_last_restir_output_reservoirs = restir_di_settings.restir_output_reservoirs;
}

bool ReSTIRDIRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) == LSS_RESTIR_DI;
}

void ReSTIRDIRenderPass::request_temporal_bufffers_clear()
{
	m_temporal_buffer_clear_requested = true;
}

float ReSTIRDIRenderPass::get_VRAM_usage() const
{
	return (m_initial_candidates_reservoirs.get_byte_size() + m_spatial_output_reservoirs_1.get_byte_size() + m_spatial_output_reservoirs_2.get_byte_size() +
			m_directional_spatial_reuse_data.get_byte_size()) /
		   1000000.0f;
}
