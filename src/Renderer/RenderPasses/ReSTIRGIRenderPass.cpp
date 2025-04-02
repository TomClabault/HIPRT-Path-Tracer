/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReSTIRGIRenderPass.h"
#include "Renderer/RenderPasses/ReSTIRRenderPassCommon.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

const std::string ReSTIRGIRenderPass::RESTIR_GI_RENDER_PASS_NAME = "ReSTIR GI Render pass";
const std::string ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID = "ReSTIR GI Initial candidates";
const std::string ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID = "ReSTIR GI Temporal reuse";
const std::string ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID = "ReSTIR GI Spatial reuse";
const std::string ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID = "ReSTIR GI Shading";
const std::string ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID = "ReSTIR GI Directional reuse compute";

const std::unordered_map<std::string, std::string> ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID, "ReSTIR_GI_InitialCandidates" },
	{ RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID, "ReSTIR_GI_TemporalReuse" },
	{ RESTIR_GI_SPATIAL_REUSE_KERNEL_ID, "ReSTIR_GI_SpatialReuse" },
	{ RESTIR_GI_SHADING_KERNEL_ID, "ReSTIR_GI_Shading" },
	{ RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FUNCTION_NAME },
};

const std::unordered_map<std::string, std::string> ReSTIRGIRenderPass::KERNEL_FILES =
{
	{ RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/GI/InitialCandidates.h"},
	{ RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/GI/TemporalReuse.h" },
	{ RESTIR_GI_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/GI/SpatialReuse.h" },
	{ RESTIR_GI_SHADING_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/GI/Shading.h" },
	{ RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FILE },
};

ReSTIRGIRenderPass::ReSTIRGIRenderPass() : ReSTIRGIRenderPass(nullptr) {}
ReSTIRGIRenderPass::ReSTIRGIRenderPass(GPURenderer* renderer) : MegaKernelRenderPass(renderer, ReSTIRGIRenderPass::RESTIR_GI_RENDER_PASS_NAME) 
{
	OROCHI_CHECK_ERROR(oroEventCreate(&m_spatial_reuse_time_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_spatial_reuse_time_stop));

	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_file_path(ReSTIRGIRenderPass::KERNEL_FILES.at(ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_function_name(ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReSTIRGIRenderPass::KERNEL_FILES.at(ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReSTIRGIRenderPass::KERNEL_FILES.at(ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
	
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID]->set_kernel_file_path(ReSTIRGIRenderPass::KERNEL_FILES.at(ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID]->set_kernel_function_name(ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_file_path(ReSTIRGIRenderPass::KERNEL_FILES.at(ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_function_name(ReSTIRGIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->synchronize_options_with(global_compiler_options);
	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->get_kernel_options().set_macro_value(ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_IS_RESTIR_GI_COMPILE_OPTION_NAME, KERNEL_OPTION_TRUE);
}

void ReSTIRGIRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used())
		return;

	m_initial_candidates_buffer.resize(new_width * new_height);
	m_temporal_buffer.resize(new_width * new_height);
	m_spatial_buffer.resize(new_width * new_height);

	ReSTIRRenderPassCommon::resize_directional_reuse_buffers<true>(m_renderer, new_width, new_height, 
		m_per_pixel_spatial_reuse_radius, 
		m_per_pixel_spatial_reuse_direction_mask_u, 
		m_per_pixel_spatial_reuse_direction_mask_ull);
}

bool ReSTIRGIRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool recompiled = false;

	bool need_temporal = m_render_data->render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass && !m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_temporal;
	if (need_temporal)
		// Temporal needed
		m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_spatial = m_render_data->render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass && !m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_spatial;
	if (need_spatial)
		// Spatial needed
		m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_directional_spatial_reuse = !m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->has_been_compiled() && m_render_data->render_settings.restir_gi_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;
	recompiled |= need_directional_spatial_reuse;
	if (need_directional_spatial_reuse)
		// Spatial needed
		m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return recompiled;
}

bool ReSTIRGIRenderPass::pre_render_update(float delta_time)
{
	MegaKernelRenderPass::pre_render_update(delta_time);

	bool render_data_invalidated = false;

	int2 render_resolution = m_renderer->m_render_resolution;

	if (is_render_pass_used())
	{
		// ReSTIR GI enabled
		bool initial_candidates_reservoir_needs_resize = m_initial_candidates_buffer.get_element_count() == 0;
		bool temporal_candidates_reservoir_needs_resize = m_temporal_buffer.get_element_count() == 0;
		bool spatial_candidates_reservoir_needs_resize = m_spatial_buffer.get_element_count() == 0;

		if (initial_candidates_reservoir_needs_resize || temporal_candidates_reservoir_needs_resize || spatial_candidates_reservoir_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			render_data_invalidated = true;

		if (initial_candidates_reservoir_needs_resize)
			m_initial_candidates_buffer.resize(render_resolution.x * render_resolution.y);

		if (temporal_candidates_reservoir_needs_resize)
			m_temporal_buffer.resize(render_resolution.x * render_resolution.y);

		if (spatial_candidates_reservoir_needs_resize)
			m_spatial_buffer.resize(render_resolution.x * render_resolution.y);

		render_data_invalidated |= ReSTIRRenderPassCommon::pre_render_update_directional_reuse_buffers<true>(*m_render_data, m_renderer,
			m_per_pixel_spatial_reuse_radius, 
			m_per_pixel_spatial_reuse_direction_mask_u,
			m_per_pixel_spatial_reuse_direction_mask_ull,
			m_spatial_reuse_statistics_hit_hits,
			m_spatial_reuse_statistics_hit_total);
	}
	else
	{
		// ReSTIR GI disabled, we're going to free the buffers if that's not already done
		if (m_initial_candidates_buffer.get_element_count() > 0)
		{
			m_initial_candidates_buffer.free();

			render_data_invalidated = true;
		}

		if (m_temporal_buffer.get_element_count() > 0)
		{
			m_temporal_buffer.free();

			render_data_invalidated = true;
		}

		if (m_spatial_buffer.get_element_count() > 0)
		{
			m_spatial_buffer.free();

			render_data_invalidated = true;
		}

		render_data_invalidated |= ReSTIRRenderPassCommon::free_directional_reuse_buffers<true>(
			m_per_pixel_spatial_reuse_radius,
			m_per_pixel_spatial_reuse_direction_mask_u,
			m_per_pixel_spatial_reuse_direction_mask_ull,
			m_spatial_reuse_statistics_hit_hits,
			m_spatial_reuse_statistics_hit_total);
	}

	if (m_render_data->render_settings.restir_gi_settings.common_spatial_pass.auto_reuse_radius)
		// A percentage of the maximum render resolution extent for automatic spatial reuse radius
		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.reuse_radius = hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * ReSTIRRenderPassCommon::AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE;

	return render_data_invalidated;
}

void ReSTIRGIRenderPass::compute_optimal_spatial_reuse_radii()
{
	bool accumulating = m_render_data->render_settings.accumulate;
	bool first_frame = m_render_data->render_settings.sample_number == 0;
	bool not_interacting = m_render_data->render_settings.wants_render_low_resolution == false;
	bool using_adaptive_directional_spatial_reuse = m_render_data->render_settings.restir_gi_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;

	if (accumulating && first_frame && not_interacting && using_adaptive_directional_spatial_reuse)
	{
		// If we're not accumulating, we have no guarantee that the camera isn't moving and so
		// there isn't really an "optimal" reuse radius per pixel to find
		//
		// But if the camera isn't moving, then the neighborhood of a pixel is fixed and we can optimize
		// the best spatial reuse radius
		//
		// Also, we're only doing this as a "prepass" at sample 0: we only need this once for the whole rendering

		void* launch_args[] = { m_render_data, m_per_pixel_spatial_reuse_direction_mask_u.get_device_pointer_address(), m_per_pixel_spatial_reuse_direction_mask_ull.get_device_pointer_address(), m_per_pixel_spatial_reuse_radius.get_device_pointer_address() };

		m_kernels[ReSTIRGIRenderPass::RESTIR_GI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
	}
}

void ReSTIRGIRenderPass::configure_initial_candidates_pass()
{
	m_render_data->render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer = m_initial_candidates_buffer.get_device_pointer();

	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();
}

static unsigned int seed;

void ReSTIRGIRenderPass::launch_initial_candidates_pass()
{
	void* launch_args[] = { m_render_data };

	seed = m_render_data->random_number;
	if (m_render_data->render_settings.nb_bounces > 0)
		// We only need to trace paths for the initial candidates if we have
		// more than 1 bounce
		m_kernels[ReSTIRGIRenderPass::RESTIR_GI_INITIAL_CANDIDATES_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
}

void ReSTIRGIRenderPass::configure_temporal_reuse_pass()
{
	if (m_render_data->render_settings.sample_number == 0)
		// First frame, using the initial candidates as the input
		m_render_data->render_settings.restir_gi_settings.temporal_pass.input_reservoirs = m_render_data->render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer;
	else
		// Not the first frame, the input to the temporal pass is the output of the last frame ReSTIR
		m_render_data->render_settings.restir_gi_settings.temporal_pass.input_reservoirs = m_render_data->render_settings.restir_gi_settings.restir_output_reservoirs;

	// For the output, using whatever buffer isn't the one we're reading from (the input buffer)
	if (m_render_data->render_settings.restir_gi_settings.temporal_pass.input_reservoirs == m_spatial_buffer.get_device_pointer())
		m_render_data->render_settings.restir_gi_settings.temporal_pass.output_reservoirs = m_temporal_buffer.get_device_pointer();
	else
		m_render_data->render_settings.restir_gi_settings.temporal_pass.output_reservoirs = m_spatial_buffer.get_device_pointer();

	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();
}

void ReSTIRGIRenderPass::launch_temporal_reuse_pass()
{
	void* launch_args[] = { m_render_data };

	if (m_renderer->get_render_settings().restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
		m_kernels[ReSTIRGIRenderPass::RESTIR_GI_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, 0);// m_renderer->get_main_stream());
}

void ReSTIRGIRenderPass::configure_spatial_reuse_pass(int spatial_pass_index)
{
	m_render_data->render_settings.restir_gi_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

	// The spatial reuse pass spatially reuse on the output of the temporal pass in the 'temporal buffer' and
	// stores in the 'spatial buffer'

	ReSTIRGIReservoir* input_reservoirs;
	ReSTIRGIReservoir* output_reservoirs;

	if (m_render_data->render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
		// and we have a temporal reuse pass so we're going to read from the temporal reservoirs
		input_reservoirs = m_render_data->render_settings.restir_gi_settings.temporal_pass.output_reservoirs;
	else
		// and we do not have a temporal reuse pass so we're just going to read from the initial candidates
		input_reservoirs = m_initial_candidates_buffer.get_device_pointer();

	if (input_reservoirs == m_temporal_buffer.get_device_pointer())
		output_reservoirs = m_spatial_buffer.get_device_pointer();
	else
		output_reservoirs = m_temporal_buffer.get_device_pointer();

	m_render_data->render_settings.restir_gi_settings.spatial_pass.input_reservoirs = input_reservoirs;
	m_render_data->render_settings.restir_gi_settings.spatial_pass.output_reservoirs = output_reservoirs;

	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();
}

void ReSTIRGIRenderPass::launch_spatial_reuse_pass()
{
	void* launch_args[] = { m_render_data };

	if (m_renderer->get_render_settings().restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass)
		m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
}

void ReSTIRGIRenderPass::configure_shading_pass()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (render_data.render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass)
		render_data.render_settings.restir_gi_settings.restir_output_reservoirs = render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs;
	else if (render_data.render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
		render_data.render_settings.restir_gi_settings.restir_output_reservoirs = render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs;
	else
		render_data.render_settings.restir_gi_settings.restir_output_reservoirs = render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer;

	m_render_data->random_number = seed;
}

void ReSTIRGIRenderPass::launch_shading_pass()
{
	void* launch_args[] = { m_render_data };

	m_kernels[ReSTIRGIRenderPass::RESTIR_GI_SHADING_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
}

bool ReSTIRGIRenderPass::launch()
{
	if (!is_render_pass_used())
		return false;

	compute_optimal_spatial_reuse_radii();

	configure_initial_candidates_pass();
	launch_initial_candidates_pass();

	/*m_renderer->synchronize_kernel();
	std::vector<ReSTIRGIReservoir> initial_cand = m_initial_candidates_buffer.download_data();
	for (ReSTIRGIReservoir& res : initial_cand)
		if (hippt::is_nan(res.UCW) || hippt::is_inf(res.UCW))
			Utils::debugbreak();*/

	configure_temporal_reuse_pass();
	launch_temporal_reuse_pass();

	/*m_renderer->synchronize_kernel();
	initial_cand = OrochiBuffer<ReSTIRGIReservoir>::download_data(m_render_data->render_settings.restir_gi_settings.temporal_pass.output_reservoirs, m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);
	for (ReSTIRGIReservoir& res : initial_cand)
		if (hippt::is_nan(res.UCW) || hippt::is_inf(res.UCW))
			Utils::debugbreak();*/
	
	for (int i = 0; i < m_render_data->render_settings.restir_gi_settings.common_spatial_pass.number_of_passes; i++)
	{
		configure_spatial_reuse_pass(i);
		launch_spatial_reuse_pass();

		/*m_renderer->synchronize_kernel();
		initial_cand = OrochiBuffer<ReSTIRGIReservoir>::download_data(m_render_data->render_settings.restir_gi_settings.spatial_pass.output_reservoirs, m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y);
		for (ReSTIRGIReservoir& res : initial_cand)
			if (hippt::is_nan(res.UCW) || hippt::is_inf(res.UCW))
				Utils::debugbreak();*/
	}

	configure_shading_pass();
	launch_shading_pass();

	return true;
}

void ReSTIRGIRenderPass::post_render_update()
{
	// If we had requested a temporal buffers clear, this has be done by this frame so we can
	// now reset the flag
	m_render_data->render_settings.restir_gi_settings.common_temporal_pass.temporal_buffer_clear_requested = false;

	MegaKernelRenderPass::post_render_update();
}

void ReSTIRGIRenderPass::update_render_data()
{
	// Setting the pointers for use in reset_render() in the camera rays kernel
	if (is_render_pass_used())
	{
		m_render_data->aux_buffers.restir_gi_reservoir_buffer_1 = m_initial_candidates_buffer.get_device_pointer();
		m_render_data->aux_buffers.restir_gi_reservoir_buffer_2 = m_spatial_buffer.get_device_pointer();
		m_render_data->aux_buffers.restir_gi_reservoir_buffer_3 = m_temporal_buffer.get_device_pointer();

		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u = m_per_pixel_spatial_reuse_direction_mask_u.get_device_pointer();
		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = m_per_pixel_spatial_reuse_direction_mask_ull.get_device_pointer();
		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius = m_per_pixel_spatial_reuse_radius.get_device_pointer();

		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.spatial_reuse_hit_rate_total = reinterpret_cast<AtomicType<unsigned long long int>*>(m_spatial_reuse_statistics_hit_total.get_device_pointer());
		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.spatial_reuse_hit_rate_hits = reinterpret_cast<AtomicType<unsigned long long int>*>(m_spatial_reuse_statistics_hit_hits.get_device_pointer());
	}
	else
	{
		// If ReSTIR GI is disabled, setting the pointers to nullptr so that the camera rays kernel
		// for example can detect that the buffers are freed and doesn't try to reset them or do
		// anything with them (which would be invalid since we would be accessing nullptr buffers)

		m_render_data->aux_buffers.restir_gi_reservoir_buffer_1 = nullptr;
		m_render_data->aux_buffers.restir_gi_reservoir_buffer_2 = nullptr;
		m_render_data->aux_buffers.restir_gi_reservoir_buffer_3 = nullptr;

		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u = nullptr;
		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = nullptr;
		m_render_data->render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius = nullptr;
	}
}

void ReSTIRGIRenderPass::reset()
{
	if (m_spatial_reuse_statistics_hit_hits.get_element_count() > 0)
	{
		m_spatial_reuse_statistics_hit_hits.memset_whole_buffer(0);
		m_spatial_reuse_statistics_hit_total.memset_whole_buffer(0);
	}

	MegaKernelRenderPass::reset();
}

std::map<std::string, std::shared_ptr<GPUKernel>> ReSTIRGIRenderPass::get_tracing_kernels()
{
	return get_all_kernels();
}

bool ReSTIRGIRenderPass::is_render_pass_used() const
{
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PSS_RESTIR_GI;
}

float ReSTIRGIRenderPass::get_VRAM_usage() const
{
	return (m_initial_candidates_buffer.get_byte_size() + 
		m_temporal_buffer.get_byte_size() + 
		m_spatial_buffer.get_byte_size() + 
		m_per_pixel_spatial_reuse_direction_mask_u.get_byte_size() +
		m_per_pixel_spatial_reuse_direction_mask_ull.get_byte_size() +
		m_per_pixel_spatial_reuse_radius.get_byte_size()) / 1000000.0f;
}
