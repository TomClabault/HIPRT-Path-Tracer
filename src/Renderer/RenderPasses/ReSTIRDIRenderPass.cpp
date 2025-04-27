/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReSTIRDIRenderPass.h"
#include "Renderer/RenderPasses/ReSTIRRenderPassCommon.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"

const std::string ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID = "ReSTIR DI Initial candidates";
const std::string ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID = "ReSTIR DI Temporal reuse";
const std::string ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID = "ReSTIR DI Spatial reuse";
const std::string ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID = "ReSTIR DI Spatiotemporal reuse";
const std::string ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID = "ReSTIR DI Lights presampling";
const std::string ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID = "ReSTIR DI Directional reuse compute";

const std::string ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME = "ReSTIR DI Render Pass";

const std::unordered_map<std::string, std::string> ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID, "ReSTIR_DI_InitialCandidates" },
	{ RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID, "ReSTIR_DI_TemporalReuse" },
	{ RESTIR_DI_SPATIAL_REUSE_KERNEL_ID, "ReSTIR_DI_SpatialReuse" },
	{ RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID, "ReSTIR_DI_SpatiotemporalReuse" },
	{ RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID, "ReSTIR_DI_LightsPresampling" },
	{ RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FUNCTION_NAME },
};

const std::unordered_map<std::string, std::string> ReSTIRDIRenderPass::KERNEL_FILES =
{
	{ RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/InitialCandidates.h" },
	{ RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/TemporalReuse.h" },
	{ RESTIR_DI_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/SpatialReuse.h" },
	{ RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/FusedSpatiotemporalReuse.h" },
	{ RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/DI/LightsPresampling.h" },
	{ RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID, ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_KERNEL_FILE },
};

ReSTIRDIRenderPass::ReSTIRDIRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME)
{
	OROCHI_CHECK_ERROR(oroEventCreate(&m_spatial_reuse_time_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_spatial_reuse_time_stop));

	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_file_path(ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->set_kernel_function_name(ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 16);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 24);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->set_kernel_file_path(ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->set_kernel_function_name(ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 0);

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_file_path(ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->set_kernel_function_name(ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID));
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->synchronize_options_with(global_compiler_options);
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->get_kernel_options().set_macro_value(ReSTIRRenderPassCommon::DIRECTIONAL_REUSE_IS_RESTIR_GI_COMPILE_OPTION_NAME, KERNEL_OPTION_FALSE);
}

void ReSTIRDIRenderPass::precompile_kernels(GPUKernelCompilerOptions partial_options, std::shared_ptr<HIPRTOrochiCtx> hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets)
{
	GPUKernelCompilerOptions options;

	options = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->get_kernel_options().deep_copy();
	partial_options.apply_onto(options);
	ThreadManager::start_thread(ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS, ThreadFunctions::precompile_kernel,
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID),
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID),
		options, hiprt_orochi_ctx, std::ref(func_name_sets));

	options = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->get_kernel_options().deep_copy();
	partial_options.apply_onto(options);
	ThreadManager::start_thread(ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS, ThreadFunctions::precompile_kernel,
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID),
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID),
		options, hiprt_orochi_ctx, std::ref(func_name_sets));

	options = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().deep_copy();
	partial_options.apply_onto(options);
	ThreadManager::start_thread(ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS, ThreadFunctions::precompile_kernel,
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID),
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID),
		options, hiprt_orochi_ctx, std::ref(func_name_sets));

	options = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().deep_copy();
	partial_options.apply_onto(options);
	ThreadManager::start_thread(ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS, ThreadFunctions::precompile_kernel,
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID),
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID),
		options, hiprt_orochi_ctx, std::ref(func_name_sets));

	options = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().deep_copy();
	partial_options.apply_onto(options);
	ThreadManager::start_thread(ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS, ThreadFunctions::precompile_kernel,
		ReSTIRDIRenderPass::KERNEL_FUNCTION_NAMES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID),
		ReSTIRDIRenderPass::KERNEL_FILES.at(ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID),
		options, hiprt_orochi_ctx, std::ref(func_name_sets));

	ThreadManager::detach_threads(ThreadManager::RESTIR_DI_PRECOMPILE_KERNELS);
}

bool ReSTIRDIRenderPass::pre_render_update(float delta_time)
{
	bool render_data_invalidated = false;

	int2 render_resolution = m_renderer->m_render_resolution;

	if (is_render_pass_used())
	{
		// ReSTIR DI enabled
		bool initial_candidates_reservoir_needs_resize = m_initial_candidates_reservoirs.size() == 0;
		bool spatial_output_1_needs_resize = m_spatial_output_reservoirs_1.size() == 0;
		bool spatial_output_2_needs_resize = m_spatial_output_reservoirs_2.size() == 0;

		if (initial_candidates_reservoir_needs_resize || spatial_output_1_needs_resize || spatial_output_2_needs_resize)
			// At least on buffer is going to be resized so buffers are invalidated
			render_data_invalidated = true;

		if (initial_candidates_reservoir_needs_resize)
			m_initial_candidates_reservoirs.resize(render_resolution.x * render_resolution.y);

		if (spatial_output_1_needs_resize)
			m_spatial_output_reservoirs_1.resize(render_resolution.x * render_resolution.y);

		if (spatial_output_2_needs_resize)
			m_spatial_output_reservoirs_2.resize(render_resolution.x * render_resolution.y);



		// Also allocating / deallocating the presampled lights buffer
		if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING) == KERNEL_OPTION_TRUE)
		{
			ReSTIRDISettings& restir_di_settings = m_renderer->get_render_settings().restir_di_settings;
			int presampled_light_count = restir_di_settings.light_presampling.number_of_subsets * restir_di_settings.light_presampling.subset_size;
			bool presampled_lights_needs_allocation = m_presampled_lights_buffer.size() != presampled_light_count;

			if (presampled_lights_needs_allocation)
			{
				m_presampled_lights_buffer.resize(presampled_light_count);

				// At least on buffer is going to be resized so buffers are invalidated
				render_data_invalidated = true;
			}
		}
		else
		{
			if (m_presampled_lights_buffer.size() > 0)
				m_presampled_lights_buffer.free();
		}

		render_data_invalidated |= ReSTIRRenderPassCommon::pre_render_update_directional_reuse_buffers<false>(*m_render_data, m_renderer,
			m_per_pixel_spatial_reuse_radius,
			m_per_pixel_spatial_reuse_direction_mask_u,
			m_per_pixel_spatial_reuse_direction_mask_ull,
			m_spatial_reuse_statistics_hit_hits,
			m_spatial_reuse_statistics_hit_total);
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

		if (m_presampled_lights_buffer.size() > 0)
		{
			m_presampled_lights_buffer.free();

			render_data_invalidated = true;
		}

		render_data_invalidated |= ReSTIRRenderPassCommon::free_directional_reuse_buffers<false>(
			m_per_pixel_spatial_reuse_radius, 
			m_per_pixel_spatial_reuse_direction_mask_u,
			m_per_pixel_spatial_reuse_direction_mask_ull,
			m_spatial_reuse_statistics_hit_hits,
			m_spatial_reuse_statistics_hit_total);
	}

	if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.auto_reuse_radius)
		// A percentage of the maximum render resolution extent for automatic spatial reuse radius
		m_render_data->render_settings.restir_di_settings.common_spatial_pass.reuse_radius = hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * ReSTIRRenderPassCommon::AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE;

	return render_data_invalidated;
}

void ReSTIRDIRenderPass::update_render_data()
{
	// Setting the pointers for use in reset_render() in the camera rays kernel
	if (is_render_pass_used())
	{
		m_render_data->aux_buffers.restir_di_reservoir_buffer_1 = m_initial_candidates_reservoirs.get_device_pointer();
		m_render_data->aux_buffers.restir_di_reservoir_buffer_2 = m_spatial_output_reservoirs_1.get_device_pointer();
		m_render_data->aux_buffers.restir_di_reservoir_buffer_3 = m_spatial_output_reservoirs_2.get_device_pointer();

		if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.do_adaptive_directional_spatial_reuse(m_render_data->render_settings.accumulate))
		{
			if (m_per_pixel_spatial_reuse_direction_mask_u.size() > 0)
				m_render_data->render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u = m_per_pixel_spatial_reuse_direction_mask_u.get_device_pointer();

			if (m_per_pixel_spatial_reuse_direction_mask_ull.size() > 0)
				m_render_data->render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = m_per_pixel_spatial_reuse_direction_mask_ull.get_device_pointer();

			m_render_data->render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_radius = m_per_pixel_spatial_reuse_radius.get_device_pointer();
		}

		if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.compute_spatial_reuse_hit_rate)
		{
			m_render_data->render_settings.restir_di_settings.common_spatial_pass.spatial_reuse_hit_rate_total = m_spatial_reuse_statistics_hit_total.get_atomic_device_pointer();
			m_render_data->render_settings.restir_di_settings.common_spatial_pass.spatial_reuse_hit_rate_hits = m_spatial_reuse_statistics_hit_hits.get_atomic_device_pointer();
		}

		// If we just got ReSTIR enabled back, setting this one arbitrarily and resetting its content
		m_render_data->render_settings.restir_di_settings.restir_output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
		std::vector<ReSTIRDIReservoir> empty_reservoirs(m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y, ReSTIRDIReservoir());
		m_spatial_output_reservoirs_1.upload_data(empty_reservoirs);
	}
	else
	{
		// If ReSTIR DI is disabled, setting the pointers to nullptr so that the camera rays kernel
		// for example can detect that the buffers are freed and doesn't try to reset them or do
		// anything with them (which would lead to a crash since we would be accessing nullptr buffers)

		m_render_data->aux_buffers.restir_di_reservoir_buffer_1 = nullptr;
		m_render_data->aux_buffers.restir_di_reservoir_buffer_2 = nullptr;
		m_render_data->aux_buffers.restir_di_reservoir_buffer_3 = nullptr;

		m_render_data->render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u = nullptr;
		m_render_data->render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = nullptr;
		m_render_data->render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_radius = nullptr;
	}
}

void ReSTIRDIRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used())
		return;

	m_initial_candidates_reservoirs.resize(new_width * new_height);
	m_spatial_output_reservoirs_2.resize(new_width * new_height);
	m_spatial_output_reservoirs_1.resize(new_width * new_height);

	ReSTIRRenderPassCommon::resize_directional_reuse_buffers<false>(m_renderer, new_width, new_height, 
		m_per_pixel_spatial_reuse_radius, 
		m_per_pixel_spatial_reuse_direction_mask_u, 
		m_per_pixel_spatial_reuse_direction_mask_ull);
}

bool ReSTIRDIRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool recompiled = false;

	bool need_spatiotemporal = m_renderer->get_render_settings().restir_di_settings.do_fused_spatiotemporal && !m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_spatiotemporal;
	if (need_spatiotemporal)
		// Spatiotemporal is needed but hasn't been compiled yet
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_temporal = m_renderer->get_render_settings().restir_di_settings.common_temporal_pass.do_temporal_reuse_pass && !m_renderer->get_render_settings().restir_di_settings.do_fused_spatiotemporal && !m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_temporal;
	if (need_temporal)
		// Temporal needed
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_spatial = m_renderer->get_render_settings().restir_di_settings.common_spatial_pass.do_spatial_reuse_pass && !m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled();
	recompiled |= need_spatial;
	if (need_spatial)
		// Spatial needed
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_presampling = m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING) == KERNEL_OPTION_TRUE && !m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->has_been_compiled();
	recompiled |= need_presampling;
	if (need_presampling)
		// Light pre sampling needed
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool need_directional_spatial_reuse = !m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->has_been_compiled() && m_render_data->render_settings.restir_di_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;
	recompiled |= need_directional_spatial_reuse;
	if (need_directional_spatial_reuse)
		// Spatial needed
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return recompiled;
}

void ReSTIRDIRenderPass::reset(bool reset_by_camera_movement)
{
	odd_frame = false;
}

bool ReSTIRDIRenderPass::launch()
{
	if (!is_render_pass_used())
		return false;

	ReSTIRDISettings& restir_di_settings = m_renderer->get_render_data().render_settings.restir_di_settings;

	// Resetting the flag here just to know that we need not to read the spatial reuse
	// pass oroEvents (if that flag isn't set to true before)
	m_spatial_reuse_events_recorded = false;

	// If ReSTIR DI is enabled

	if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING) == KERNEL_OPTION_TRUE)
		launch_presampling_lights_pass();

	compute_optimal_spatial_reuse_radii();

	launch_initial_candidates_pass();

	if (m_render_data->render_settings.restir_di_settings.do_fused_spatiotemporal)
		// Launching the fused spatiotemporal kernel
		launch_spatiotemporal_pass();
	else
	{
		// Launching the temporal and spatial passes separately

		if (restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
			launch_temporal_reuse_pass();

		if (restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
			launch_spatial_reuse_passes();
	}

	configure_output_buffer();

	odd_frame = !odd_frame;

	return true;
}

void ReSTIRDIRenderPass::post_render_update()
{
	// If we had requested a temporal buffers clear, this has be done by this frame so we can
	// now reset the flag
	m_render_data->render_settings.restir_di_settings.common_temporal_pass.temporal_buffer_clear_requested = false;
}

void ReSTIRDIRenderPass::compute_optimal_spatial_reuse_radii()
{
	bool accumulating = m_render_data->render_settings.accumulate;
	bool first_frame = m_render_data->render_settings.sample_number == 0;
	bool not_interacting = m_render_data->render_settings.wants_render_low_resolution == false;
	bool using_adaptive_directional_spatial_reuse = m_render_data->render_settings.restir_di_settings.common_spatial_pass.use_adaptive_directional_spatial_reuse;

	if (accumulating && first_frame && not_interacting && using_adaptive_directional_spatial_reuse)
	{
		// If we're not accumulating, we have no guarantee that the camera isn't moving and so
		// there isn't really an "optimal" reuse radius per pixel to find
		//
		// But if the camera isn't moving, then the neighborhood of a pixel is fixed and we can optimize
		// the best spatial reuse radius
		//
		// Also, we're only doing this as a "prepass" at sample 0: we only need this once for the whole rendering

		unsigned int* per_pixel_spatial_reuse_direction_mask_u = m_per_pixel_spatial_reuse_direction_mask_u.size() > 0 ? m_per_pixel_spatial_reuse_direction_mask_u.data() : nullptr;
		unsigned long long int* per_pixel_spatial_reuse_direction_mask_ull = m_per_pixel_spatial_reuse_direction_mask_ull.size() > 0 ? m_per_pixel_spatial_reuse_direction_mask_ull.data() : nullptr;
		unsigned char* per_pixel_spatial_reuse_radius = m_per_pixel_spatial_reuse_radius.data();
		void* launch_args[] = { m_render_data, &per_pixel_spatial_reuse_direction_mask_u, &per_pixel_spatial_reuse_direction_mask_ull, &per_pixel_spatial_reuse_radius };

		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_DIRECTIONAL_REUSE_COMPUTE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
	}
}

LightPresamplingParameters ReSTIRDIRenderPass::configure_light_presampling_pass()
{
	LightPresamplingParameters parameters;
	/**
	 * Parameters specific to the kernel
	 */

	 // From all the lights of the scene, how many subsets to presample
	parameters.number_of_subsets = m_render_data->render_settings.restir_di_settings.light_presampling.number_of_subsets;
	// How many lights to presample in each subset
	parameters.subset_size = m_render_data->render_settings.restir_di_settings.light_presampling.subset_size;
	// Buffer that holds the presampled lights
	parameters.out_light_samples = m_presampled_lights_buffer.get_device_pointer();

	// For each presampled light, the probability that this is going to be an envmap sample
	parameters.envmap_sampling_probability = m_render_data->render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability;

	return parameters;
}

void ReSTIRDIRenderPass::launch_presampling_lights_pass()
{
	LightPresamplingParameters launch_parameters = configure_light_presampling_pass();

	void* launch_args[] = { &launch_parameters, m_render_data };
	int thread_count = m_render_data->render_settings.restir_di_settings.light_presampling.number_of_subsets * m_render_data->render_settings.restir_di_settings.light_presampling.subset_size;

	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->launch_asynchronous(32, 1, thread_count, 1, launch_args, m_renderer->get_main_stream());
}

void ReSTIRDIRenderPass::configure_initial_pass()
{
	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();
	if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING) == KERNEL_OPTION_TRUE)
		m_render_data->render_settings.restir_di_settings.light_presampling.light_samples = m_presampled_lights_buffer.get_device_pointer();
	else
		m_render_data->render_settings.restir_di_settings.light_presampling.light_samples = nullptr;
	m_render_data->render_settings.restir_di_settings.initial_candidates.output_reservoirs = m_initial_candidates_reservoirs.get_device_pointer();
}

void ReSTIRDIRenderPass::launch_initial_candidates_pass()
{
	void* launch_args[] = { m_render_data };

	configure_initial_pass();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
}

void ReSTIRDIRenderPass::configure_temporal_pass()
{
	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();
	m_render_data->render_settings.restir_di_settings.common_temporal_pass.permutation_sampling_random_bits = m_renderer->get_rng_generator().xorshift32();

	// The input of the temporal pass is the output of last frame's
	// ReSTIR (and also the initial candidates but this is implicit
	// and hardcoded in the shader)
	m_render_data->render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_render_data->render_settings.restir_di_settings.restir_output_reservoirs;

	if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
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
		m_render_data->render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_initial_candidates_reservoirs.get_device_pointer();
	else
	{
		// Else, no spatial reuse, the output of the temporal pass is going to be in its own buffer (because otherwise, 
		// if we output in the initial candidates buffer, then it's going to be overriden by the initial candidates pass of the next frame).
		// Alternatively using m_spatial_output_reservoirs_1 and m_spatial_output_reservoirs_2 to avoid race conditions
		if (odd_frame)
			m_render_data->render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
		else
			m_render_data->render_settings.restir_di_settings.temporal_pass.output_reservoirs = m_spatial_output_reservoirs_2.get_device_pointer();
	}
}

void ReSTIRDIRenderPass::launch_temporal_reuse_pass()
{
	void* launch_args[] = { m_render_data };

	configure_temporal_pass();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
}

void ReSTIRDIRenderPass::configure_temporal_pass_for_fused_spatiotemporal()
{
	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();
	m_render_data->render_settings.restir_di_settings.common_temporal_pass.permutation_sampling_random_bits = m_renderer->get_rng_generator().xorshift32();

	// The input of the temporal pass is the output of last frame's
	// ReSTIR (and also the initial candidates but this is implicit
	// and hardcoded in the shader)
	m_render_data->render_settings.restir_di_settings.temporal_pass.input_reservoirs = m_render_data->render_settings.restir_di_settings.restir_output_reservoirs;

	// Not needed. In the fused spatiotemporal pass, everything is output by the spatial pass
	m_render_data->render_settings.restir_di_settings.temporal_pass.output_reservoirs = nullptr;
}

void ReSTIRDIRenderPass::configure_spatial_pass(int spatial_pass_index)
{
	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();
	m_render_data->render_settings.restir_di_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

	if (spatial_pass_index == 0)
	{
		if (m_render_data->render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
			// For the first spatial reuse pass, we hardcode reading from the output of the temporal pass and storing into 'm_spatial_output_reservoirs_1'
			m_render_data->render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_render_data->render_settings.restir_di_settings.temporal_pass.output_reservoirs;
		else
			// If there is no temporal reuse pass, using the initial candidates as the input to the spatial reuse pass
			m_render_data->render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_render_data->render_settings.restir_di_settings.initial_candidates.output_reservoirs;

		m_render_data->render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
	}
	else
	{
		// And then, starting at the second spatial reuse pass, we read from the output of the previous spatial pass and store
		// in either m_spatial_output_reservoirs_1 or m_spatial_output_reservoirs_2, depending on which one isn't the input (we don't
		// want to store in the same buffers that is used for output because that's a race condition so
		// we're ping-ponging between the two outputs of the spatial reuse pass)

		if ((spatial_pass_index & 1) == 0)
		{
			m_render_data->render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_spatial_output_reservoirs_2.get_device_pointer();
			m_render_data->render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
		}
		else
		{
			m_render_data->render_settings.restir_di_settings.spatial_pass.input_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
			m_render_data->render_settings.restir_di_settings.spatial_pass.output_reservoirs = m_spatial_output_reservoirs_2.get_device_pointer();

		}
	}
}

void ReSTIRDIRenderPass::configure_spatial_pass_for_fused_spatiotemporal(int spatial_pass_index)
{
	ReSTIRDISettings& restir_settings = m_render_data->render_settings.restir_di_settings;
	restir_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

	m_render_data->random_number = m_renderer->get_rng_generator().xorshift32();

	if (spatial_pass_index == 0)
	{
		// The input of the spatial resampling in the fused spatiotemporal pass is the
		// temporal buffer of the last frame i.e. the input to the temporal pass
		//
		// Note, this line of code below assumes that the temporal pass was configured
		// prior to calling this function such that
		// 'restir_settings.temporal_pass.input_reservoirs'
		// is the proper pointer
		restir_settings.spatial_pass.input_reservoirs = restir_settings.temporal_pass.input_reservoirs;
	}
	else
	{
		// If this is not the first spatial reuse pass, the input is the output of the previous pass
		restir_settings.spatial_pass.input_reservoirs = restir_settings.spatial_pass.output_reservoirs;
	}

	// Outputting in whichever isn't the input
	if (restir_settings.spatial_pass.input_reservoirs == m_spatial_output_reservoirs_1.get_device_pointer())
		restir_settings.spatial_pass.output_reservoirs = m_spatial_output_reservoirs_2.get_device_pointer();
	else
		restir_settings.spatial_pass.output_reservoirs = m_spatial_output_reservoirs_1.get_device_pointer();
}

void ReSTIRDIRenderPass::launch_spatial_reuse_passes()
{
	void* launch_args[] = { m_render_data };

	// Emitting an event for timing all the spatial reuse passes combined
	OROCHI_CHECK_ERROR(oroEventRecord(m_spatial_reuse_time_start, m_renderer->get_main_stream()));

	for (int spatial_reuse_pass = 0; spatial_reuse_pass < m_render_data->render_settings.restir_di_settings.common_spatial_pass.number_of_passes; spatial_reuse_pass++)
	{
		configure_spatial_pass(spatial_reuse_pass);
		m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
	}

	// Emitting the stop event
	OROCHI_CHECK_ERROR(oroEventRecord(m_spatial_reuse_time_stop, m_renderer->get_main_stream()));
	m_spatial_reuse_events_recorded = true;
}

void ReSTIRDIRenderPass::configure_spatiotemporal_pass()
{
	// The buffers of the temporal pass are going to be configured in the same way
	configure_temporal_pass_for_fused_spatiotemporal();

	// But the spatial pass is going to read from the input of the temporal pass i.e. the temporal buffer of the last frame, it's not going to read from the output of the temporal pass
	configure_spatial_pass_for_fused_spatiotemporal(0);
}

void ReSTIRDIRenderPass::launch_spatiotemporal_pass()
{
	void* launch_args[] = { m_render_data };

	configure_spatiotemporal_pass();
	m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());

	if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.number_of_passes > 1)
	{
		// We have some more spatial reuse passes to do

		OROCHI_CHECK_ERROR(oroEventRecord(m_spatial_reuse_time_start, m_renderer->get_main_stream()));

		for (int spatial_pass_index = 1; spatial_pass_index < m_render_data->render_settings.restir_di_settings.common_spatial_pass.number_of_passes; spatial_pass_index++)
		{
			configure_spatial_pass_for_fused_spatiotemporal(spatial_pass_index);
			m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y, launch_args, m_renderer->get_main_stream());
		}

		// Emitting the stop event
		OROCHI_CHECK_ERROR(oroEventRecord(m_spatial_reuse_time_stop, m_renderer->get_main_stream()));
		m_spatial_reuse_events_recorded = true;
	}
}

void ReSTIRDIRenderPass::compute_render_times()
{
	if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) != LSS_RESTIR_DI)
		return;

	std::unordered_map<std::string, float>& ms_time_per_pass = m_renderer->get_render_pass_times();
	ReSTIRDISettings& restir_di_settings = m_render_data->render_settings.restir_di_settings;

	if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING) == KERNEL_OPTION_TRUE)
		ms_time_per_pass[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID] = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID]->get_last_execution_time();

	ms_time_per_pass[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID] = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID]->get_last_execution_time();
	if (restir_di_settings.do_fused_spatiotemporal)
	{
		ms_time_per_pass[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID] = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID]->get_last_execution_time();

		if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.number_of_passes > 1 && m_spatial_reuse_events_recorded)
			OROCHI_CHECK_ERROR(oroEventElapsedTime(&ms_time_per_pass[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID], m_spatial_reuse_time_start, m_spatial_reuse_time_stop));
	}
	else
	{
		if (restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
			ms_time_per_pass[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID] = m_kernels[ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID]->get_last_execution_time();

		if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.number_of_passes > 1 && m_spatial_reuse_events_recorded)
			OROCHI_CHECK_ERROR(oroEventElapsedTime(&ms_time_per_pass[ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID], m_spatial_reuse_time_start, m_spatial_reuse_time_stop));
	}
}

std::map<std::string, std::shared_ptr<GPUKernel>> ReSTIRDIRenderPass::get_all_kernels()
{
	if (!is_render_pass_used())
		return {};

	std::map<std::string, std::shared_ptr<GPUKernel>> active_kernels = m_kernels;

	ReSTIRDISettings& restir_di_settings = m_renderer->get_render_settings().restir_di_settings;
	if (restir_di_settings.do_fused_spatiotemporal)
	{
		// If using spatiotemporal, these two kernels aren't active so we're not returning them
		active_kernels.erase(ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID);

		if (m_render_data->render_settings.restir_di_settings.common_spatial_pass.number_of_passes == 1)
			// If we only have one spatial reuse pass, it's already handled by the fused spatiotemporal
			// pass so we need the spatial kernels
			active_kernels.erase(ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID);
	}
	else
		// Not using fused spatiotemporal
		active_kernels.erase(ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID);

	if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING) == KERNEL_OPTION_FALSE)
		// Not using light presampling
		active_kernels.erase(ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID);

	return active_kernels;
}

std::map<std::string, std::shared_ptr<GPUKernel>> ReSTIRDIRenderPass::get_tracing_kernels()
{
	std::map<std::string, std::shared_ptr<GPUKernel>> out = m_kernels;

	// The presampling light kernel isn't a trace kernel
	out.erase(ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID);

	return out;
}

void ReSTIRDIRenderPass::configure_output_buffer()
{
	ReSTIRDISettings& restir_di_settings = m_render_data->render_settings.restir_di_settings;

	// Keeping in mind which was the buffer used last for the output of the spatial reuse pass as this is the buffer that
	// we're going to use as the input to the temporal reuse pass of the next frame
	if (restir_di_settings.common_spatial_pass.do_spatial_reuse_pass || restir_di_settings.do_fused_spatiotemporal)
		// If there was spatial reuse, using the output of the spatial reuse pass as the input of the temporal
		// pass of next frame
		restir_di_settings.restir_output_reservoirs = restir_di_settings.spatial_pass.output_reservoirs;
	else if (restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
		// If there was a temporal reuse pass, using that output as the input of the next temporal reuse pass
		restir_di_settings.restir_output_reservoirs = restir_di_settings.temporal_pass.output_reservoirs;
	else
		// No spatial or temporal, the output of ReSTIR is just the output of the initial candidates pass
		restir_di_settings.restir_output_reservoirs = restir_di_settings.initial_candidates.output_reservoirs;
}

bool ReSTIRDIRenderPass::is_render_pass_used() const
{
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI;
}

float ReSTIRDIRenderPass::get_VRAM_usage() const
{
	return (m_initial_candidates_reservoirs.get_byte_size() + 
		m_spatial_output_reservoirs_1.get_byte_size() +
		m_spatial_output_reservoirs_2.get_byte_size() +
		m_presampled_lights_buffer.get_byte_size()) / 1000000.0f;
}
