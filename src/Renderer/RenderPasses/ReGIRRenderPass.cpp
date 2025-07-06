/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

#include "UI/RenderWindow.h"

const std::string ReGIRRenderPass::REGIR_GRID_PRE_POPULATE = "ReGIR Pre-population";
const std::string ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID = "ReGIR Grid fill 1st hits";
const std::string ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID = "ReGIR Grid fill 2nd hits";
const std::string ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID = "ReGIR Spatial reuse 1st hits";
const std::string ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID = "ReGIR Spatial reuse 2nd hits";
const std::string ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID = "ReGIR Pre-integration";
const std::string ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID = "ReGIR Pre-integration grid fill";
const std::string ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID = "ReGIR Pre-integration spatial reuse";
const std::string ReGIRRenderPass::REGIR_REHASH_KERNEL_ID = "ReGIR Rehash kernel";
const std::string ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID = "ReGIR Supersampling copy";

const std::string ReGIRRenderPass::REGIR_RENDER_PASS_NAME = "ReGIR Render Pass";

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ REGIR_GRID_PRE_POPULATE, "ReGIR_Grid_Prepopulate" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID, "ReGIR_Grid_Fill_Temporal_Reuse" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID, "ReGIR_Grid_Fill_Temporal_Reuse" },
	{ REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID, "ReGIR_Spatial_Reuse" },
	{ REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID, "ReGIR_Spatial_Reuse" },
	{ REGIR_PRE_INTEGRATION_KERNEL_ID , "ReGIR_Pre_integration" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID, "ReGIR_Grid_Fill_Temporal_Reuse"},
	{ REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID, "ReGIR_Spatial_Reuse"},
	{ REGIR_REHASH_KERNEL_ID, "ReGIR_Rehash" },
	{ REGIR_SUPERSAMPLING_COPY_KERNEL_ID, "ReGIR_Supersampling_Copy" },
};

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FILES =
{
	{ REGIR_GRID_PRE_POPULATE, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridPrepopulate.h" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridFillTemporalReuse.h" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridFillTemporalReuse.h" },
	{ REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SpatialReuse.h" },
	{ REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SpatialReuse.h" },
	{ REGIR_PRE_INTEGRATION_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/PreIntegration.h" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridFillTemporalReuse.h"},
	{ REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SpatialReuse.h"},
	{ REGIR_REHASH_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/Rehash.h" },
	{ REGIR_SUPERSAMPLING_COPY_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SupersamplingCopy.h" },
};

ReGIRRenderPass::ReGIRRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReGIRRenderPass::REGIR_RENDER_PASS_NAME)
{
	m_hash_grid_storage.set_regir_render_pass(this);
	OROCHI_CHECK_ERROR(oroStreamCreate(&m_secondary_stream));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_oro_event));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_event_pre_integration_duration_start));
	OROCHI_CHECK_ERROR(oroEventCreate(&m_event_pre_integration_duration_stop));

	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	std::unordered_set<std::string> options_not_synchronized = GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED;
	options_not_synchronized.insert(GPUKernelCompilerOptions::BSDF_OVERRIDE);
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_PRE_POPULATE));
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_PRE_POPULATE));
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->synchronize_options_with(global_compiler_options, options_not_synchronized);
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::BSDF_OVERRIDE, BSDF_LAMBERTIAN);
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);





	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);




		
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	options_not_synchronized = GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED;
	options_not_synchronized.insert(GPUKernelCompilerOptions::REGIR_GRID_FILL_SPATIAL_REUSE_ACCUMULATE_PRE_INTEGRATION);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->synchronize_options_with(global_compiler_options, options_not_synchronized);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_SPATIAL_REUSE_ACCUMULATE_PRE_INTEGRATION, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->synchronize_options_with(global_compiler_options, options_not_synchronized);
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_SPATIAL_REUSE_ACCUMULATE_PRE_INTEGRATION, KERNEL_OPTION_TRUE);





	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_REHASH_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_REHASH_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);

	m_kernels[ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID));
}

bool ReGIRRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool updated = false;

	if (!m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}





	if (!m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}





	if (!m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}





	if (!m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	return updated;
}

bool ReGIRRenderPass::pre_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();
	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	bool updated = false;

	if (is_render_pass_used())
	{
		bool storage_updated = m_hash_grid_storage.pre_render_update(render_data);
		if (storage_updated)
			m_grid_cells_alive_count_staging_host_pinned_buffer.resize_host_pinned_mem(1);

		updated |= storage_updated;
	}
	else
	{
		if (m_hash_grid_storage.free())
			updated = true;
	}

	return updated;
}

void callback_reset_imgui_status_text(void* payload)
{
	RenderWindow* render_window = reinterpret_cast<RenderWindow*>(payload);

	render_window->clear_ImGui_status_text();
}

bool ReGIRRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return false;

	if (render_data.render_settings.sample_number == 0 && !m_render_window->is_interacting())
	{
		m_render_window->set_ImGui_status_text("ReGIR Prepopulation pass...");
		launch_grid_pre_population(render_data);

		m_render_window->set_ImGui_status_text("ReGIR Supersampling fill...");
		launch_supersampling_fill(render_data);

		m_render_window->set_ImGui_status_text("ReGIR Pre-integration...");
		launch_pre_integration(render_data);

		OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_renderer->get_main_stream(), callback_reset_imgui_status_text, m_render_window));
	}

	// This needs to be called before the rehash because the
	// rehash needs the updated number of cells alive to function
	update_all_cell_alive_count();

	if (rehash(render_data))
	{
		// A rehashing with supersampling enabled will empty the supersampling grid so we need to fill it again
		m_render_window->set_ImGui_status_text("ReGIR Supersampling fill...");
		launch_supersampling_fill(render_data);

		// Same with the pre integration factors of the grid cells
		m_render_window->set_ImGui_status_text("ReGIR Pre-integration...");
		launch_pre_integration(render_data);

		OROCHI_CHECK_ERROR(oroLaunchHostFunc(m_renderer->get_main_stream(), callback_reset_imgui_status_text, m_render_window));
	}

	render_data.render_settings.regir_settings.supersampling.correl_reduction_current_grid = m_hash_grid_storage.get_supersampling_current_frame();
	render_data.render_settings.regir_settings.supersampling.correl_frames_available = m_hash_grid_storage.get_supersampling_frames_available();

	bool skip_frame_primary_hits = render_data.render_settings.sample_number % (render_data.render_settings.regir_settings.frame_skip_primary_hit_grid + 1) != 0;
	if (m_number_of_cells_alive_primary_hits > 0 && !skip_frame_primary_hits)
	{
		launch_grid_fill_temporal_reuse(render_data, true, false, m_renderer->get_main_stream());
		launch_spatial_reuse(render_data, true, false, m_renderer->get_main_stream());
	}

	bool skip_frame_secondary_hits = render_data.render_settings.sample_number % (render_data.render_settings.regir_settings.frame_skip_secondary_hit_grid + 1) != 0;
	if (m_number_of_cells_alive_secondary_hits > 0 && !skip_frame_secondary_hits)
	{
		launch_grid_fill_temporal_reuse(render_data, false, false, m_renderer->get_main_stream());
		launch_spatial_reuse(render_data, false, false, m_renderer->get_main_stream());
	}

	return true;
}

void ReGIRRenderPass::launch_grid_pre_population(HIPRTRenderData& render_data)
{
	bool has_rehashed = false;

	render_data.random_number = m_renderer->get_rng_generator().xorshift32();

	do
	{
		update_all_cell_alive_count();

		void* launch_args[] = { &render_data };

		// Only launching / 4 in each dimension because we don't need a super high precision for the grid pre-population.
		// 
		// We just need some rays bouncing around the scene but that's it
		m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->launch_synchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight,
			m_renderer->m_render_resolution.x / ReGIR_GridPrepopulationResolutionDownscale, m_renderer->m_render_resolution.y / ReGIR_GridPrepopulationResolutionDownscale,
			launch_args);

		update_all_cell_alive_count();

		has_rehashed = rehash(render_data);
	} while (has_rehashed);
}

bool ReGIRRenderPass::rehash(HIPRTRenderData& render_data)
{
	if (m_hash_grid_storage.try_rehash(render_data))
	{
		update_render_data();
		
		// We also want the local 'render_data' parameter here to be updated such
		// that the grid fill and spatial reuse passes can use the rehashed (and resized) grid
		m_hash_grid_storage.to_device(render_data);

		return true;
	}

	return false;
}

void ReGIRRenderPass::launch_grid_fill_temporal_reuse(HIPRTRenderData& render_data, bool primary_hit, bool for_pre_integration, oroStream_t stream)
{
	render_data.random_number = m_renderer->get_rng_generator().xorshift32();

	unsigned int number_of_cells_alive = primary_hit ? m_number_of_cells_alive_primary_hits : m_number_of_cells_alive_secondary_hits;
	unsigned int reservoirs_per_cell = render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(primary_hit);

	void* launch_args[] = { &render_data, &number_of_cells_alive, &primary_hit };

	// Only launching a maximum of render_resolution.x * render_resolution.y thread at a time.
	// 
	// Why? Because with visibility reuse, we're shooting rays from the kernel.
	// Shooting rays uses the global stack buffer (and shared mem) for the BVH traversal and the global
	// stack buffer is limited in size (it is sized by the number of pixels on the screen since
	// it's usually used for tracing one ray per pixel). So we need to limit the number of rays
	// that are launched per each kernel here
	//
	// So we're launching the kernel with a maximum of render_resolution.x * render_resolution.y threads so that
	// we don't overrun the global BVH traversal stack buffer
	//
	// To make sure one kernel launch still covers all the reservoirs that we have to cover, the kernel code
	// uses a while loop such that a single thread potentially computes more than 1 reservoir
	unsigned int nb_threads = hippt::min(number_of_cells_alive * reservoirs_per_cell, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));
	
	if (for_pre_integration)
		m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, stream ? stream : m_renderer->get_main_stream());
	else
	{
		if (primary_hit)
			m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, stream ? stream : m_renderer->get_main_stream());
		else
			m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, stream ? stream : m_renderer->get_main_stream());
	}
}

void ReGIRRenderPass::launch_spatial_reuse(HIPRTRenderData& render_data, bool primary_hit, bool for_pre_integration, oroStream_t stream)
{
	if (!render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
		return;

	ReGIRHashGridSoADevice input_reservoirs = render_data.render_settings.regir_settings.get_initial_reservoirs_grid(primary_hit);
	ReGIRHashGridSoADevice output_reservoirs = render_data.render_settings.regir_settings.get_raw_spatial_output_reservoirs_grid(primary_hit);
	ReGIRHashCellDataSoADevice output_reservoirs_cell_data = render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit);
	
	unsigned int number_of_cells_alive = primary_hit ? m_number_of_cells_alive_primary_hits : m_number_of_cells_alive_secondary_hits;
	unsigned int reservoirs_per_cell = render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(primary_hit);
	
	for (int i = 0; i < render_data.render_settings.regir_settings.spatial_reuse.spatial_reuse_pass_count; i++)
	{
		render_data.random_number = m_renderer->get_rng_generator().xorshift32();
		render_data.render_settings.regir_settings.spatial_reuse.spatial_reuse_pass_index = i;

		void* launch_args[] = { &render_data, &input_reservoirs, &output_reservoirs, &output_reservoirs_cell_data, &number_of_cells_alive, &primary_hit };

		// Same reason for nb_threads here as explained in the GridFill kernel launch
		unsigned int nb_threads = hippt::min(number_of_cells_alive * reservoirs_per_cell, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));

		if (for_pre_integration)
			m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FOR_PRE_INTEGRATION_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, stream ? stream : m_renderer->get_main_stream());
		else
		{
			if (primary_hit)
				m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, stream);
			else
				m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, stream);
		}

		// Swapping the input and output for the next spatial reuse apss (if any)
		std::swap(input_reservoirs, output_reservoirs);
	}
}

void ReGIRRenderPass::launch_supersampling_fill(HIPRTRenderData& render_data)
{
	if (!render_data.render_settings.regir_settings.supersampling.do_correlation_reduction)
		return;

	unsigned int seed_backup = render_data.random_number;

	for (int i = 0; i < render_data.render_settings.regir_settings.supersampling.correlation_reduction_factor; i++)
	{
		render_data.random_number = m_local_rng.xorshift32();

		launch_grid_fill_temporal_reuse(render_data, true, false, m_renderer->get_main_stream());
		launch_spatial_reuse(render_data, true, false, m_renderer->get_main_stream());
		launch_supersampling_copy(render_data);

		m_hash_grid_storage.increment_supersampling_counters(render_data);

		render_data.render_settings.regir_settings.supersampling.correl_reduction_current_grid = m_hash_grid_storage.get_supersampling_current_frame();
		render_data.render_settings.regir_settings.supersampling.correl_frames_available = m_hash_grid_storage.get_supersampling_frames_available();
	}

	render_data.random_number = seed_backup;
}

void ReGIRRenderPass::launch_supersampling_copy(HIPRTRenderData& render_data)
{
	if (!render_data.render_settings.regir_settings.supersampling.do_correlation_reduction)
		return;

	void* launch_args[] = { &render_data };

	m_kernels[ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID]->launch_asynchronous(64, 1, m_number_of_cells_alive_primary_hits * render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(true), 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::launch_pre_integration(HIPRTRenderData& render_data)
{
	update_all_cell_alive_count();

	// --------------- Record the start of the overall pre integration process
	OROCHI_CHECK_ERROR(oroEventRecord(m_event_pre_integration_duration_start, m_renderer->get_main_stream()));
	// --------------- Record the start of the overall pre integration process


	unsigned int backup = render_data.render_settings.regir_settings.grid_fill_primary_hits.light_sample_count_per_cell_reservoir;
	render_data.render_settings.regir_settings.grid_fill_primary_hits.light_sample_count_per_cell_reservoir = render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_SAMPLE_COUNT_PER_RESERVOIR;
	render_data.render_settings.regir_settings.grid_fill_secondary_hits.light_sample_count_per_cell_reservoir = render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_SAMPLE_COUNT_PER_RESERVOIR;

	// Important to launch the pre integration for the secondary hits first
	// so that we can then 
	launch_pre_integration_internal(render_data, true, m_secondary_stream);
	// The primary hit pre-integration is going to happen on the secondary stream so
	// for everything to be in order we're going to have the main stream wait for the completion
	// of the first hit pre-integration.
	//
	// Recording an event after the first pre-integration is over
	OROCHI_CHECK_ERROR(oroEventRecord(m_oro_event, m_secondary_stream));

	// Launching the pre integration for the secondary hits on another stream such that the pre integration
	// for primary and secondary hits can execute in parallell
	launch_pre_integration_internal(render_data, false, m_renderer->get_main_stream());

	// Waiting to be sure that the pre-integration for the first hits is over before continuing
	OROCHI_CHECK_ERROR(oroStreamWaitEvent(m_renderer->get_main_stream(), m_oro_event, /* oroEventWaitDefault */ 0));




	// --------------- Record the end of the overall pre integration process
	OROCHI_CHECK_ERROR(oroEventRecord(m_event_pre_integration_duration_stop, m_renderer->get_main_stream()));
	// --------------- Record the end of the overall pre integration process

	render_data.render_settings.regir_settings.grid_fill_primary_hits.light_sample_count_per_cell_reservoir = backup;
	render_data.render_settings.regir_settings.grid_fill_secondary_hits.light_sample_count_per_cell_reservoir = backup;

	m_pre_integration_executed = true;
}

void ReGIRRenderPass::launch_pre_integration_internal(HIPRTRenderData& render_data, bool primary_hit, oroStream_t stream)
{
	unsigned int seed_backup = render_data.random_number;
	unsigned int nb_cells_alive = primary_hit ? m_number_of_cells_alive_primary_hits : m_number_of_cells_alive_secondary_hits;
	unsigned int nb_threads = hippt::min(nb_cells_alive, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));

	if (nb_cells_alive == 0)
		return;

	m_hash_grid_storage.clear_pre_integrated_RIS_integral_factors(primary_hit);

	for (int i = 0; i < render_data.render_settings.DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS; i++)
	{
		render_data.random_number = m_local_rng.xorshift32();

		launch_grid_fill_temporal_reuse(render_data, primary_hit, true, stream ? stream : m_renderer->get_main_stream());
		launch_spatial_reuse(render_data, primary_hit, true, stream ? stream : m_renderer->get_main_stream());
	}

	render_data.random_number = seed_backup;
}

void ReGIRRenderPass::launch_rehashing_kernel(HIPRTRenderData& render_data, bool primary_hit, ReGIRHashGridSoADevice& new_hash_grid_soa, ReGIRHashCellDataSoADevice& new_hash_cell_data)
{
	if (render_data.render_settings.nb_bounces == 0 && !primary_hit)
		// Rehashing for the secondary hits but we don't have secondary hit grid cells because the renderer is doing 0 bounces
		return;

	unsigned int* cell_alive_list_ptr = m_hash_grid_storage.get_hash_cell_data_soa(primary_hit).m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>();
	unsigned int old_cell_count = m_hash_grid_storage.get_hash_cell_data_soa(primary_hit).size();
	unsigned int old_cell_alive_count = primary_hit ? m_number_of_cells_alive_primary_hits : m_number_of_cells_alive_secondary_hits;

	// The old number of cells alive is the number of cells that we're going to have to rehash
	
	void* launch_args[] = { 
		&render_data.current_camera,
		
		&render_data.render_settings.regir_settings.hash_grid,
		&new_hash_grid_soa, &new_hash_cell_data,
		
		&m_hash_grid_storage.get_hash_cell_data_device_soa(render_data.render_settings.regir_settings, primary_hit),
		&cell_alive_list_ptr, // old cell alive list
		&old_cell_alive_count
	};
	
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->launch_synchronous(64, 1, old_cell_alive_count, 1, launch_args);
}

void ReGIRRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return;

	launch_supersampling_copy(render_data);

	m_hash_grid_storage.post_sample_update_async(render_data);
}

void ReGIRRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used())
		m_hash_grid_storage.to_device(render_data);
	else
	{
		render_data.render_settings.regir_settings.initial_reservoirs_primary_hits_grid = ReGIRHashGridSoADevice();
		render_data.render_settings.regir_settings.initial_reservoirs_secondary_hits_grid = ReGIRHashGridSoADevice();
		render_data.render_settings.regir_settings.spatial_output_primary_hits_grid = ReGIRHashGridSoADevice();
		render_data.render_settings.regir_settings.spatial_output_secondary_hits_grid = ReGIRHashGridSoADevice();

		render_data.render_settings.regir_settings.hash_cell_data_primary_hits = ReGIRHashCellDataSoADevice();
		render_data.render_settings.regir_settings.hash_cell_data_secondary_hits = ReGIRHashCellDataSoADevice();
	}
}

void ReGIRRenderPass::compute_render_times()
{
	if (!is_render_pass_used())
		// No times to compute if the render pass is disabled / not being used
		return;

	// The default implementation iterates over all kernels and adds their time to the
	// render pass times of the renderer
	std::unordered_map<std::string, float>& render_pass_times = m_renderer->get_render_pass_times();
	for (auto& name_to_kernel : get_all_kernels())
	{
		float execution_time = m_kernels[name_to_kernel.first]->compute_execution_time();

		// Scaling the execution time based on the frame skip settings because if skipping 1 frame
		// for example, the grid fill and spatial reuse kernels essentially run every 2 frames so
		// they take twice as less time to run overall
		const std::string& kernel_name = name_to_kernel.first;
		if (kernel_name == ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID || kernel_name == ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID)
			execution_time /= m_renderer->get_render_data().render_settings.regir_settings.frame_skip_primary_hit_grid + 1;
		else if (kernel_name == ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID || kernel_name == ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID)
			execution_time /= m_renderer->get_render_data().render_settings.regir_settings.frame_skip_secondary_hit_grid + 1;
		else if (kernel_name == ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID && m_pre_integration_executed)
		{
			// Special case for the pre integration where we want to take into account the whole time
			// including the grid fill / spatial reuse passes of the pre integration and all the
			// pre integration passes at the same time.
			//
			// If we didn't override that behavior, the pre integration time would just be the time that the
			// last pre integration kernel took which is clearly inaccurate

			float duration;
			OROCHI_CHECK_ERROR(oroEventElapsedTime(&duration, m_event_pre_integration_duration_start, m_event_pre_integration_duration_stop));
			render_pass_times[name_to_kernel.first] = duration;

			continue;
		}

		render_pass_times[name_to_kernel.first] = execution_time;
	}
}

void ReGIRRenderPass::update_perf_metrics(std::shared_ptr<PerformanceMetricsComputer> perf_metrics)
{
	if (!is_render_pass_used())
		// No metrics to update if the render pass is disabled / not being used
		return;

	// Add the render pass times computed by 'compute_render_times()' (which was called before
	// 'update_perf_metrics') into the performance metrics computer
	std::unordered_map<std::string, float>& render_pass_times = m_renderer->get_render_pass_times();
	for (auto& name_to_kernel : get_all_kernels())
	{
		float execution_time = render_pass_times[name_to_kernel.first];

		const std::string& kernel_name = name_to_kernel.first;
		if (kernel_name == ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_FIRST_HITS_KERNEL_ID || kernel_name == ReGIRRenderPass::REGIR_SPATIAL_REUSE_FIRST_HITS_KERNEL_ID)
			execution_time /= m_renderer->get_render_data().render_settings.regir_settings.frame_skip_primary_hit_grid + 1;
		else if (kernel_name == ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_SECONDARY_HITS_KERNEL_ID || kernel_name == ReGIRRenderPass::REGIR_SPATIAL_REUSE_SECONDARY_HITS_KERNEL_ID)
			execution_time /= m_renderer->get_render_data().render_settings.regir_settings.frame_skip_secondary_hit_grid + 1;
		else if (kernel_name == ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID && m_pre_integration_executed)
		{
			// Special case for the pre integration where we want to take into account the whole time
			// including the grid fill / spatial reuse passes of the pre integration and all the
			// pre integration passes at the same time.
			//
			// If we didn't override that behavior, the pre integration time would just be the time that the
			// last pre integration kernel took which is clearly inaccurate

			float duration;
			OROCHI_CHECK_ERROR(oroEventElapsedTime(&duration, m_event_pre_integration_duration_start, m_event_pre_integration_duration_stop));
			perf_metrics->add_value(name_to_kernel.first, duration);

			continue;
		}

		perf_metrics->add_value(name_to_kernel.first, execution_time);
	}
}

float ReGIRRenderPass::get_full_frame_time()
{
	float sum = 0.0f;

	for (auto& name_to_kernel : get_all_kernels())
	{
		if (name_to_kernel.first == ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID ||
			name_to_kernel.first == ReGIRRenderPass::REGIR_GRID_PRE_POPULATE ||
			name_to_kernel.first == ReGIRRenderPass::REGIR_REHASH_KERNEL_ID)
			// Pre integration and pre population passes are a bit exceptional
			// so we don't want to include them in the frame time
			continue;

		sum += name_to_kernel.second->get_last_execution_time();
	}

	return sum;
}

void ReGIRRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (m_hash_grid_storage.get_byte_size() > 0)
		m_hash_grid_storage.reset();
}

bool ReGIRRenderPass::is_render_pass_used() const
{
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) == LSS_BASE_REGIR;
}

float ReGIRRenderPass::get_VRAM_usage() const
{
	return (m_hash_grid_storage.get_byte_size()) / 1000000.0f;
}

unsigned int ReGIRRenderPass::get_number_of_cells_alive(bool primary_hit) const
{
	return primary_hit ? m_number_of_cells_alive_primary_hits  : m_number_of_cells_alive_secondary_hits;
}

unsigned int ReGIRRenderPass::get_total_number_of_cells_alive(bool primary_hit) const
{
	return m_hash_grid_storage.get_total_number_of_cells(primary_hit);
}

GPURenderer* ReGIRRenderPass::get_renderer()
{
	return m_renderer;
}

void ReGIRRenderPass::update_all_cell_alive_count()
{
	m_hash_grid_storage.get_hash_cell_data_soa(true).m_grid_cells_alive_count.download_data_into(m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer());
	m_number_of_cells_alive_primary_hits = m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer()[0];

	if (m_renderer->get_render_data().render_settings.nb_bounces > 0)
	{
		m_hash_grid_storage.get_hash_cell_data_soa(false).m_grid_cells_alive_count.download_data_into(m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer());
		m_number_of_cells_alive_secondary_hits = m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer()[0];
	}
	else
		// No bounces = no secondary hit cells
		m_number_of_cells_alive_secondary_hits = 0;
}

float ReGIRRenderPass::get_alive_cells_ratio(bool primary_hit) const
{
	unsigned int total_number_of_cells = m_hash_grid_storage.get_total_number_of_cells(primary_hit);

	if (total_number_of_cells == 0)
		return 0.0f;

	return get_number_of_cells_alive(primary_hit) / static_cast<float>(total_number_of_cells);
}
