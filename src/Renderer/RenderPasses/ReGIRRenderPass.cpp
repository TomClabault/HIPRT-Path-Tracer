/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

#include "UI/RenderWindow.h"

const std::string ReGIRRenderPass::REGIR_GRID_PRE_POPULATE = "ReGIR Pre-population";
const std::string ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID = "ReGIR Grid fill & temp. reuse";
const std::string ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID = "ReGIR Spatial reuse";
const std::string ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID = "ReGIR Pre-integration";
const std::string ReGIRRenderPass::REGIR_REHASH_KERNEL_ID = "ReGIR Rehash kernel";
const std::string ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID = "ReGIR Supersampling copy";

const std::string ReGIRRenderPass::REGIR_RENDER_PASS_NAME = "ReGIR Render Pass";

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ REGIR_GRID_PRE_POPULATE, "ReGIR_Grid_Prepopulate" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, "ReGIR_Grid_Fill_Temporal_Reuse" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, "ReGIR_Spatial_Reuse" },
	{ REGIR_PRE_INTEGRATION_KERNEL_ID , "ReGIR_Pre_integration" },
	{ REGIR_REHASH_KERNEL_ID, "ReGIR_Rehash" },
	{ REGIR_SUPERSAMPLING_COPY_KERNEL_ID, "ReGIR_Supersampling_Copy" },
};

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FILES =
{
	{ REGIR_GRID_PRE_POPULATE, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridPrepopulate.h" },
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridFillTemporalReuse.h" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SpatialReuse.h" },
	{ REGIR_PRE_INTEGRATION_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/PreIntegration.h" },
	{ REGIR_REHASH_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/Rehash.h" },
	{ REGIR_SUPERSAMPLING_COPY_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SupersamplingCopy.h" },
};

ReGIRRenderPass::ReGIRRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReGIRRenderPass::REGIR_RENDER_PASS_NAME)
{
	m_hash_grid_storage.set_regir_render_pass(this);

	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_PRE_POPULATE));
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_PRE_POPULATE));
	m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::BSDF_OVERRIDE, BSDF_LAMBERTIAN);

	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_REHASH_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_REHASH_KERNEL_ID));

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

	if (!m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
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

bool ReGIRRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return false;
	else if (render_data.render_settings.sample_number % (render_data.render_settings.regir_settings.frame_skip + 1) != 0)
		return false;

	if (render_data.render_settings.sample_number == 0 && !m_render_window->is_interacting())
	{
		m_render_window->set_ImGui_status_text("ReGIR Prepopulation pass...");
		launch_grid_pre_population(render_data);

		m_render_window->set_ImGui_status_text("ReGIR Supersampling fill...");
		launch_supersampling_fill(render_data);
		
		m_render_window->set_ImGui_status_text("ReGIR Pre-integration...");
		launch_pre_integration(render_data);

		m_render_window->clear_ImGui_status_text();
	}

	// This needs to be called before the rehash because the
	// rehash needs the updated number of cells alive to function
	update_cell_alive_count();

	if (rehash(render_data))
	{
		// A rehashing with supersampling enabled will empty the supersampling grid so we need to fill it again
		m_render_window->set_ImGui_status_text("ReGIR Supersampling fill...");
		launch_supersampling_fill(render_data);

		// Same with the pre integration factors of the grid cells
		m_render_window->set_ImGui_status_text("ReGIR Pre-integration...");
		launch_pre_integration(render_data);

		m_render_window->clear_ImGui_status_text();
	}

	render_data.render_settings.regir_settings.supersampling.correl_reduction_current_grid = m_hash_grid_storage.get_supersampling_current_frame();
	render_data.render_settings.regir_settings.supersampling.correl_frames_available = m_hash_grid_storage.get_supersampling_frames_available();

	if (m_number_of_cells_alive > 0)
	{
		launch_grid_fill_temporal_reuse(render_data);

		launch_spatial_reuse(render_data);
	}

	return true;
}

void ReGIRRenderPass::launch_grid_pre_population(HIPRTRenderData& render_data)
{
	bool has_rehashed = false;

	do
	{
		update_cell_alive_count();

		void* launch_args[] = { &render_data };

		// Only launching / 4 in each dimension because we don't need a super high precision for the grid pre-population.
		// 
		// We just need some rays bouncing around the scene but that's it
		m_kernels[ReGIRRenderPass::REGIR_GRID_PRE_POPULATE]->launch_synchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight,
			m_renderer->m_render_resolution.x / ReGIR_GridPrepoluationResolutionDownscale, m_renderer->m_render_resolution.y / ReGIR_GridPrepoluationResolutionDownscale,
			launch_args);

		update_cell_alive_count();

		has_rehashed = rehash(render_data);
		if (has_rehashed)
			// Also updating the "true" render data of the renderer which is not the same as
			// the copy that we got here as the argument passed to this function
			update_render_data();
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

void ReGIRRenderPass::launch_grid_fill_temporal_reuse(HIPRTRenderData& render_data)
{
	void* launch_args[] = { &render_data, &m_number_of_cells_alive };

	unsigned int reservoirs_per_cell = render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell();

	// Only launching a maximum of render_resolution.x * render_resolution.y thread at a time.
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
	unsigned int nb_threads = hippt::min(m_number_of_cells_alive * reservoirs_per_cell, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));
	
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::launch_spatial_reuse(HIPRTRenderData& render_data)
{
	if (!render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
		return;

	void* launch_args[] = { &render_data, &m_number_of_cells_alive };

	unsigned int reservoirs_per_cell = render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell();

	// Same reason for nb_threads here as explained in the GridFill kernel launch
	unsigned int nb_threads = hippt::min(m_number_of_cells_alive * reservoirs_per_cell, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));
	
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::launch_supersampling_fill(HIPRTRenderData& render_data)
{
	if (!render_data.render_settings.regir_settings.supersampling.do_correlation_reduction)
		return;

	unsigned int seed_backup = render_data.random_number;

	for (int i = 0; i < render_data.render_settings.regir_settings.supersampling.correlation_reduction_factor; i++)
	{
		render_data.random_number = m_local_rng.xorshift32();

		launch_grid_fill_temporal_reuse(render_data);
		launch_spatial_reuse(render_data);
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

	m_kernels[ReGIRRenderPass::REGIR_SUPERSAMPLING_COPY_KERNEL_ID]->launch_asynchronous(64, 1, m_number_of_cells_alive * render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell(), 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::launch_pre_integration(HIPRTRenderData& render_data)
{
	unsigned int seed_backup = render_data.random_number;
	unsigned int nb_cells_alive = update_cell_alive_count();
	unsigned int nb_threads = hippt::min(nb_cells_alive, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));

	m_hash_grid_storage.clear_pre_integrated_RIS_integral_factors();

	for (int i = 0; i < REGIR_PRE_INTEGRATION_ITERATIONS; i++)
	{
		render_data.random_number = m_local_rng.xorshift32();

		launch_grid_fill_temporal_reuse(render_data);
		launch_spatial_reuse(render_data);
		
		void* launch_args[] = { &render_data, &m_number_of_cells_alive };
		m_kernels[ReGIRRenderPass::REGIR_PRE_INTEGRATION_KERNEL_ID]->launch_synchronous(64, 1, nb_threads, 1, launch_args);
	}

	render_data.random_number = seed_backup;
}

void ReGIRRenderPass::launch_rehashing_kernel(HIPRTRenderData& render_data, 
	ReGIRHashGridSoADevice& new_hash_grid_soa, ReGIRHashCellDataSoADevice& new_hash_cell_data)
{
	unsigned int* cell_alive_list_ptr = m_hash_grid_storage.get_hash_cell_data_soa().m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>();
	unsigned int old_cell_count = m_hash_grid_storage.get_hash_cell_data_soa().size();
	unsigned int old_cell_alive_count = m_number_of_cells_alive;
	
	// The old number of cells alive is the number of cells that we're going to have to rehash
	
	void* launch_args[] = { 
		&render_data.current_camera,
		
		&render_data.render_settings.regir_settings.hash_grid,
		&new_hash_grid_soa, &new_hash_cell_data,
		
		&render_data.render_settings.regir_settings.hash_cell_data, // old hash cell data
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
		render_data.render_settings.regir_settings.initial_reservoirs_grid = ReGIRHashGridSoADevice();
		render_data.render_settings.regir_settings.spatial_output_grid = ReGIRHashGridSoADevice();

		render_data.render_settings.regir_settings.hash_cell_data = ReGIRHashCellDataSoADevice();
	}
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

unsigned int ReGIRRenderPass::get_number_of_cells() const
{
	return m_hash_grid_storage.get_total_number_of_cells();
}

unsigned int ReGIRRenderPass::get_number_of_cells_alive() const
{
	return m_number_of_cells_alive;
}

unsigned int ReGIRRenderPass::update_cell_alive_count()
{
	m_hash_grid_storage.get_hash_cell_data_soa().m_grid_cells_alive_count.download_data_into(m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer());
	m_number_of_cells_alive = m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer()[0];

	return m_number_of_cells_alive;
}

float ReGIRRenderPass::get_alive_cells_ratio() const
{
	return m_number_of_cells_alive / static_cast<float>(m_hash_grid_storage.get_total_number_of_cells());
}
