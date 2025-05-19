/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

const std::string ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID = "ReGIR Grid fill & temp. reuse";
const std::string ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID = "ReGIR Spatial reuse";
const std::string ReGIRRenderPass::REGIR_REHASH_KERNEL_ID = "ReGIR Rehash kernel";

const std::string ReGIRRenderPass::REGIR_RENDER_PASS_NAME = "ReGIR Render Pass";

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, "ReGIR_Grid_Fill_Temporal_Reuse" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, "ReGIR_Spatial_Reuse" },
	{ REGIR_REHASH_KERNEL_ID, "ReGIR_Rehash" },
};

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FILES =
{
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridFillTemporalReuse.h" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SpatialReuse.h" },
	{ REGIR_REHASH_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/Rehash.h" },
};

ReGIRRenderPass::ReGIRRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReGIRRenderPass::REGIR_RENDER_PASS_NAME)
{
	m_hash_grid_storage.set_regir_render_pass(this);

	std::shared_ptr<GPUKernelCompilerOptions> global_compiler_options = m_renderer->get_global_compiler_options();

	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	// Disabling the shared memory stack traversal here because ReGIR kernels are not dispatched with a number of threads equal to the render resolution
	// which means that the global stack traversal BVH buffer may be too small to manage the traversal of all the rays that will be launched
	// in parallel by the ReGIR kernels
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->synchronize_options_with(global_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	// Disabling the shared memory stack traversal here because ReGIR kernels are not dispatched with a number of threads equal to the render resolution
	// which means that the global stack traversal BVH buffer may be too small to manage the traversal of all the rays that will be launched
	// in parallel by the ReGIR kernels
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);

	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_REHASH_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_REHASH_KERNEL_ID));
}

bool ReGIRRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool grid_fill_compiled = m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->has_been_compiled();
	if (!grid_fill_compiled)
		m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool spatial_reuse_compiled = m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->has_been_compiled();
	if (!spatial_reuse_compiled)
		m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	bool rehash_kernel_compiled = m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->has_been_compiled();
	if (!rehash_kernel_compiled)
		m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return !grid_fill_compiled || !spatial_reuse_compiled || !rehash_kernel_compiled;
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

	// This needs to be called before the rehash because the 
	// rehash needs the updated number of cells alive to function
	update_cell_alive_count();

	if (m_hash_grid_storage.try_rehash(render_data))
	{
		update_render_data();
		
		// We also want the local 'render_data' parameter here to be updated such
		// that the grid fill and spatial reuse passes can use the rehashed (and resized) grid
		m_hash_grid_storage.to_device(render_data);
	}


	render_data.render_settings.regir_settings.temporal_reuse.current_grid_index = m_current_grid_index;

	if (m_number_of_cells_alive > 0)
	{
		launch_grid_fill_temporal_reuse(render_data);
		launch_spatial_reuse(render_data);
	}

	return true;
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

void ReGIRRenderPass::launch_rehashing_kernel(HIPRTRenderData& render_data, ReGIRHashGridSoADevice& new_hash_grid, ReGIRHashCellDataSoADevice& new_hash_cell_data)
{
	unsigned int* cell_alive_list_ptr = m_hash_grid_storage.get_hash_cell_data_soa().m_hash_cell_data.template get_buffer_data_ptr<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>();
	unsigned int old_cell_count = m_hash_grid_storage.get_hash_cell_data_soa().size();
	unsigned int old_cell_alive_count = m_number_of_cells_alive;
	
	// The old number of cells alive is the number of cells that we're going to have to rehash
	
	void* launch_args[] = { 
		&render_data.current_camera.position,
		
		&new_hash_grid, &new_hash_cell_data,
		
		&render_data.render_settings.regir_settings.hash_cell_data, // old hash cell data
		&cell_alive_list_ptr, // old cell alive list		
		&old_cell_alive_count
	};
	
	m_kernels[ReGIRRenderPass::REGIR_REHASH_KERNEL_ID]->launch_synchronous(64, 1, old_cell_alive_count, 1, launch_args);

	// We need to re-upload the cell alive count because there may have possibly been severe collisions during the reinsertion
	// and maybe some cells could not be reinserted in the new hash table --> the cell alive count is different
	m_number_of_cells_alive  = m_hash_grid_storage.get_hash_cell_data_soa().m_grid_cells_alive_count.download_data()[0];
}

void ReGIRRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return;
	
	if (render_data.render_settings.regir_settings.temporal_reuse.do_temporal_reuse)
	{
		m_current_grid_index++;
		m_current_grid_index %= render_data.render_settings.regir_settings.temporal_reuse.temporal_history_length;
	}
}

void ReGIRRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used())
		m_hash_grid_storage.to_device(render_data);
	else
	{
		render_data.render_settings.regir_settings.grid_fill_grid = ReGIRHashGridSoADevice();
		render_data.render_settings.regir_settings.spatial_grid = ReGIRHashGridSoADevice();

		render_data.render_settings.regir_settings.hash_cell_data = ReGIRHashCellDataSoADevice();
	}
}

void ReGIRRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.render_settings.regir_settings.temporal_reuse.current_grid_index = 0;

	if (m_hash_grid_storage.get_byte_size() > 0)
	{
		// Resetting the 'cell alive' buffers
		std::vector<unsigned int> init_data_alive(m_hash_grid_storage.get_total_number_of_cells(), 0);
		m_hash_grid_storage.get_hash_cell_data_soa().m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>().upload_data(init_data_alive);

		// Resetting the count buffers
		unsigned int zero = 0;
		m_hash_grid_storage.get_hash_cell_data_soa().m_grid_cells_alive_count.upload_data(&zero);

		m_hash_grid_storage.reset();
	}
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
