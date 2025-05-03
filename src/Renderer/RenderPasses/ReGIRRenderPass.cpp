/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

const std::string ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID = "ReGIR Grid fill & temp. reuse";
const std::string ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID = "ReGIR Spatial reuse";
const std::string ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID = "ReGIR Cell liveness copy";

const std::string ReGIRRenderPass::REGIR_RENDER_PASS_NAME = "ReGIR Render Pass";

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, "ReGIR_Grid_Fill_Temporal_Reuse" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, "ReGIR_Spatial_Reuse" },
	{ REGIR_CELL_LIVENESS_COPY_KERNEL_ID, "ReGIR_CellLivenessCopy" },
};

const std::unordered_map<std::string, std::string> ReGIRRenderPass::KERNEL_FILES =
{
	{ REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/GridFillTemporalReuse.h" },
	{ REGIR_SPATIAL_REUSE_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/SpatialReuse.h" },
	{ REGIR_CELL_LIVENESS_COPY_KERNEL_ID, DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReGIR/CellLivenessCopy.h" },
};

ReGIRRenderPass::ReGIRRenderPass(GPURenderer* renderer) : RenderPass(renderer, ReGIRRenderPass::REGIR_RENDER_PASS_NAME)
{
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

	m_kernels[ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID]->set_kernel_file_path(ReGIRRenderPass::KERNEL_FILES.at(ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID));
	m_kernels[ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID]->set_kernel_function_name(ReGIRRenderPass::KERNEL_FUNCTION_NAMES.at(ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID));
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

	bool cell_liveness_copy_compiled = m_kernels[ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID]->has_been_compiled();
	if (!cell_liveness_copy_compiled)
		m_kernels[ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return !grid_fill_compiled || !spatial_reuse_compiled || !cell_liveness_copy_compiled;
}

bool ReGIRRenderPass::pre_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	bool updated = false;

	if (is_render_pass_used())
	{
		// Resizing the grid if it is not the right size
		if (m_grid_buffers.size() != render_data.render_settings.regir_settings.get_total_number_of_reservoirs_ReGIR())
		{
			m_grid_buffers.resize(render_data.render_settings.regir_settings.get_total_number_of_reservoirs_ReGIR());

			updated = true;
		}

		if (render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse && m_spatial_reuse_output_grid_buffer.size() != render_data.render_settings.regir_settings.get_number_of_reservoirs_per_grid())
		{
			m_spatial_reuse_output_grid_buffer.resize(render_data.render_settings.regir_settings.get_number_of_reservoirs_per_grid());

			updated = true;
		}

		if (m_grid_cells_alive_buffer.size() != render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid())
		{
			m_grid_cells_alive_staging_buffer.resize(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid());
			m_grid_cells_alive_buffer.resize(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid());
			m_grid_cells_alive_list_buffer.resize(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid());
			m_grid_cells_alive_count_staging_buffer.resize(1);
			m_grid_cells_alive_count_staging_host_pinned_buffer.resize_host_pinned_mem(1);

			std::vector<unsigned int> init_data_inactive(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid(), 0u);
			m_grid_cells_alive_staging_buffer.upload_data(init_data_inactive);

			// Initializing all the cells to alive
			std::vector<unsigned char> init_data_alive(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid(), 1u);
			m_grid_cells_alive_buffer.upload_data(init_data_alive);

			// All cells are alive at the beginning of the render
			unsigned int all_cells = render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid();
			render_data.render_settings.regir_settings.shading.grid_cells_alive_count = all_cells;
			m_grid_cells_alive_count_staging_buffer.upload_data(&all_cells);

			updated = true;
		}

		// Representative cells data
		if (m_distance_to_center_buffer.size() != render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid())
		{
			m_distance_to_center_buffer.resize(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid());
			m_representative_points_buffer.resize(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid());
			m_representative_normals_buffer.resize(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid());
			m_representative_primitive_buffer.resize(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid());
			
			reset_representative_data();

			updated = true;
		}

		// Some precomputations that can be done here instead of being done on the GPU for each pixel
		{
			ReGIRGridSettings& grid = render_data.render_settings.regir_settings.grid;

			float3 grid_resolution_float = make_float3(grid.grid_resolution.x, grid.grid_resolution.y, grid.grid_resolution.z);
			float3 cell_size = grid.extents / grid_resolution_float;

			render_data.render_settings.regir_settings.m_cell_size = cell_size;
			render_data.render_settings.regir_settings.m_cell_diagonal_length = hippt::length(cell_size * 0.5f);
			render_data.render_settings.regir_settings.m_total_number_of_cells = render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid();
			render_data.render_settings.regir_settings.m_total_number_of_reservoirs = render_data.render_settings.regir_settings.get_total_number_of_reservoirs_ReGIR();
			render_data.render_settings.regir_settings.m_number_of_reservoirs_per_cell = render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell();
			render_data.render_settings.regir_settings.m_number_of_reservoirs_per_grid = render_data.render_settings.regir_settings.get_number_of_reservoirs_per_grid();
		}
	}
	else
	{
		if (m_grid_buffers.size() > 0)
		{
			m_grid_buffers.free();

			updated = true;
		}

		if (m_spatial_reuse_output_grid_buffer.size() > 0)
		{
			m_spatial_reuse_output_grid_buffer.free();

			updated = true;
		}

		if (m_grid_cells_alive_buffer.size() > 0)
		{
			m_grid_cells_alive_buffer.free();
			m_grid_cells_alive_staging_buffer.free();
			m_grid_cells_alive_count_staging_buffer.free();
			m_grid_cells_alive_list_buffer.free();

			updated = true;
		}

		if (m_distance_to_center_buffer.size() > 0)
		{
			m_distance_to_center_buffer.free();
			m_representative_points_buffer.free();
			m_representative_normals_buffer.free();
			m_representative_primitive_buffer.free();

			updated = true;
		}
	}

	return updated;
}

bool ReGIRRenderPass::launch(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return false;

	if (compiler_options.get_macro_value(GPUKernelCompilerOptions::REGIR_DO_DISPATCH_COMPACTION) == KERNEL_OPTION_TRUE)
	{
		m_grid_cells_alive_count_staging_buffer.download_data(m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer());
		render_data.render_settings.regir_settings.shading.grid_cells_alive_count = m_grid_cells_alive_count_staging_host_pinned_buffer.get_host_pinned_pointer()[0];
	}
	else
		render_data.render_settings.regir_settings.shading.grid_cells_alive_count = render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid();

	render_data.render_settings.regir_settings.temporal_reuse.current_grid_index = m_current_grid_index;

	if (render_data.render_settings.regir_settings.shading.grid_cells_alive_count > 0)
	{
		launch_grid_fill_temporal_reuse(render_data);
		launch_spatial_reuse(render_data);
	}

	return true;
}

void ReGIRRenderPass::launch_grid_fill_temporal_reuse(HIPRTRenderData& render_data)
{
	void* launch_args[] = { &render_data };

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
	unsigned int nb_threads = hippt::min(render_data.render_settings.regir_settings.shading.grid_cells_alive_count * reservoirs_per_cell, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));
	m_kernels[ReGIRRenderPass::REGIR_GRID_FILL_TEMPORAL_REUSE_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::launch_spatial_reuse(HIPRTRenderData& render_data)
{
	if (!render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
		return;

	void* launch_args[] = { &render_data };

	unsigned int reservoirs_per_cell = render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell();

	// Same reason for nb_threads here as explained in the GridFill kernel launch
	unsigned int nb_threads = hippt::min(render_data.render_settings.regir_settings.shading.grid_cells_alive_count * reservoirs_per_cell, (unsigned int)(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y));
	m_kernels[ReGIRRenderPass::REGIR_SPATIAL_REUSE_KERNEL_ID]->launch_asynchronous(64, 1, nb_threads, 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::launch_cell_liveness_copy_pass(HIPRTRenderData& render_data)
{
	if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
		return;

	unsigned int* staging_ptr = m_grid_cells_alive_staging_buffer.get_device_pointer();
	unsigned char* non_staging_ptr = m_grid_cells_alive_buffer.get_device_pointer();

	void* launch_args[] = { &staging_ptr, &non_staging_ptr, &render_data.render_settings.regir_settings };

	m_kernels[ReGIRRenderPass::REGIR_CELL_LIVENESS_COPY_KERNEL_ID]->launch_asynchronous(64, 1, render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid(), 1, launch_args, m_renderer->get_main_stream());
}

void ReGIRRenderPass::post_sample_update(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return;
	
	if (render_data.render_settings.regir_settings.temporal_reuse.do_temporal_reuse)
	{
		m_current_grid_index++;
		m_current_grid_index %= render_data.render_settings.regir_settings.temporal_reuse.temporal_history_length;
	}

	launch_cell_liveness_copy_pass(render_data);
}

void ReGIRRenderPass::update_render_data()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (is_render_pass_used())
	{
		render_data.render_settings.regir_settings.grid.grid_origin = m_renderer->get_scene_metadata().scene_bounding_box.mini;
		render_data.render_settings.regir_settings.grid.extents = m_renderer->get_scene_metadata().scene_bounding_box.get_extents();

		render_data.render_settings.regir_settings.grid_fill.grid_buffers = m_grid_buffers.to_device();
		if (render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
			render_data.render_settings.regir_settings.spatial_reuse.output_grid = m_spatial_reuse_output_grid_buffer.to_device();
		render_data.render_settings.regir_settings.representative.distance_to_center = m_distance_to_center_buffer.get_atomic_device_pointer();
		render_data.render_settings.regir_settings.representative.representative_normals = m_representative_normals_buffer.get_device_pointer();
		render_data.render_settings.regir_settings.representative.representative_points = m_representative_points_buffer.get_device_pointer();
		render_data.render_settings.regir_settings.representative.representative_primitive = m_representative_primitive_buffer.get_atomic_device_pointer();

		render_data.render_settings.regir_settings.shading.grid_cells_alive = m_grid_cells_alive_buffer.get_device_pointer();
		render_data.render_settings.regir_settings.shading.grid_cells_alive_staging = m_grid_cells_alive_staging_buffer.get_atomic_device_pointer();
		render_data.render_settings.regir_settings.shading.grid_cells_alive_count_staging = m_grid_cells_alive_count_staging_buffer.get_atomic_device_pointer();
		render_data.render_settings.regir_settings.shading.grid_cells_alive_list = m_grid_cells_alive_list_buffer.get_device_pointer();
	}
	else
	{
		render_data.render_settings.regir_settings.grid_fill.grid_buffers = ReGIRGridBufferSoADevice();
		render_data.render_settings.regir_settings.spatial_reuse.output_grid = ReGIRGridBufferSoADevice();

		render_data.render_settings.regir_settings.representative.distance_to_center = nullptr;
		render_data.render_settings.regir_settings.representative.representative_normals = nullptr;
		render_data.render_settings.regir_settings.representative.representative_points = nullptr;
		render_data.render_settings.regir_settings.representative.representative_primitive = nullptr;

		render_data.render_settings.regir_settings.shading.grid_cells_alive = nullptr;
		render_data.render_settings.regir_settings.shading.grid_cells_alive_staging = nullptr;
		render_data.render_settings.regir_settings.shading.grid_cells_alive_count_staging = nullptr;
		render_data.render_settings.regir_settings.shading.grid_cells_alive_list = nullptr;
	}
}

void ReGIRRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.render_settings.regir_settings.temporal_reuse.current_grid_index = 0;

	if (m_grid_cells_alive_buffer.size() > 0)
	{
		// Resetting the 'cell alive' buffers
		std::vector<unsigned char> init_data_alive(render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid(), 1);
		m_grid_cells_alive_buffer.upload_data(init_data_alive);

		// Resetting the count buffers
		unsigned int all_cells = render_data.render_settings.regir_settings.get_total_number_of_cells_per_grid();
		render_data.render_settings.regir_settings.shading.grid_cells_alive_count = all_cells;
		m_grid_cells_alive_count_staging_buffer.upload_data(&all_cells);
	}
}

void ReGIRRenderPass::reset_representative_data()
{
	if (m_distance_to_center_buffer.size() > 0)
	{
		{
			std::vector<float> distance_reset(m_distance_to_center_buffer.size(), ReGIRRepresentative::UNDEFINED_DISTANCE);
			m_distance_to_center_buffer.upload_data(distance_reset);
		}

		{
			std::vector<unsigned int> points_reset(m_representative_points_buffer.size(), ReGIRRepresentative::UNDEFINED_POINT);
			m_representative_points_buffer.upload_data(points_reset);
		}
		
		{
			std::vector<Octahedral24BitNormalPadded32b> normals_reset(m_representative_normals_buffer.size(), Octahedral24BitNormalPadded32b::pack_static(ReGIRRepresentative::UNDEFINED_NORMAL));
			m_representative_normals_buffer.upload_data(normals_reset);
		}

		{
			std::vector<int> primitive_reset(m_representative_primitive_buffer.size(), ReGIRRepresentative::UNDEFINED_PRIMITIVE);
			m_representative_primitive_buffer.upload_data(primitive_reset);
		}
	}
}

bool ReGIRRenderPass::is_render_pass_used() const
{
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY) == LSS_BASE_REGIR;
}

float ReGIRRenderPass::get_VRAM_usage() const
{
	return (
		m_grid_buffers.get_byte_size() + 
		m_spatial_reuse_output_grid_buffer.get_byte_size() + 

		m_distance_to_center_buffer.get_byte_size() + 
		m_representative_points_buffer.get_byte_size() + 
		m_representative_normals_buffer.get_byte_size() + 
		m_representative_primitive_buffer.get_byte_size() +

		m_grid_cells_alive_buffer.get_byte_size() + 
		m_grid_cells_alive_staging_buffer.get_byte_size() +
		m_grid_cells_alive_count_staging_buffer.get_byte_size() +
		m_grid_cells_alive_list_buffer.get_byte_size()) / 1000000.0f;
}

float ReGIRRenderPass::get_alive_cells_ratio() const
{
	std::vector<unsigned char> alive = m_grid_cells_alive_buffer.download_data();
	std::size_t count = 0;
	for (unsigned char cell : alive)
		if (cell == 1)
			count++;

	return count / static_cast<float>(m_grid_cells_alive_buffer.size());;
}
