/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReSTIRPGRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

const std::string ReSTIRPGRenderPass::RESTIR_PG_RENDER_PASS_NAME				   = "ReSTIR PG Render Pass";
const std::string ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL				   = "ReSTIR PG Splatting";
const std::string ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL					   = "ReSTIR PG Fitting";
const std::string ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL = "ReSTIR PG Reset statistics";
const std::string ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID					   = "ReSTIR PG Reset hash grid";
const std::string ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL		   = "ReSTIR PG Reset distributions";

ReSTIRPGRenderPass::ReSTIRPGRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: ReSTIRPGRenderPass(ReSTIRPGRenderPass::RESTIR_PG_RENDER_PASS_NAME, renderer, options)
{
}

ReSTIRPGRenderPass::ReSTIRPGRenderPass(const std::string& name, GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(name, renderer, options)
{
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL);
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PG/Splatting.h");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL]->set_kernel_function_name("ReSTIR_PG_Splatting");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL]->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);

	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL);
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PG/Fitting.h");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL]->set_kernel_function_name("ReSTIR_PG_Fitting");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL]->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);

	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL);
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																									  "/ReSTIR/PG/ResetSufficientStatistics.h");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL]->set_kernel_function_name("ReSTIR_PG_ResetSufficientStatistics");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL]->synchronize_options_with(m_compiler_options,
																										  GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);

	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID);
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PG/ResetHashGrid.h");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID]->set_kernel_function_name("ReSTIR_PG_ResetHashGrid");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID]->synchronize_options_with(m_compiler_options, GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);

	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL);
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/ReSTIR/PG/ResetDistributions.h");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL]->set_kernel_function_name("ReSTIR_PG_ResetDistributions");
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL]->synchronize_options_with(m_compiler_options,
																								  GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
}

bool ReSTIRPGRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
													  const std::vector<hiprtFuncNameSet>& func_name_sets,
													  bool silent,
													  bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool updated = false;

	if (!m_kernels[ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	if (!m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL]->has_been_compiled())
	{
		updated = true;
		m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);
	}

	return updated;
}

void ReSTIRPGRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	if (!is_render_pass_used())
		return;

	unsigned int nb_bounces = m_renderer->get_render_data().render_settings.nb_bounces;

	m_splatting_samples_buffer.resize(new_width * new_height * nb_bounces);
	m_already_splatted_samples_buffer.resize(new_width * new_height);
}

bool ReSTIRPGRenderPass::pre_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	bool updated = false;

	if (!is_render_pass_used())
	{
		if (m_splatting_samples_buffer.size() != 0)
		{
			m_splatting_samples_buffer.free();
			m_already_splatted_samples_buffer.free();
		}

		if (m_hash_grid_distributions_soa_buffer.get_last_resize_component_count() != 0)
		{
			m_hash_grid_distributions_soa_buffer.free();
			m_hash_grid_checksums_buffer.free();
			m_grid_cell_alive_buffer.free();
			m_grid_cell_alive_count_buffer.free();
			m_grid_cell_alive_list_buffer.free();

			m_hash_grid_distributions_sufficient_statistics_soa_buffer.free();
		}

		updated = true;
	}
	else
	{
		unsigned int component_count =
			m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT);

		if (m_splatting_samples_buffer.size() !=
			render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y * render_data.render_settings.nb_bounces)
		{
			m_splatting_samples_buffer.resize(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y *
											  render_data.render_settings.nb_bounces);
			m_already_splatted_samples_buffer.resize(render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y);

			updated = true;
		}

		if (m_hash_grid_checksums_buffer.size() != HASH_GRID_INITIAL_CELL_COUNT)
		{
			m_hash_grid_checksums_buffer.resize(HASH_GRID_INITIAL_CELL_COUNT);
			m_hash_grid_checksums_buffer.memset_whole_buffer(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
			m_grid_cell_alive_buffer.resize(HASH_GRID_INITIAL_CELL_COUNT);
			m_grid_cell_alive_buffer.memset_whole_buffer(0);
			m_grid_cell_alive_count_buffer.resize(1);
			m_grid_cell_alive_count_buffer.memset_whole_buffer(0);
			m_grid_cell_alive_list_buffer.resize(HASH_GRID_INITIAL_CELL_COUNT);

			updated = true;
		}

		if (m_hash_grid_distributions_sufficient_statistics_soa_buffer.get_last_resize_component_count() != component_count)
		{
			m_hash_grid_distributions_sufficient_statistics_soa_buffer.resize(HASH_GRID_INITIAL_CELL_COUNT, component_count);

			updated = true;
		}

		if (m_hash_grid_distributions_soa_buffer.get_total_element_count() != HASH_GRID_INITIAL_CELL_COUNT * component_count)
		{
			m_hash_grid_distributions_soa_buffer.resize(HASH_GRID_INITIAL_CELL_COUNT, component_count);

			updated = true;
		}

		// Resetting everything
		if (render_data.render_settings.sample_number == 0)
		{
			if (updated)
				update_render_data();

			void* launch_args[] = { &render_data };
			m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_HASH_GRID]->launch_asynchronous(
				256, 1, m_hash_grid_distributions_soa_buffer.get_last_resize_number_of_cells(), 1, launch_args, m_renderer->get_main_stream());
			std::vector<unsigned int> checksums = m_hash_grid_checksums_buffer.download_data();
			for (unsigned int checksum : checksums)
				if (checksum != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				{
					std::cerr
						<< "Error: During the first sample of the first frame, the hash grid checksums buffer should be initialized to "
						<< HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX
						<< " but it is not. This may indicate a problem with the GPU memory management (buffers not being properly cleared after resizing "
						   "for example) or a problem with the kernel that resets the hash grid."
						<< std::endl;

					break;
				}

			m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_DISTRIBUTIONS_KERNEL]->launch_asynchronous(
				256, 1,
				m_hash_grid_distributions_soa_buffer.get_last_resize_number_of_cells() * m_hash_grid_distributions_soa_buffer.get_last_resize_component_count(),
				1, launch_args, m_renderer->get_main_stream());

			m_grid_cell_alive_buffer.memset_whole_buffer(0);
		}
	}

	return updated;
}

bool ReSTIRPGRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return false;

	{
		std::vector<ReSTIRPGSplattingSample> splatting_samples_CPU = m_splatting_samples_buffer.download_data();

		unsigned int DEBUGx			  = 142;
		unsigned int DEBUGy			  = 245;
		unsigned int DEBUGpixel_index = DEBUGx + DEBUGy * render_data.render_settings.render_resolution.x;

		std::cerr << "Splatting sample at " << DEBUGx << ", " << DEBUGy << " (pixel index " << DEBUGpixel_index << "): "
				  << "position = (" << splatting_samples_CPU[DEBUGpixel_index].position.x << ", " << splatting_samples_CPU[DEBUGpixel_index].position.y << ", "
				  << splatting_samples_CPU[DEBUGpixel_index].position.z << ")"
				  << ", normal = (" << splatting_samples_CPU[DEBUGpixel_index].normal.x << ", " << splatting_samples_CPU[DEBUGpixel_index].normal.y << ", "
				  << splatting_samples_CPU[DEBUGpixel_index].normal.z << ")"
				  << ", incident_direction = (" << splatting_samples_CPU[DEBUGpixel_index].incident_direction.x << ", "
				  << splatting_samples_CPU[DEBUGpixel_index].incident_direction.y << ", " << splatting_samples_CPU[DEBUGpixel_index].incident_direction.z
				  << std::endl;
	}

	m_already_splatted_samples_buffer.memset_whole_buffer(0);

	void* launch_args[] = { &render_data };
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_SPLATTING_KERNEL]->launch_asynchronous(
		KernelBlockWidthHeight, KernelBlockWidthHeight, render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y,
		launch_args, m_renderer->get_main_stream());

	ReSTIRPGSplattingSample empty_sample;
	m_splatting_samples_buffer.memset_whole_buffer(empty_sample);

	{
		std::vector<float> sufficient_statistics_direction_x_CPU =
			m_hash_grid_distributions_sufficient_statistics_soa_buffer.m_sufficient_statistics_data
				.template get_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_X>()
				.download_data();
		std::vector<float> sufficient_statistics_direction_y_CPU =
			m_hash_grid_distributions_sufficient_statistics_soa_buffer.m_sufficient_statistics_data
				.template get_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_Y>()
				.download_data();
		std::vector<float> sufficient_statistics_direction_z_CPU =
			m_hash_grid_distributions_sufficient_statistics_soa_buffer.m_sufficient_statistics_data
				.template get_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_DIRECTIONS_SUMS_Z>()
				.download_data();
		std::vector<float> sufficient_statistics_responsibility_weights_CPU =
			m_hash_grid_distributions_sufficient_statistics_soa_buffer.m_sufficient_statistics_data
				.template get_buffer<ReSTIRPGSufficientStatisticsSoAHostBuffers::RESTIR_PG_RESPONSIBILITY_WEIGHTS_SUM>()
				.download_data();

		std::vector<unsigned long long int> debug1_CPU = OrochiBuffer<unsigned long long int>::download_data(
			reinterpret_cast<unsigned long long int*>(render_data.render_settings.DEBUG_BUFFER_ULL_1), 1024);
		unsigned int DEBUGcell_index = 0; // debug1_CPU[0];

		std::cerr << "Sufficient statistics for cell " << DEBUGcell_index << ": direction sum = (" << sufficient_statistics_direction_x_CPU[DEBUGcell_index]
				  << ", " << sufficient_statistics_direction_y_CPU[DEBUGcell_index] << ", " << sufficient_statistics_direction_z_CPU[DEBUGcell_index] << ")"
				  << ", responsibility weights sum = " << sufficient_statistics_responsibility_weights_CPU[DEBUGcell_index] << std::endl;
	}

	unsigned int grid_cell_alive_count = m_grid_cell_alive_count_buffer.download_data()[0];
	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_FITTING_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, 1, grid_cell_alive_count, 1, launch_args,
																				 m_renderer->get_main_stream());

	m_kernels[ReSTIRPGRenderPass::RESTIR_PG_RESET_SUFFICIENT_STATISTICS_KERNEL]->launch_asynchronous(
		256, 1,
		render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells *
			compiler_options.get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT),
		1, launch_args, m_renderer->get_main_stream());

	return true;
}

void ReSTIRPGRenderPass::update_render_data()
{
	if (!is_render_pass_used())
		return;

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.render_settings.restir_pg_settings.splatting_samples		= m_splatting_samples_buffer.get_device_pointer();
	render_data.render_settings.restir_pg_settings.already_splatted_samples = m_already_splatted_samples_buffer.get_atomic_device_pointer();

	render_data.render_settings.restir_pg_settings.hash_grid_distributions_soa = m_hash_grid_distributions_soa_buffer.to_device();
	render_data.render_settings.restir_pg_settings.hash_grid_checksums		   = m_hash_grid_checksums_buffer.get_atomic_device_pointer();
	render_data.render_settings.restir_pg_settings.grid_cell_alive			   = m_grid_cell_alive_buffer.get_atomic_device_pointer();
	render_data.render_settings.restir_pg_settings.grid_cell_alive_count	   = m_grid_cell_alive_count_buffer.get_atomic_device_pointer();
	render_data.render_settings.restir_pg_settings.grid_cell_alive_list		   = m_grid_cell_alive_list_buffer.get_device_pointer();

	render_data.render_settings.restir_pg_settings.hash_grid_distributions_sufficient_statistics_soa =
		m_hash_grid_distributions_sufficient_statistics_soa_buffer.to_device();

	render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells = m_hash_grid_distributions_soa_buffer.get_last_resize_number_of_cells();
}

bool ReSTIRPGRenderPass::is_render_pass_used() const
{
	bool restir_path_sampling_used = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PSS_RESTIR_GI;
	bool using_restir_pg		   = m_compiler_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_ENABLE) == KERNEL_OPTION_TRUE;
	bool bounces				   = m_renderer->get_render_data().render_settings.nb_bounces > 0;

	return restir_path_sampling_used && using_restir_pg && bounces;
}
