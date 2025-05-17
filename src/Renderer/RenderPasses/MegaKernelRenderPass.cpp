/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"

const std::string MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME = "Megakernel Render Pass";
const std::string MegaKernelRenderPass::MEGAKERNEL_KERNEL = "Megakernel (1 SPP)";

MegaKernelRenderPass::MegaKernelRenderPass() : MegaKernelRenderPass(nullptr) {}
MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer) : MegaKernelRenderPass(renderer, MegaKernelRenderPass::MEGAKERNEL_RENDER_PASS_NAME) {}
MegaKernelRenderPass::MegaKernelRenderPass(GPURenderer* renderer, const std::string& name) : RenderPass(renderer, name) 
{
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL] = std::make_shared<GPUKernel>();
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/Megakernel.h");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->set_kernel_function_name("MegaKernel");
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->synchronize_options_with(m_renderer->get_global_compiler_options(), GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, KERNEL_OPTION_TRUE);
	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, 8);
}

void MegaKernelRenderPass::resize(unsigned int new_width, unsigned int new_height)
{
	m_render_resolution.x = new_width;
	m_render_resolution.y = new_height;
}

bool MegaKernelRenderPass::pre_render_update(float delta_time)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used())
		return false;

	// Resetting this flag as this is a new frame
	render_data.render_settings.do_update_status_buffers = false;

	if (!render_data.render_settings.accumulate)
		render_data.render_settings.sample_number = 0;

	return false;
}

bool MegaKernelRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return false;
		
	render_data.random_number = m_renderer->get_rng_generator().xorshift32();
	
	void* launch_args[] = { &render_data };

	m_kernels[MegaKernelRenderPass::MEGAKERNEL_KERNEL]->launch_asynchronous(KernelBlockWidthHeight, KernelBlockWidthHeight, m_render_resolution.x, m_render_resolution.y, launch_args, m_renderer->get_main_stream());

	if( m_renderer->get_ReGIR_render_pass()->is_render_pass_used())
	{
		std::cout << "Count after (auto) megakernel: " << m_renderer->get_ReGIR_render_pass()->m_hash_grid_storage.get_hash_cell_data_soa().m_grid_cells_alive_count.download_data()[0] << std::endl;
		unsigned int manual_count = 0;
		std::vector<unsigned int> cell_alive_list_after = m_renderer->get_ReGIR_render_pass()->m_hash_grid_storage.get_hash_cell_data_soa().m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>().download_data();
		for (unsigned int cell : cell_alive_list_after)
			if (cell > 0)
				manual_count++;

		std::vector<unsigned int> cell_hashes = m_renderer->get_ReGIR_render_pass()->m_hash_grid_storage.get_hash_cell_data_soa().m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELL_HASH_KEYS>().download_data();
		std::vector<unsigned int> cell_alive_indices = m_renderer->get_ReGIR_render_pass()->m_hash_grid_storage.get_hash_cell_data_soa().m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>().download_data();
		std::unordered_set<unsigned int> cell_hashes_set;
	}

	return true;
}

void MegaKernelRenderPass::reset(bool reset_by_camera_movement)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (!is_render_pass_used())
		return;

	if (render_data.render_settings.accumulate)
		if (m_renderer->get_application_settings()->auto_sample_per_frame)
			render_data.render_settings.samples_per_frame = 1;

	render_data.render_settings.denoiser_AOV_accumulation_counter = 0;

	render_data.render_settings.sample_number = 0;
}

bool MegaKernelRenderPass::is_render_pass_used() const
{
	// Only active if we're not using ReSTIR GI because if we are using ReSTIR, the path tracing is done in
	// the initial candidates kernel
	return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) != PSS_RESTIR_GI;
}
