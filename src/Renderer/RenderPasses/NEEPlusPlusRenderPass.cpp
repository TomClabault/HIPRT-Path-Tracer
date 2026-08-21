/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NEEPlusPlusRenderPass.h"
#include "Threads/ThreadFunctions.h"
#include "Threads/ThreadManager.h"
#include "UI/RenderWindow.h"

const std::string NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE = "NEE++ Pre-population";

const std::string NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME = "NEE++ Render Pass";

const std::unordered_map<std::string, std::string> NEEPlusPlusRenderPass::KERNEL_FUNCTION_NAMES = {
	{ NEE_PLUS_PLUS_PRE_POPULATE, "NEEPlusPlus_Grid_Prepopulate" },
};

const std::unordered_map<std::string, std::string> NEEPlusPlusRenderPass::KERNEL_FILES = {
	{ NEE_PLUS_PLUS_PRE_POPULATE, DEVICE_KERNELS_DIRECTORY "/NEE++/GridPrepopulate.h" },
};

NEEPlusPlusRenderPass::NEEPlusPlusRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME, renderer, options)
{
	m_render_data_host_pinned.resize_host_pinned_mem(1);

	std::unordered_set<std::string> options_not_synchronized = GPURenderer::KERNEL_OPTIONS_NOT_SYNCHRONIZED;
	options_not_synchronized.insert(GPUKernelCompilerOptions::BSDF_OVERRIDE);

	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE);
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->set_kernel_file_path(
		NEEPlusPlusRenderPass::KERNEL_FILES.at(NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE));
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->set_kernel_function_name(
		NEEPlusPlusRenderPass::KERNEL_FUNCTION_NAMES.at(NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE));
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::BSDF_OVERRIDE,
																									   BSDF_LAMBERTIAN);
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->synchronize_options_with(m_compiler_options, options_not_synchronized);

	m_nee_plus_plus_storage.set_nee_plus_plus_render_pass(this);
}

bool NEEPlusPlusRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
														 const std::vector<hiprtFuncNameSet>& func_name_sets,
														 bool silent,
														 bool use_cache)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	bool nee_plus_plus__grid_populate_compiled = m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->has_been_compiled();
	if (!nee_plus_plus__grid_populate_compiled)
		m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return !nee_plus_plus__grid_populate_compiled;
}

bool NEEPlusPlusRenderPass::pre_frame_render_update(float delta_time)
{
	if (!is_render_pass_used(*m_compiler_options))
		return m_nee_plus_plus_storage.free();

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	return m_nee_plus_plus_storage.pre_frame_render_update(render_data, m_render_window->is_interacting());
}

void NEEPlusPlusRenderPass::update_render_data()
{
	m_nee_plus_plus_storage.update_render_data(m_renderer->get_render_data(), *m_compiler_options);
}

bool NEEPlusPlusRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!is_render_pass_used(compiler_options))
		return false;

	if (render_data.render_settings.sample_number == 0 && !m_render_window->is_interacting() && render_data.render_settings.accumulate)
	{
		m_render_window->set_ImGui_status_text("NEE++ Prepopulation pass...");
		launch_grid_pre_population(render_data);
		m_render_window->clear_ImGui_status_text();
	}

	if (m_nee_plus_plus_storage.try_resize(render_data, compiler_options, m_max_vram_usage_megabytes))
		update_render_data();

	return true;
}

void NEEPlusPlusRenderPass::launch_grid_pre_population(HIPRTRenderData& render_data)
{
	bool has_rehashed = false;

	do
	{
		// Just making sure that this is not set to false
		render_data.nee_plus_plus.m_update_visibility_map = true;

		HIPRTRenderData* host_pinned_render_data = m_render_data_host_pinned.get_host_pinned_pointer();
		*host_pinned_render_data				 = render_data;

		m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->upload_to_module_global("NEE_PLUS_PLUS_RENDER_DATA", host_pinned_render_data,
																							  sizeof(HIPRTRenderData), m_renderer->get_main_stream());

		m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->launch_asynchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight, m_renderer->m_render_resolution.x / NEEPlusPlus_GridPrepoluationResolutionDownscale,
			m_renderer->m_render_resolution.y / NEEPlusPlus_GridPrepoluationResolutionDownscale, nullptr, m_renderer->get_main_stream());

		has_rehashed = m_nee_plus_plus_storage.try_resize(render_data, *m_compiler_options, m_max_vram_usage_megabytes);
		if (has_rehashed)
			update_render_data();

	} while (has_rehashed);
}

void NEEPlusPlusRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

float NEEPlusPlusRenderPass::get_full_frame_time()
{
	float sum = 0.0f;

	for (auto& name_to_kernel : get_all_kernels())
	{
		if (name_to_kernel.first == NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE)
			// Not counting the pre population pass in the frame time since this is only
			// done on the very first frame of the render, not really reprensentative of the
			// true frame time
			continue;

		sum += name_to_kernel.second->get_last_execution_time();
	}

	return sum;
}

void NEEPlusPlusRenderPass::reset(bool reset_by_camera_movement)
{
	if (!is_render_pass_used(*m_compiler_options))
		return;

	m_nee_plus_plus_storage.reset();
}

bool NEEPlusPlusRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	// Only active if we're not using ReSTIR GI because if we are using ReSTIR, the path tracing is done in
	// the initial candidates kernel
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE;
}

NEEPlusPlusHashGridStorage& NEEPlusPlusRenderPass::get_nee_plus_plus_storage()
{
	return m_nee_plus_plus_storage;
}

float& NEEPlusPlusRenderPass::get_max_vram_usage()
{
	return m_max_vram_usage_megabytes;
}

std::size_t NEEPlusPlusRenderPass::get_vram_usage_bytes() const
{
	return m_nee_plus_plus_storage.get_byte_size();
}

float NEEPlusPlusRenderPass::get_load_factor() const
{
	return m_nee_plus_plus_storage.get_load_factor();
}
