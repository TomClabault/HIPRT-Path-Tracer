/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NEEPlusPlusRenderPass.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"
#include "UI/RenderWindow.h"
 
const std::string NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE = "NEE++ Pre-population";

const std::string NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME = "NEE++ Render Pass";

const std::unordered_map<std::string, std::string> NEEPlusPlusRenderPass::KERNEL_FUNCTION_NAMES =
{
	{ NEE_PLUS_PLUS_PRE_POPULATE, "NEEPlusPlus_Grid_Prepopulate" },
};

const std::unordered_map<std::string, std::string> NEEPlusPlusRenderPass::KERNEL_FILES =
{
	{ NEE_PLUS_PLUS_PRE_POPULATE, DEVICE_KERNELS_DIRECTORY "/NEE++/GridPrepopulate.h" },
};

NEEPlusPlusRenderPass::NEEPlusPlusRenderPass() : NEEPlusPlusRenderPass(nullptr) {}
NEEPlusPlusRenderPass::NEEPlusPlusRenderPass(GPURenderer* renderer) : NEEPlusPlusRenderPass(renderer, NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME) {}
NEEPlusPlusRenderPass::NEEPlusPlusRenderPass(GPURenderer* renderer, const std::string& name) : RenderPass(renderer, name) 
{
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE] = std::make_shared<GPUKernel>();
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->set_kernel_file_path(NEEPlusPlusRenderPass::KERNEL_FILES.at(NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE));
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->set_kernel_function_name(NEEPlusPlusRenderPass::KERNEL_FUNCTION_NAMES.at(NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE));
	m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->get_kernel_options().set_macro_value(GPUKernelCompilerOptions::BSDF_OVERRIDE, BSDF_LAMBERTIAN);

	m_nee_plus_plus_storage.set_nee_plus_plus_render_pass(this);
}

bool NEEPlusPlusRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool nee_plus_plus__grid_populate_compiled = m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->has_been_compiled();
	if (!nee_plus_plus__grid_populate_compiled)
		m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return !nee_plus_plus__grid_populate_compiled;
}
 
bool NEEPlusPlusRenderPass::pre_render_update(float delta_time)
{
	if (!is_render_pass_used())
		return m_nee_plus_plus_storage.free();

    HIPRTRenderData& render_data = m_renderer->get_render_data();

	return m_nee_plus_plus_storage.pre_render_update(render_data, m_render_window->is_interacting());
}
 
void NEEPlusPlusRenderPass::update_render_data()
{
	m_nee_plus_plus_storage.update_render_data(m_renderer->get_render_data());
}

bool NEEPlusPlusRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) 
{
	if (!m_render_pass_used_this_frame)
		return false;

	if (render_data.render_settings.sample_number == 0 && !m_render_window->is_interacting())
	{
		m_render_window->set_ImGui_status_text("NEE++ Prepopulation pass...");
		launch_grid_pre_population(render_data);
		m_render_window->clear_ImGui_status_text();
	}

	if (m_nee_plus_plus_storage.try_resize(render_data, m_max_vram_usage_megabytes))
		update_render_data();

	return true;
}

void NEEPlusPlusRenderPass::launch_grid_pre_population(HIPRTRenderData& render_data)
{
	bool has_rehashed = false;

	do
	{
		void* launch_args[] = { &render_data };

		m_kernels[NEEPlusPlusRenderPass::NEE_PLUS_PLUS_PRE_POPULATE]->launch_asynchronous(
			KernelBlockWidthHeight, KernelBlockWidthHeight,
			m_renderer->m_render_resolution.x / NEEPlusPlus_GridPrepoluationResolutionDownscale, m_renderer->m_render_resolution.y / NEEPlusPlus_GridPrepoluationResolutionDownscale,
			launch_args, m_renderer->get_main_stream());

		has_rehashed = m_nee_plus_plus_storage.try_resize(render_data, m_max_vram_usage_megabytes);
		if (has_rehashed)
			update_render_data();

	} while (has_rehashed);
}

void NEEPlusPlusRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}
 
void NEEPlusPlusRenderPass::reset(bool reset_by_camera_movement)
{
     if (!is_render_pass_used())
         return;

	m_nee_plus_plus_storage.reset();
}
 
bool NEEPlusPlusRenderPass::is_render_pass_used() const
{
     // Only active if we're not using ReSTIR GI because if we are using ReSTIR, the path tracing is done in
     // the initial candidates kernel
     return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE;
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
