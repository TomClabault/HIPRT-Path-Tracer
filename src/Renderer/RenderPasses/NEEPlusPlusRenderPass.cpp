/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NEEPlusPlusRenderPass.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"
 
const std::string NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME = "NEE++ Render Pass";

NEEPlusPlusRenderPass::NEEPlusPlusRenderPass() : NEEPlusPlusRenderPass(nullptr) {}
NEEPlusPlusRenderPass::NEEPlusPlusRenderPass(GPURenderer* renderer) : NEEPlusPlusRenderPass(renderer, NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME) {}
NEEPlusPlusRenderPass::NEEPlusPlusRenderPass(GPURenderer* renderer, const std::string& name) : RenderPass(renderer, name) 
{
	m_nee_plus_plus_storage.set_nee_plus_plus_render_pass(this);
}

bool NEEPlusPlusRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	return false;
}
 
bool NEEPlusPlusRenderPass::pre_render_update(float delta_time)
{
	if (!is_render_pass_used())
		return m_nee_plus_plus_storage.free();

    HIPRTRenderData& render_data = m_renderer->get_render_data();

	return m_nee_plus_plus_storage.pre_render_update(render_data);
}
 
void NEEPlusPlusRenderPass::update_render_data()
{
	m_nee_plus_plus_storage.update_render_data();
}

bool NEEPlusPlusRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) 
{
	if (!m_render_pass_used_this_frame)
		return false;

	if (render_data.render_settings.sample_number == 0)
		launch_grid_pre_population(render_data);

	return true;
}

void NEEPlusPlusRenderPass::launch_grid_pre_population(HIPRTRenderData& render_data)
{

}

void NEEPlusPlusRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return;
	
	m_nee_plus_plus_storage.post_sample_update_async(render_data);
}
 
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

std::size_t NEEPlusPlusRenderPass::get_vram_usage_bytes() const
{
	return m_nee_plus_plus_storage.get_byte_size();
}
