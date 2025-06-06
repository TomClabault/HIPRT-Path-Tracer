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
}

bool NEEPlusPlusRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	return false;
}
 
bool NEEPlusPlusRenderPass::pre_render_update(float delta_time)
{
    HIPRTRenderData& render_data = m_renderer->get_render_data();

    bool updated = false;

    if (!is_render_pass_used())
	{
		// Not using NEE++, we just need to free the buffers if they weren't already

		if (m_nee_plus_plus.total_num_rays.size() != 0)
		{
			m_nee_plus_plus.total_num_rays.free();
			m_nee_plus_plus.total_unoccluded_rays.free();
			m_nee_plus_plus.checksum_buffer.free();

			m_nee_plus_plus.total_shadow_ray_queries.free();
			m_nee_plus_plus.shadow_rays_actually_traced.free();
		}

		return true;
	}

	// Allocating / deallocating buffers
	if (m_nee_plus_plus.total_num_rays.size() != 300000001)
	{
		m_nee_plus_plus.total_num_rays.resize(300000001);
		m_nee_plus_plus.total_unoccluded_rays.resize(300000001);
		m_nee_plus_plus.checksum_buffer.resize(300000001);
		
		m_nee_plus_plus.shadow_rays_actually_traced.resize(1);
		m_nee_plus_plus.total_shadow_ray_queries.resize(1);

		updated = true;
	}

	// Clearing the visibility map if this has been asked by the user
	if (render_data.nee_plus_plus.m_reset_visibility_map)
	{
		// Clearing the visibility map by memseting everything to 0
		m_nee_plus_plus.total_num_rays.memset_whole_buffer(0);
		m_nee_plus_plus.total_unoccluded_rays.memset_whole_buffer(0);
		m_nee_plus_plus.checksum_buffer.memset_whole_buffer(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

		m_nee_plus_plus.total_shadow_ray_queries.memset_whole_buffer(0);
		m_nee_plus_plus.shadow_rays_actually_traced.memset_whole_buffer(0);
	}

	if (render_data.render_settings.sample_number > render_data.nee_plus_plus.m_stop_update_samples)
		// Past a certain number of samples, there isn't really a point to keep updating, the visibility map
		// is probably converged enough that it doesn't make a difference anymore
		render_data.nee_plus_plus.m_update_visibility_map = false;

    return updated;
}
 
void NEEPlusPlusRenderPass::update_render_data()
{
    HIPRTRenderData& render_data = m_renderer->get_render_data();

    if (is_render_pass_used())
    {
	    render_data.nee_plus_plus.m_entries_buffer.total_num_rays = m_nee_plus_plus.total_num_rays.get_atomic_device_pointer();
	    render_data.nee_plus_plus.m_entries_buffer.total_unoccluded_rays = m_nee_plus_plus.total_unoccluded_rays.get_atomic_device_pointer();
	    render_data.nee_plus_plus.m_entries_buffer.checksum_buffer = m_nee_plus_plus.checksum_buffer.get_atomic_device_pointer();
		render_data.nee_plus_plus.m_total_number_of_cells = 300000001;

	    render_data.nee_plus_plus.shadow_rays_actually_traced = m_nee_plus_plus.shadow_rays_actually_traced.get_atomic_device_pointer();
    	render_data.nee_plus_plus.total_shadow_ray_queries = m_nee_plus_plus.total_shadow_ray_queries.get_atomic_device_pointer();
    }
    else
    {
		render_data.nee_plus_plus.m_entries_buffer.total_num_rays = nullptr;
	    render_data.nee_plus_plus.m_entries_buffer.total_unoccluded_rays = nullptr;
	    render_data.nee_plus_plus.m_entries_buffer.checksum_buffer = nullptr;

		render_data.nee_plus_plus.shadow_rays_actually_traced = nullptr;
		render_data.nee_plus_plus.total_shadow_ray_queries = nullptr;
	}
}

bool NEEPlusPlusRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) { return m_render_pass_used_this_frame; }

void NEEPlusPlusRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	if (!m_render_pass_used_this_frame)
		return;
		
	OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.total_shadow_ray_queries_cpu, m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
	OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.shadow_rays_actually_traced_cpu, m_nee_plus_plus.shadow_rays_actually_traced.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));

	if (render_data.render_settings.sample_number % 50 == 0) 
	{
		std::size_t counter = 0;
		auto vec = m_nee_plus_plus.checksum_buffer.download_data();
		for (auto check : vec)
			if (check != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
				counter++;

		printf("NEE++: %zu cells have been updated this frame.\n", counter);
	}
}
 
void NEEPlusPlusRenderPass::reset(bool reset_by_camera_movement)
{
     if (!is_render_pass_used())
         return;

    HIPRTRenderData& render_data = m_renderer->get_render_data();

    render_data.nee_plus_plus.m_reset_visibility_map = true;
	render_data.nee_plus_plus.m_update_visibility_map = true;

	// Resetting the counters
	if (m_nee_plus_plus.total_shadow_ray_queries.is_allocated())
	{
		m_nee_plus_plus.total_shadow_ray_queries.memset_whole_buffer(1);
		m_nee_plus_plus.shadow_rays_actually_traced.memset_whole_buffer(1);
	}
	m_nee_plus_plus.total_shadow_ray_queries_cpu = 1;
	m_nee_plus_plus.shadow_rays_actually_traced_cpu = 1;
}
 
bool NEEPlusPlusRenderPass::is_render_pass_used() const
{
     // Only active if we're not using ReSTIR GI because if we are using ReSTIR, the path tracing is done in
     // the initial candidates kernel
     return m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE;
}
 
NEEPlusPlusGPUData& NEEPlusPlusRenderPass::get_nee_plus_plus_data()
{
    return m_nee_plus_plus;
}

std::size_t NEEPlusPlusRenderPass::get_vram_usage_bytes() const
{
	return m_nee_plus_plus.total_unoccluded_rays.get_byte_size() + m_nee_plus_plus.total_num_rays.get_byte_size();
}
