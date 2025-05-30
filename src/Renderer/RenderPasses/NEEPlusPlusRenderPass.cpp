/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NEEPlusPlusRenderPass.h"
#include "Threads/ThreadManager.h"
#include "Threads/ThreadFunctions.h"
 
const std::string NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME = "NEE++ Render Pass";
const std::string NEEPlusPlusRenderPass::FINALIZE_ACCUMULATION_KERNEL_ID = "NEE++ Finalize accumulation";

NEEPlusPlusRenderPass::NEEPlusPlusRenderPass() : NEEPlusPlusRenderPass(nullptr) {}
NEEPlusPlusRenderPass::NEEPlusPlusRenderPass(GPURenderer* renderer) : NEEPlusPlusRenderPass(renderer, NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME) {}
NEEPlusPlusRenderPass::NEEPlusPlusRenderPass(GPURenderer* renderer, const std::string& name) : RenderPass(renderer, name) 
{
    m_kernels[NEEPlusPlusRenderPass::FINALIZE_ACCUMULATION_KERNEL_ID] = std::make_shared<GPUKernel>();
	m_kernels[NEEPlusPlusRenderPass::FINALIZE_ACCUMULATION_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY "/NEE++/NEEPlusPlusFinalizeAccumulation.h");
    m_kernels[NEEPlusPlusRenderPass::FINALIZE_ACCUMULATION_KERNEL_ID]->set_kernel_function_name("NEEPlusPlusFinalizeAccumulation");
}

bool NEEPlusPlusRenderPass::pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets, bool silent, bool use_cache)
{
	if (!is_render_pass_used())
		return false;

	bool finalize_accumulation_kernel_compiled = m_kernels[NEEPlusPlusRenderPass::FINALIZE_ACCUMULATION_KERNEL_ID]->has_been_compiled();
	if (!finalize_accumulation_kernel_compiled)
		m_kernels[NEEPlusPlusRenderPass::FINALIZE_ACCUMULATION_KERNEL_ID]->compile(hiprt_orochi_ctx, func_name_sets, use_cache, silent);

	return !finalize_accumulation_kernel_compiled;
}
 
bool NEEPlusPlusRenderPass::pre_render_update(float delta_time)
{
    HIPRTRenderData& render_data = m_renderer->get_render_data();

    bool updated = false;

    if (!is_render_pass_used())
	{
		// Not using NEE++, we just need to free the buffers if they weren't already

		if (m_nee_plus_plus.packed_buffer.size() != 0)
		{
			m_nee_plus_plus.packed_buffer.free();
			m_nee_plus_plus.total_shadow_ray_queries.free();
			m_nee_plus_plus.shadow_rays_actually_traced.free();
		}

		return true;
	}

	float3 min_grid_extent_with_envmap, max_grid_extent_with_envmap;
	m_nee_plus_plus.get_grid_extents(m_nee_plus_plus.grid_dimensions_no_envmap, min_grid_extent_with_envmap, max_grid_extent_with_envmap);

	// Adding (2, 2, 2) for envmap NEE++
	render_data.nee_plus_plus.grid_dimensions = m_nee_plus_plus.grid_dimensions_no_envmap + make_int3(2, 2, 2);
	render_data.nee_plus_plus.grid_min_point = min_grid_extent_with_envmap;
 	render_data.nee_plus_plus.grid_max_point = max_grid_extent_with_envmap;

	// Allocating / deallocating buffers
	unsigned int matrix_element_count = m_nee_plus_plus.get_visibility_matrix_element_count(render_data.nee_plus_plus.grid_dimensions);
	if (m_nee_plus_plus.packed_buffer.size() != matrix_element_count)
	{
		m_nee_plus_plus.packed_buffer.resize(matrix_element_count);
		m_nee_plus_plus.shadow_rays_actually_traced.resize(1);
		m_nee_plus_plus.total_shadow_ray_queries.resize(1);

		updated = true;
	}

	// Clearing the visibility map if this has been asked by the user
	if (render_data.nee_plus_plus.reset_visibility_map)
	{
		// Clearing the visibility map by memseting everything to 0
		m_nee_plus_plus.packed_buffer.memset_whole_buffer(0);
		m_nee_plus_plus.total_shadow_ray_queries.memset_whole_buffer(1);
		m_nee_plus_plus.shadow_rays_actually_traced.memset_whole_buffer(1);
	}

	if (render_data.render_settings.sample_number > m_nee_plus_plus.stop_update_samples)
		// Past a certain number of samples, there isn't really a point to keep updating, the visibility map
		// is probably converged enough that it doesn't make a difference anymore
		render_data.nee_plus_plus.update_visibility_map = false;

	m_nee_plus_plus.milliseconds_before_finalizing_accumulation -= delta_time;
	m_nee_plus_plus.milliseconds_before_finalizing_accumulation = hippt::max(0.0f, m_nee_plus_plus.milliseconds_before_finalizing_accumulation); // Clamping for nice display in ImGui (0.0f instead of negative values)
	if (m_nee_plus_plus.milliseconds_before_finalizing_accumulation <= 0.0f && render_data.nee_plus_plus.packed_buffers != nullptr)
	{
		m_nee_plus_plus.milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_TIMER;

		// Because the visibility map data is packed, we can't just use a memcpy() to copy from the accumulation
		// buffers to the visibilit map, we have to use a kernel that the does unpacking-copy
		void* launch_args[] = { &render_data.nee_plus_plus };
		m_kernels[NEEPlusPlusRenderPass::FINALIZE_ACCUMULATION_KERNEL_ID]->launch_asynchronous(256, 1, matrix_element_count, 1, launch_args, m_renderer->get_main_stream());
	}

	m_nee_plus_plus.statistics_refresh_timer -= delta_time;
	if (m_nee_plus_plus.statistics_refresh_timer <= 0.0f && render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics)
	{
		m_nee_plus_plus.statistics_refresh_timer = NEEPlusPlusGPUData::STATISTICS_REFRESH_TIMER;

		OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.total_shadow_ray_queries_cpu, m_nee_plus_plus.total_shadow_ray_queries.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
		OROCHI_CHECK_ERROR(oroMemcpy(&m_nee_plus_plus.shadow_rays_actually_traced_cpu, m_nee_plus_plus.shadow_rays_actually_traced.get_device_pointer(), sizeof(unsigned long long int), oroMemcpyDeviceToHost));
	}

    return updated;
}
 
void NEEPlusPlusRenderPass::update_render_data()
{
    HIPRTRenderData& render_data = m_renderer->get_render_data();

    if (is_render_pass_used())
    {
	    render_data.nee_plus_plus.packed_buffers = m_nee_plus_plus.packed_buffer.get_atomic_device_pointer();
	    render_data.nee_plus_plus.shadow_rays_actually_traced = m_nee_plus_plus.shadow_rays_actually_traced.get_atomic_device_pointer();
    	render_data.nee_plus_plus.total_shadow_ray_queries = m_nee_plus_plus.total_shadow_ray_queries.get_atomic_device_pointer();
    }
    else
    {
	render_data.nee_plus_plus.packed_buffers = nullptr;
		render_data.nee_plus_plus.shadow_rays_actually_traced = nullptr;
		render_data.nee_plus_plus.total_shadow_ray_queries = nullptr;
	}
}

bool NEEPlusPlusRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) { return false; }
 
void NEEPlusPlusRenderPass::reset(bool reset_by_camera_movement)
{
     if (!is_render_pass_used())
         return;

    HIPRTRenderData& render_data = m_renderer->get_render_data();

    render_data.nee_plus_plus.reset_visibility_map = true;
	render_data.nee_plus_plus.update_visibility_map = true;

	// Resetting the counters
	if (m_nee_plus_plus.total_shadow_ray_queries.is_allocated())
	{
		m_nee_plus_plus.total_shadow_ray_queries.memset_whole_buffer(1);
		m_nee_plus_plus.shadow_rays_actually_traced.memset_whole_buffer(1);
	}
	m_nee_plus_plus.total_shadow_ray_queries_cpu = 1;
	m_nee_plus_plus.shadow_rays_actually_traced_cpu = 1;

	m_nee_plus_plus.milliseconds_before_finalizing_accumulation = NEEPlusPlusGPUData::FINALIZE_ACCUMULATION_START_TIMER;
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
