/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NEE_PLUS_PLUS_HASH_GRID_STORAGE_H
#define RENDERER_NEE_PLUS_PLUS_HASH_GRID_STORAGE_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/RenderData.h"

class NEEPlusPlusRenderPass;

class NEEPlusPlusHashGridStorage
{
public:
	static constexpr unsigned int DEFAULT_GRID_SIZE = 10000;

	void set_nee_plus_plus_render_pass(NEEPlusPlusRenderPass* nee_plus_plus_render_pass);

	bool pre_render_update(HIPRTRenderData& render_data);
	void post_sample_update_async(HIPRTRenderData& render_data);
	
	void update_render_data();
	bool free();
	void reset();

	std::size_t get_shadow_rays_actually_traced() const;
	std::size_t get_total_shadow_rays_queries() const;
	std::size_t get_byte_size() const;

private:
	NEEPlusPlusRenderPass* m_nee_plus_plus_render_pass;

	OrochiBuffer<unsigned int> m_total_unoccluded_rays;
	OrochiBuffer<unsigned int> m_total_num_rays;

	OrochiBuffer<unsigned int> m_checksum_buffer;
	
	// Counters on the GPU for tracking 
	OrochiBuffer<unsigned long long int> m_total_shadow_ray_queries;
	OrochiBuffer<unsigned long long int> m_shadow_rays_actually_traced;

	// Same counters but on the CPU for displaying the stats in ImGui.
	// These counters are updated 
	unsigned long long int m_total_shadow_ray_queries_cpu = 1;
	unsigned long long int m_shadow_rays_actually_traced_cpu = 1;
};

#endif