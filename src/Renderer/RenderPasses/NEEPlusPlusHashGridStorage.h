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
	static constexpr unsigned int DEFAULT_GRID_SIZE = 1000000;

	void set_nee_plus_plus_render_pass(NEEPlusPlusRenderPass* nee_plus_plus_render_pass);

	bool pre_render_update(HIPRTRenderData& render_data, bool is_interacting_camera);
	
	void update_render_data(HIPRTRenderData& render_data);
	bool free();
	void reset();

	bool try_resize(HIPRTRenderData& render_data, float max_megabyte_size);

	unsigned int update_cell_alive_count();
	unsigned int get_cell_alive_count();

	std::size_t get_shadow_rays_actually_traced_from_GPU() const;
	std::size_t get_total_shadow_rays_queries_from_GPU() const;
	std::size_t get_byte_size() const;

private:
	NEEPlusPlusRenderPass* m_nee_plus_plus_render_pass;

	OrochiBuffer<unsigned int> m_total_unoccluded_rays;
	OrochiBuffer<unsigned int> m_total_num_rays;

	OrochiBuffer<unsigned int> m_checksum_buffer;
	
	// Counters on the GPU for tracking 
	OrochiBuffer<unsigned long long int> m_total_shadow_ray_queries;
	OrochiBuffer<unsigned long long int> m_shadow_rays_actually_traced;

	OrochiBuffer<unsigned int> m_total_cells_alive_count;
	OrochiBuffer<unsigned int> m_total_cells_alive_count_cpu_host_pinned_buffer;
	unsigned int m_total_cells_alive_count_cpu = 0;
};

#endif