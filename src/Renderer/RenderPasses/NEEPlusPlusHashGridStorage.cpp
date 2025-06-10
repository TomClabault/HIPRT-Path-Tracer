/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NEEPlusPlusHashGridStorage.h"
#include "Renderer/RenderPasses/NEEPlusPlusRenderPass.h"

void NEEPlusPlusHashGridStorage::set_nee_plus_plus_render_pass(NEEPlusPlusRenderPass* nee_plus_plus_render_pass)
{
	m_nee_plus_plus_render_pass = nee_plus_plus_render_pass;
}

bool NEEPlusPlusHashGridStorage::pre_render_update(HIPRTRenderData& render_data, bool is_interacting_camera)
{
	bool updated = false;

	// Allocating the buffers
	if (m_total_num_rays.size() == 0)
	{
		m_total_num_rays.resize(NEEPlusPlusHashGridStorage::DEFAULT_GRID_SIZE);
		m_total_unoccluded_rays.resize(NEEPlusPlusHashGridStorage::DEFAULT_GRID_SIZE);
		m_checksum_buffer.resize(NEEPlusPlusHashGridStorage::DEFAULT_GRID_SIZE);

		m_shadow_rays_actually_traced.resize(1);
		m_total_shadow_ray_queries.resize(1);
		m_total_cells_alive_count.resize(1);
		m_total_cells_alive_count_cpu_host_pinned_buffer.resize_host_pinned_mem(1);

		updated = true;
	}

	// Clearing the visibility map if this has been asked by the user
	if (render_data.nee_plus_plus.m_reset_visibility_map && !is_interacting_camera)
	{
		// Clearing the visibility map by memseting everything to 0
		// Adn let's not reset everything while moving the camera because that's annoying for
		// the performance

		m_total_num_rays.memset_whole_buffer(0);
		m_total_unoccluded_rays.memset_whole_buffer(0);
		m_checksum_buffer.memset_whole_buffer(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

		m_total_shadow_ray_queries.memset_whole_buffer(0);
		m_shadow_rays_actually_traced.memset_whole_buffer(0);
		m_total_cells_alive_count.memset_whole_buffer(0);
	}

	if (render_data.render_settings.sample_number > render_data.nee_plus_plus.m_stop_update_samples)
		// Past a certain number of samples, there isn't really a point to keep updating, the visibility map
		// is probably converged enough that it doesn't make a difference anymore
		render_data.nee_plus_plus.m_update_visibility_map = false;

	return updated;
}

void NEEPlusPlusHashGridStorage::update_render_data(HIPRTRenderData& render_data)
{
	if (m_nee_plus_plus_render_pass->is_render_pass_used())
	{
		render_data.nee_plus_plus.m_entries_buffer.total_num_rays = m_total_num_rays.get_atomic_device_pointer();
		render_data.nee_plus_plus.m_entries_buffer.total_unoccluded_rays = m_total_unoccluded_rays.get_atomic_device_pointer();
		render_data.nee_plus_plus.m_entries_buffer.checksum_buffer = m_checksum_buffer.get_atomic_device_pointer();
		render_data.nee_plus_plus.m_total_number_of_cells = m_checksum_buffer.size();

		render_data.nee_plus_plus.m_shadow_rays_actually_traced = m_shadow_rays_actually_traced.get_atomic_device_pointer();
		render_data.nee_plus_plus.m_total_shadow_ray_queries = m_total_shadow_ray_queries.get_atomic_device_pointer();
		render_data.nee_plus_plus.m_total_cells_alive_count = m_total_cells_alive_count.get_atomic_device_pointer();
	}
	else
	{
		render_data.nee_plus_plus.m_entries_buffer.total_num_rays = nullptr;
		render_data.nee_plus_plus.m_entries_buffer.total_unoccluded_rays = nullptr;
		render_data.nee_plus_plus.m_entries_buffer.checksum_buffer = nullptr;

		render_data.nee_plus_plus.m_shadow_rays_actually_traced = nullptr;
		render_data.nee_plus_plus.m_total_shadow_ray_queries = nullptr;
		render_data.nee_plus_plus.m_total_cells_alive_count = nullptr;
	}
}

bool NEEPlusPlusHashGridStorage::free()
{
	if (m_total_num_rays.size() != 0)
	{
		m_total_num_rays.free();
		m_total_unoccluded_rays.free();
		m_checksum_buffer.free();

		m_total_shadow_ray_queries.free();
		m_shadow_rays_actually_traced.free();

		m_total_cells_alive_count.free();
		m_total_cells_alive_count_cpu_host_pinned_buffer.free();

		return true;
	}

	return false;
}

void NEEPlusPlusHashGridStorage::reset()
{
	HIPRTRenderData& render_data = m_nee_plus_plus_render_pass->m_renderer->get_render_data();

	render_data.nee_plus_plus.m_reset_visibility_map = true;
	render_data.nee_plus_plus.m_update_visibility_map = true;

	// Resetting the counters
	if (m_total_shadow_ray_queries.is_allocated())
	{
		m_total_shadow_ray_queries.memset_whole_buffer(1);
		m_shadow_rays_actually_traced.memset_whole_buffer(1);
		m_total_cells_alive_count.memset_whole_buffer(0);
	}
}

bool NEEPlusPlusHashGridStorage::try_resize(HIPRTRenderData& render_data, float max_megabyte_size)
{
	update_cell_alive_count();

	float load_factor = m_total_cells_alive_count_cpu / (float)m_checksum_buffer.size();
	bool load_factor_too_high = load_factor > 0.75f;
	bool maximum_size_not_reached = get_byte_size() / 1000000.0f <= max_megabyte_size * 0.95f;
	bool maximum_size_exceeded = get_byte_size() / 1000000.0f >= max_megabyte_size * 1.05f;
	if ((load_factor_too_high && maximum_size_not_reached) || maximum_size_exceeded)
	{
		unsigned int current_cell_count = m_checksum_buffer.size();
		unsigned int new_cell_count = current_cell_count * 1.5f;

		float cell_size_byte = get_byte_size() / current_cell_count;
		float estimated_new_size_bytes = cell_size_byte * new_cell_count;
		if (estimated_new_size_bytes > max_megabyte_size * 1000000.0f)
			// If the estimated new size is larger than the maximum size,
			// we need to compute the proper number of cells to fit the maximum size
			new_cell_count = static_cast<unsigned int>(max_megabyte_size * 1000000.0f / cell_size_byte);

		m_total_unoccluded_rays.resize(new_cell_count);
		m_total_num_rays.resize(new_cell_count);
		m_checksum_buffer.resize(new_cell_count);

		m_total_unoccluded_rays.memset_whole_buffer(0);
		m_total_num_rays.memset_whole_buffer(0);
		m_checksum_buffer.memset_whole_buffer(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
		m_total_cells_alive_count.memset_whole_buffer(0);

		update_render_data(render_data);

		return true;
	}

	return false;
}

unsigned int NEEPlusPlusHashGridStorage::update_cell_alive_count()
{
	m_total_cells_alive_count.download_data_into(m_total_cells_alive_count_cpu_host_pinned_buffer.get_host_pinned_pointer());
	m_total_cells_alive_count_cpu = m_total_cells_alive_count_cpu_host_pinned_buffer.get_host_pinned_pointer()[0];

	return get_cell_alive_count();
}

unsigned int NEEPlusPlusHashGridStorage::get_cell_alive_count()
{
	return m_total_cells_alive_count_cpu;
}

std::size_t NEEPlusPlusHashGridStorage::get_shadow_rays_actually_traced_from_GPU() const
{
	auto data = m_shadow_rays_actually_traced.download_data();
	if (data.size() > 0)
		return data[0];
	else
		return 0;
}

std::size_t NEEPlusPlusHashGridStorage::get_total_shadow_rays_queries_from_GPU() const
{
	auto data = m_total_shadow_ray_queries.download_data();
	if (data.size() > 0)
		return data[0];
	else
		return 0;
}

std::size_t NEEPlusPlusHashGridStorage::get_byte_size() const
{
	return m_total_unoccluded_rays.get_byte_size() +
		m_total_num_rays.get_byte_size() +
		m_checksum_buffer.get_byte_size() +

		m_total_shadow_ray_queries.get_byte_size() +
		m_shadow_rays_actually_traced.get_byte_size() +
		m_total_cells_alive_count.get_byte_size();
}
