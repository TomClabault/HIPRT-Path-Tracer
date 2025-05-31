/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/RenderPasses/ReGIRHashGridStorage.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

void ReGIRHashGridStorage::set_regir_render_pass(ReGIRRenderPass* regir_render_pass)
{
	m_regir_render_pass = regir_render_pass;
}

unsigned int ReGIRHashGridStorage::get_total_number_of_cells() const
{
	return m_total_number_of_cells;
}

std::size_t ReGIRHashGridStorage::get_byte_size() const
{
	return m_grid_buffers.get_byte_size() + m_spatial_reuse_output_grid_buffer.get_byte_size() + m_supersample_grid.get_byte_size() + m_hash_cell_data.get_byte_size();
}

bool ReGIRHashGridStorage::pre_render_update(HIPRTRenderData& render_data)
{
	bool updated = false;

	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	bool grid_not_allocated = m_total_number_of_cells == 0;
	bool grid_res_changed = m_current_grid_min_cell_size != regir_settings.hash_grid.m_grid_cell_min_size || m_grid_cell_target_projected_size != regir_settings.hash_grid.m_grid_cell_target_projected_size;
	bool reservoirs_per_cell_changed = regir_settings.get_number_of_reservoirs_per_cell() != m_grid_buffers.m_reservoirs_per_cell;

	bool needs_grid_resize = grid_not_allocated || grid_res_changed || reservoirs_per_cell_changed;

	if (needs_grid_resize)
	{
		if (grid_not_allocated)
			m_total_number_of_cells = ReGIRHashGridStorage::DEFAULT_GRID_CELL_COUNT; // Default grid size

		m_current_grid_min_cell_size = regir_settings.hash_grid.m_grid_cell_min_size;
		m_grid_cell_target_projected_size = regir_settings.hash_grid.m_grid_cell_target_projected_size;

		m_grid_buffers.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell());
		m_hash_cell_data.resize(m_total_number_of_cells);

		updated = true;
	}

	if (regir_settings.spatial_reuse.do_spatial_reuse)
	{
		bool spatial_grid_not_allocated = m_spatial_reuse_output_grid_buffer.m_total_number_of_cells == 0;
		bool reservoirs_per_cell_spatial_changed = regir_settings.get_number_of_reservoirs_per_cell() != m_spatial_reuse_output_grid_buffer.m_reservoirs_per_cell;

		bool needs_spatial_grid_resize = spatial_grid_not_allocated || grid_res_changed || reservoirs_per_cell_spatial_changed;

		if (needs_spatial_grid_resize)
		{
			// Resizing the spatial buffer
			m_spatial_reuse_output_grid_buffer.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell());

			updated = true;
		}
	}
	else
	{
		if (m_spatial_reuse_output_grid_buffer.m_total_number_of_cells > 0)
			m_spatial_reuse_output_grid_buffer.free();
	}

	if (regir_settings.supersampling.do_supersampling)
	{
		bool supersample_grid_not_allocated = m_supersample_grid.m_total_number_of_cells == 0;
		bool supersample_reservoirs_count_changed = regir_settings.get_number_of_reservoirs_per_cell() != m_supersample_grid.m_reservoirs_per_cell / regir_settings.supersampling.supersampling_factor;
		bool needs_supersample_grid_resize = supersample_grid_not_allocated || grid_res_changed || supersample_reservoirs_count_changed;

		if (needs_supersample_grid_resize)
		{
			m_supersample_grid.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell() * regir_settings.supersampling.supersampling_factor);

			m_supersampling_curent_grid_offset = 0;
			m_supersampling_frames_available = 0;

			updated = true;
		}
	}
	else
	{
		if (m_supersample_grid.m_total_number_of_cells > 0)
			m_supersample_grid.free();
	}

	return updated;
}

void ReGIRHashGridStorage::post_sample_update_async(HIPRTRenderData& render_data)
{
	increment_supersampling_counters(render_data);
}

void ReGIRHashGridStorage::increment_supersampling_counters(HIPRTRenderData& render_data)
{
	m_supersampling_curent_grid_offset++;
	m_supersampling_curent_grid_offset %= render_data.render_settings.regir_settings.supersampling.supersampling_factor;

	m_supersampling_frames_available++;
	m_supersampling_frames_available = hippt::min(m_supersampling_frames_available, render_data.render_settings.regir_settings.supersampling.supersampling_factor);
}

bool ReGIRHashGridStorage::try_rehash(HIPRTRenderData& render_data)
{
	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	// We don't need a full reset, instead checking if we need to dynamically grow the size of the hash
	// table to keep the load factor in check
	if (m_regir_render_pass->get_alive_cells_ratio() > 0.75f)
	{
		unsigned int m_grid_cells_alive = m_regir_render_pass->update_cell_alive_count();
		if (m_grid_cells_alive > 0)
		{
			// Increasing the number of cells
			m_total_number_of_cells *= 1.5;

			// Allocating a larger hash table
			ReGIRHashGridSoAHost<OrochiBuffer> new_hash_grid_soa;
			new_hash_grid_soa.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell());

			ReGIRHashCellDataSoAHost<OrochiBuffer> new_hash_cell_data;
			new_hash_cell_data.resize(m_total_number_of_cells);

			ReGIRHashGridSoADevice new_hash_grid_device;
			new_hash_grid_soa.to_device(new_hash_grid_device);

			ReGIRHashCellDataSoADevice new_hash_cell_data_device = new_hash_cell_data.to_device();

			// For each cell alive, we're going to insert it in the new, larger, hash table, with a GPU kernel to do that
			m_regir_render_pass->launch_rehashing_kernel(render_data, new_hash_grid_device, new_hash_cell_data_device);

			m_grid_buffers = std::move(new_hash_grid_soa);
			if (regir_settings.spatial_reuse.do_spatial_reuse)
				m_spatial_reuse_output_grid_buffer.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell());
			if (regir_settings.supersampling.do_supersampling)
			{
				m_supersample_grid.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell() * regir_settings.supersampling.supersampling_factor);

				m_supersampling_curent_grid_offset = 0;
				m_supersampling_frames_available = 0;
			}
			m_hash_cell_data = std::move(new_hash_cell_data);

			// We need to update the cell alive count because there may have possibly been collisions that couldn't be resolved during the rehashing
			// and maybe some cells could not be reinserted in the new hash table --> the cell alive count is different (lower) --> need to update
			m_regir_render_pass->update_cell_alive_count();
			
			return true;
		}
	}

	return false;
}

void ReGIRHashGridStorage::reset()
{
	std::vector<int> primitive_reset(m_total_number_of_cells, ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE);
	m_hash_cell_data.m_hash_cell_data.template get_buffer<REGIR_HASH_CELL_PRIM_INDEX>().upload_data(primitive_reset);

	std::vector<unsigned int> hash_keys_reset(m_total_number_of_cells, HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
	m_hash_cell_data.m_hash_cell_data.template get_buffer<REGIR_HASH_CELL_HASH_KEYS>().upload_data(hash_keys_reset);
}

bool ReGIRHashGridStorage::free()
{
	bool updated = false;

	if (m_grid_buffers.get_byte_size() > 0)
	{
		m_grid_buffers.free();

		updated = true;
	}

	if (m_spatial_reuse_output_grid_buffer.get_byte_size() > 0)
	{
		m_spatial_reuse_output_grid_buffer.free();

		updated = true;
	}

	if (m_supersample_grid.get_byte_size() > 0)
	{
		m_supersample_grid.free();

		updated = true;
	}

	if (m_hash_cell_data.get_byte_size() > 0)
	{
		m_hash_cell_data.free();

		updated = true;
	}

	m_total_number_of_cells = 0;

	return updated;
}

void ReGIRHashGridStorage::to_device(HIPRTRenderData& render_data)
{
	m_grid_buffers.to_device(render_data.render_settings.regir_settings.initial_reservoirs_grid);

	if (render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
		m_spatial_reuse_output_grid_buffer.to_device(render_data.render_settings.regir_settings.spatial_output_grid);

	if (render_data.render_settings.regir_settings.supersampling.do_supersampling)
		m_supersample_grid.to_device(render_data.render_settings.regir_settings.supersampling.supersampling_grid);

	render_data.render_settings.regir_settings.hash_cell_data = m_hash_cell_data.to_device();
}

ReGIRHashCellDataSoAHost<OrochiBuffer>& ReGIRHashGridStorage::get_hash_cell_data_soa()
{
	return m_hash_cell_data;
}

unsigned int ReGIRHashGridStorage::get_supersampling_current_frame() const
{
	return m_supersampling_curent_grid_offset;
}

unsigned int ReGIRHashGridStorage::get_supersampling_frames_available() const
{
	return m_supersampling_frames_available;
}
