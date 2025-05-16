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
	return m_grid_buffers.get_byte_size() + m_spatial_reuse_output_grid_buffer.get_byte_size() + m_hash_cell_data.get_byte_size();
}

bool ReGIRHashGridStorage::pre_render_update(HIPRTRenderData& render_data)
{
	bool updated = false;

	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	if (m_total_number_of_cells == 0 ||
		m_current_grid_resolution.x != regir_settings.grid_fill_grid.grid_resolution.x ||
		m_current_grid_resolution.y != regir_settings.grid_fill_grid.grid_resolution.y ||
		m_current_grid_resolution.z != regir_settings.grid_fill_grid.grid_resolution.z)
	{
		m_total_number_of_cells = m_current_grid_resolution.x * m_current_grid_resolution.y * m_current_grid_resolution.z * m_hash_grid_current_overallocation_factor;
		m_current_grid_resolution = regir_settings.grid_fill_grid.grid_resolution;

		// We need a full reset of the grid
		m_grid_buffers.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell());
		if (regir_settings.spatial_reuse.do_spatial_reuse)
			// Also resizing the spatial reuse buffer
			m_spatial_reuse_output_grid_buffer.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell());

		m_hash_cell_data.resize(m_total_number_of_cells);

		updated = true;
	}
	else
	{
		// We don't need a full reset, instead checking if we need to dynamically grow the size of the hash
		// table to keep the load factor in check
		printf("Test rehash: %d / %d = %f%%\n", m_hash_cell_data.m_grid_cells_alive_count.download_data()[0], m_total_number_of_cells, m_regir_render_pass->get_alive_cells_ratio() * 100.0f);

		if (m_regir_render_pass->get_alive_cells_ratio() > 1.75f)
		{
			unsigned int m_grid_cells_alive = m_regir_render_pass->update_cell_alive_count();
			if (m_grid_cells_alive > 0)
			{
				std::cout << "Rehashing" << std::endl;

				unsigned int grid_cell_alive_count_before = 0;
				std::vector<unsigned int> data_alive_before = m_hash_cell_data.m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>().download_data();
				for (unsigned int& cell_alive : data_alive_before)
					if (cell_alive > 0)
					grid_cell_alive_count_before++;

				std::cout << "Cell alive count before (auto count): " << m_hash_cell_data.m_grid_cells_alive_count.download_data()[0] << std::endl;
				std::cout << "Cell alive count before (manual count): " << grid_cell_alive_count_before << std::endl;

				// Increasing the allocation factor 
				m_hash_grid_current_overallocation_factor *= 1.5f;

				m_total_number_of_cells = m_current_grid_resolution.x * m_current_grid_resolution.y * m_current_grid_resolution.z * m_hash_grid_current_overallocation_factor;

				// Allocating a larger hash table
				ReGIRHashGridSoAHost<OrochiBuffer> new_hash_grid;
				new_hash_grid.resize(m_total_number_of_cells, render_data.render_settings.regir_settings.get_number_of_reservoirs_per_cell());

				ReGIRHashCellDataSoAHost<OrochiBuffer> new_hash_cell_data;
				new_hash_cell_data.resize(m_total_number_of_cells);

				OrochiBuffer<unsigned int> new_grid_cells_alive_buffer(m_total_number_of_cells);

				// Initializing all the cells to inactive
				std::vector<unsigned int> init_data_alive(m_total_number_of_cells, 0u);
				new_grid_cells_alive_buffer.upload_data(init_data_alive);

				OrochiBuffer<unsigned int> new_grid_cells_alive_list(m_total_number_of_cells);
				std::vector<unsigned int> zero_data(m_total_number_of_cells);
				new_grid_cells_alive_list.upload_data(zero_data);

				ReGIRHashGridSoADevice new_hash_grid_device = new_hash_grid.to_device(m_current_grid_resolution);
				ReGIRHashCellDataSoADevice new_hash_cell_data_device = new_hash_cell_data.to_device();

				// For each cell alive, we're going to insert it in the new, larger, hash table, with a GPU kernel to do that
				m_regir_render_pass->launch_rehashing_kernel(render_data,
					new_hash_grid_device, new_hash_cell_data_device,
					new_grid_cells_alive_buffer.get_device_pointer(), new_grid_cells_alive_list.get_device_pointer());

				m_hash_cell_data.m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>() = std::move(new_grid_cells_alive_buffer);
				m_hash_cell_data.m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE_LIST>() = std::move(new_grid_cells_alive_list);

				m_grid_buffers = std::move(new_hash_grid);
				if (regir_settings.spatial_reuse.do_spatial_reuse)
					m_spatial_reuse_output_grid_buffer.resize(m_total_number_of_cells, regir_settings.get_number_of_reservoirs_per_cell());
				m_hash_cell_data = std::move(new_hash_cell_data);

				unsigned int grid_cell_alive_count = 0;
				std::vector<unsigned int> data_alive = m_hash_cell_data.m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>().download_data();
				for (unsigned int& cell_alive : data_alive)
					if (cell_alive > 0)
						grid_cell_alive_count++;

				std::cout << "Cell alive count after rehashing (manual count): " << grid_cell_alive_count << " / " << data_alive.size() << std::endl;

				updated = true;
			}
		}
	}

	return updated;
}

void ReGIRHashGridStorage::reset()
{
	std::vector<int> primitive_reset(m_total_number_of_cells, ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE);
	m_hash_cell_data.m_hash_cell_data.template get_buffer<REGIR_HASH_CELL_PRIM_INDEX>().upload_data(primitive_reset);

	std::vector<unsigned int> hash_keys_reset(m_total_number_of_cells, ReGIRHashCellDataSoADevice::UNDEFINED_HASH_KEY);
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

	if (m_hash_cell_data.get_byte_size() > 0)
	{
		m_hash_cell_data.free();

		updated = true;
	}

	return updated;
}

void ReGIRHashGridStorage::to_device(HIPRTRenderData& render_data)
{
	render_data.render_settings.regir_settings.grid_fill_grid = m_grid_buffers.to_device(render_data.render_settings.regir_settings.grid_fill_grid.grid_resolution);
	if (render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
		render_data.render_settings.regir_settings.spatial_grid = m_spatial_reuse_output_grid_buffer.to_device(render_data.render_settings.regir_settings.grid_fill_grid.grid_resolution);

	render_data.render_settings.regir_settings.hash_cell_data = m_hash_cell_data.to_device();
}

ReGIRHashCellDataSoAHost<OrochiBuffer>& ReGIRHashGridStorage::get_hash_cell_data_soa()
{
	return m_hash_cell_data;
}
