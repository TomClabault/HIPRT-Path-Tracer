/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/ReGIRHashGridStorage.h"
#include "Renderer/RenderPasses/ReGIRRenderPass.h"

void ReGIRHashGridStorage::set_regir_render_pass(ReGIRRenderPass* regir_render_pass)
{
	m_regir_render_pass = regir_render_pass;
}

std::size_t ReGIRHashGridStorage::get_byte_size() const
{
	return m_presampled_lights.get_byte_size() + 

		m_initial_reservoirs_primary_hits_grid.get_byte_size() +
		m_initial_reservoirs_secondary_hits_grid.get_byte_size() +

		m_spatial_output_primary_hits_grid.get_byte_size() +
		m_spatial_output_secondary_hits_grid.get_byte_size() +

		m_hash_cell_data_primary_hits.get_byte_size() +
		m_hash_cell_data_secondary_hits.get_byte_size() +

		m_supersample_grid_primary_hits.get_byte_size() +

		m_canonical_pre_integration_factors_primary_hits.get_byte_size() +
		m_canonical_pre_integration_factors_secondary_hits.get_byte_size();
}

bool ReGIRHashGridStorage::pre_render_update(HIPRTRenderData& render_data)
{
	bool updated = false;

	updated |= pre_render_update_internal(render_data, true);
	updated |= pre_render_update_internal(render_data, false);

	return updated;
}

bool ReGIRHashGridStorage::pre_render_update_internal(HIPRTRenderData& render_data, bool primary_hit)
{
	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
	if (render_data.render_settings.nb_bounces == 0 && !primary_hit)
	{
		// For the special case of 0 bounces in the scene, we can free the secondary hits cells because
		// they are never going to be used
		return free_internal(false);
	}

	bool grid_not_allocated = get_total_number_of_cells(primary_hit) == 0;
	bool grid_res_changed = m_current_grid_min_cell_size != regir_settings.hash_grid.m_grid_cell_min_size || m_grid_cell_target_projected_size != regir_settings.hash_grid.m_grid_cell_target_projected_size;
	bool reservoirs_per_cell_changed = regir_settings.get_number_of_reservoirs_per_cell(primary_hit) != get_initial_grid_buffers(primary_hit).m_reservoirs_per_cell;

	bool needs_grid_resize = grid_not_allocated || grid_res_changed || reservoirs_per_cell_changed;

	bool updated = false;
	if (needs_grid_resize)
	{
		get_total_number_of_cells(primary_hit) = primary_hit ? ReGIRHashGridStorage::DEFAULT_GRID_CELL_COUNT_PRIMARY_HITS : ReGIRHashGridStorage::DEFAULT_GRID_CELL_COUNT_SECONDARY_HITS; // Default grid size

		m_current_grid_min_cell_size = regir_settings.hash_grid.m_grid_cell_min_size;
		m_grid_cell_target_projected_size = regir_settings.hash_grid.m_grid_cell_target_projected_size;

		get_initial_grid_buffers(primary_hit).resize(get_total_number_of_cells(primary_hit), regir_settings.get_number_of_reservoirs_per_cell(primary_hit));
		get_hash_cell_data_soa(primary_hit).resize(get_total_number_of_cells(primary_hit));

		get_non_canonical_factors(primary_hit).resize(get_total_number_of_cells(primary_hit));
		get_canonical_factors(primary_hit).resize(get_total_number_of_cells(primary_hit));

		updated = true;
	}

	if (regir_settings.spatial_reuse.do_spatial_reuse)
	{
		bool spatial_grid_not_allocated = get_spatial_grid_buffers(primary_hit).m_total_number_of_cells == 0;
		bool reservoirs_per_cell_spatial_changed = regir_settings.get_number_of_reservoirs_per_cell(primary_hit) != get_spatial_grid_buffers(primary_hit).m_reservoirs_per_cell;

		bool needs_spatial_grid_resize = spatial_grid_not_allocated || grid_res_changed || reservoirs_per_cell_spatial_changed;

		if (needs_spatial_grid_resize)
		{
			// Resizing the spatial buffer
			get_spatial_grid_buffers(primary_hit).resize(get_total_number_of_cells(primary_hit), regir_settings.get_number_of_reservoirs_per_cell(primary_hit));

			updated = true;
		}
	}
	else
	{
		if (get_spatial_grid_buffers(primary_hit).m_total_number_of_cells > 0)
			get_spatial_grid_buffers(primary_hit).free();
	}

	if (primary_hit)
	{
		if (regir_settings.supersampling.do_correlation_reduction)
		{
			bool supersample_grid_not_allocated = m_supersample_grid_primary_hits.m_total_number_of_cells == 0;
			bool supersample_reservoirs_count_changed = regir_settings.get_number_of_reservoirs_per_cell(primary_hit) != m_supersample_grid_primary_hits.m_reservoirs_per_cell / regir_settings.supersampling.correlation_reduction_factor;
			bool needs_supersample_grid_resize = supersample_grid_not_allocated || grid_res_changed || supersample_reservoirs_count_changed;

			if (needs_supersample_grid_resize)
			{
				m_supersample_grid_primary_hits.resize(get_total_number_of_cells(true), regir_settings.get_number_of_reservoirs_per_cell(true) * regir_settings.supersampling.correlation_reduction_factor);

				m_supersampling_curent_grid_offset = 0;
				m_supersampling_frames_available = 0;

				updated = true;
			}
		}
		else
		{
			if (m_supersample_grid_primary_hits.m_total_number_of_cells > 0)
				m_supersample_grid_primary_hits.free();
		}

		if (m_regir_render_pass->get_renderer()->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_DO_LIGHT_PRESAMPLING))
		{
			unsigned int presampled_lights_count_needed = render_data.render_settings.regir_settings.presampled_lights.get_presampled_light_count();
			if (m_presampled_lights.size() != presampled_lights_count_needed)
			{
				// If the current presampled light buffer isn't the right size, resizing
				m_presampled_lights.resize(presampled_lights_count_needed);

				updated = true;
			}
		}
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
	m_supersampling_curent_grid_offset %= render_data.render_settings.regir_settings.supersampling.correlation_reduction_factor;

	m_supersampling_frames_available++;
	m_supersampling_frames_available = hippt::min(m_supersampling_frames_available, render_data.render_settings.regir_settings.supersampling.correlation_reduction_factor);
}

bool ReGIRHashGridStorage::try_rehash(HIPRTRenderData& render_data)
{
	bool rehashed = false;

	rehashed |= try_rehash_internal(render_data, true);
	rehashed |= try_rehash_internal(render_data, false);

	return rehashed;
}

bool ReGIRHashGridStorage::try_rehash_internal(HIPRTRenderData& render_data, bool primary_hit)
{
	ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	// We don't need a full reset, instead checking if we need to dynamically grow the size of the hash
	// table to keep the load factor in check
	float cell_alive_ratio = m_regir_render_pass->get_alive_cells_ratio(primary_hit);	
	if (cell_alive_ratio > 0.60f)
	{
		m_regir_render_pass->update_all_cell_alive_count(render_data);

		unsigned int m_grid_cells_alive = m_regir_render_pass->get_number_of_cells_alive(primary_hit);
		if (m_grid_cells_alive > 0)
		{
			// Increasing the number of cells
			get_total_number_of_cells(primary_hit) *= 1.5;

			// Allocating a larger hash table
			ReGIRHashGridSoAHost<OrochiBuffer> new_hash_grid_soa;
			new_hash_grid_soa.resize(get_total_number_of_cells(primary_hit), regir_settings.get_number_of_reservoirs_per_cell(primary_hit));

			ReGIRHashCellDataSoAHost<OrochiBuffer> new_hash_cell_data;
			new_hash_cell_data.resize(get_total_number_of_cells(primary_hit));

			ReGIRHashGridSoADevice new_hash_grid_device;
			new_hash_grid_soa.to_device(new_hash_grid_device);

			ReGIRHashCellDataSoADevice new_hash_cell_data_device = new_hash_cell_data.to_device();

			// For each cell alive, we're going to insert it in the new, larger, hash table, with a GPU kernel to do that
			m_regir_render_pass->launch_rehashing_kernel(render_data, primary_hit, new_hash_grid_device, new_hash_cell_data_device);

			get_initial_grid_buffers(primary_hit) = std::move(new_hash_grid_soa);
			if (regir_settings.spatial_reuse.do_spatial_reuse)
				get_spatial_grid_buffers(primary_hit).resize(get_total_number_of_cells(primary_hit), regir_settings.get_number_of_reservoirs_per_cell(primary_hit));
			if (regir_settings.supersampling.do_correlation_reduction && primary_hit)
			{
				m_supersample_grid_primary_hits.resize(get_total_number_of_cells(true), regir_settings.get_number_of_reservoirs_per_cell(true) * regir_settings.supersampling.correlation_reduction_factor);

				m_supersampling_curent_grid_offset = 0;
				m_supersampling_frames_available = 0;
			}
			get_hash_cell_data_soa(primary_hit) = std::move(new_hash_cell_data);

			get_non_canonical_factors(primary_hit).resize(get_total_number_of_cells(primary_hit));
			get_canonical_factors(primary_hit).resize(get_total_number_of_cells(primary_hit));

			// We need to update the cell alive count because there may have possibly been collisions that couldn't be resolved during the rehashing
			// and maybe some cells could not be reinserted in the new hash table --> the cell alive count is different (lower) --> need to update
			m_regir_render_pass->update_all_cell_alive_count(render_data);

			return true;
		}
	}

	return false;
}

void ReGIRHashGridStorage::reset()
{
	reset_internal(true);

	if (m_regir_render_pass->get_renderer()->get_render_data().render_settings.nb_bounces > 0 && get_initial_grid_buffers(false).get_byte_size() > 0)
	{
		// If the renderer has more than 0 bounce, then we actually have secondary grid cells to reset
		reset_internal(false);
	}
}

void ReGIRHashGridStorage::reset_internal(bool primary_hit)
{
	// Resetting the 'cell alive' buffers
	get_hash_cell_data_soa(primary_hit).m_hash_cell_data.template get_buffer<ReGIRHashCellDataSoAHostBuffers::REGIR_HASH_CELLS_ALIVE>().memset_whole_buffer(0);

	// Resetting the pre-integration factors buffer
	get_non_canonical_factors(primary_hit).memset_whole_buffer(0.0f);
	get_canonical_factors(primary_hit).memset_whole_buffer(0.0f);

	// Resetting the count buffers
	get_hash_cell_data_soa(primary_hit).m_grid_cells_alive_count.memset_whole_buffer(0);

	get_hash_cell_data_soa(primary_hit).m_hash_cell_data.template get_buffer<REGIR_HASH_CELL_PRIM_INDEX>().memset_whole_buffer(ReGIRHashCellDataSoADevice::UNDEFINED_PRIMITIVE);
	get_hash_cell_data_soa(primary_hit).m_hash_cell_data.template get_buffer<REGIR_HASH_CELL_CHECKSUMS>().memset_whole_buffer(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

	// Resetting the reservoirs
	get_initial_grid_buffers(primary_hit).reservoirs.get_buffer<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>().memset_whole_buffer(ReGIRReservoir::UNDEFINED_UCW);
	if (m_regir_render_pass->get_renderer()->get_render_data().render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
	{
		if (get_spatial_grid_buffers(primary_hit).reservoirs.get_buffer<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>().size() > 0)
			// We need to check the size before the reset because the reset method is called before the pre_render_update method
			// (where the buffer is allocated) so this reset call may try to reset a buffer that wasn't allocated
			get_spatial_grid_buffers(primary_hit).reservoirs.get_buffer<ReGIRReservoirSoAHostBuffers::REGIR_RESERVOIR_UCW>().memset_whole_buffer(ReGIRReservoir::UNDEFINED_UCW);
	}
}

bool ReGIRHashGridStorage::free()
{
	bool updated = false;

	updated |= free_internal(true);
	updated |= free_internal(false);

	return updated;
}

bool ReGIRHashGridStorage::free_internal(bool primary_hit)
{
	bool updated = false;

	if (get_initial_grid_buffers(primary_hit).get_byte_size() > 0)
	{
		get_initial_grid_buffers(primary_hit).free();

		updated = true;
	}

	if (get_spatial_grid_buffers(primary_hit).get_byte_size() > 0)
	{
		get_spatial_grid_buffers(primary_hit).free();

		updated = true;
	}

	if (m_supersample_grid_primary_hits.get_byte_size() > 0 && primary_hit)
	{
		m_supersample_grid_primary_hits.free();

		updated = true;
	}

	if (get_hash_cell_data_soa(primary_hit).get_byte_size() > 0)
	{
		get_hash_cell_data_soa(primary_hit).free();

		updated = true;
	}

	if (get_non_canonical_factors(primary_hit).get_byte_size() > 0)
	{
		get_non_canonical_factors(primary_hit).free();

		updated = true;
	}

	if (get_canonical_factors(primary_hit).get_byte_size() > 0)
	{
		get_canonical_factors(primary_hit).free();

		updated = true;
	}

	if (primary_hit && m_presampled_lights.get_byte_size() > 0)
		// Only freeing the presampled lights on the first hit by convention
		m_presampled_lights.free();

	if (primary_hit)
		m_total_number_of_cells_primary_hits = 0;
	else
		m_total_number_of_cells_secondary_hits = 0;

	return updated;
}

void ReGIRHashGridStorage::clear_pre_integrated_RIS_integral_factors(bool primary_hit)
{
	get_non_canonical_factors(primary_hit).memset_whole_buffer(0.0f);
	get_canonical_factors(primary_hit).memset_whole_buffer(0.0f);
}

void ReGIRHashGridStorage::to_device(HIPRTRenderData& render_data)
{
	if (m_regir_render_pass->get_renderer()->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_DO_LIGHT_PRESAMPLING))
		m_presampled_lights.to_device(render_data.render_settings.regir_settings.presampled_lights.presampled_lights_soa);

	// Primary hits grid cells
	m_initial_reservoirs_primary_hits_grid.to_device(render_data.render_settings.regir_settings.initial_reservoirs_primary_hits_grid);

	if (render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
		m_spatial_output_primary_hits_grid.to_device(render_data.render_settings.regir_settings.spatial_output_primary_hits_grid);

	if (render_data.render_settings.regir_settings.supersampling.do_correlation_reduction)
		m_supersample_grid_primary_hits.to_device(render_data.render_settings.regir_settings.supersampling.correlation_reduction_grid);

	render_data.render_settings.regir_settings.hash_cell_data_primary_hits = m_hash_cell_data_primary_hits.to_device();

	render_data.render_settings.regir_settings.non_canonical_pre_integration_factors_primary_hits = get_non_canonical_factors(true).get_atomic_device_pointer();
	render_data.render_settings.regir_settings.canonical_pre_integration_factors_primary_hits = get_canonical_factors(true).get_atomic_device_pointer();

	// Secondary hits grid cells
	if (render_data.render_settings.nb_bounces > 0)
	{
		m_initial_reservoirs_secondary_hits_grid.to_device(render_data.render_settings.regir_settings.initial_reservoirs_secondary_hits_grid);

		if (render_data.render_settings.regir_settings.spatial_reuse.do_spatial_reuse)
			m_spatial_output_secondary_hits_grid.to_device(render_data.render_settings.regir_settings.spatial_output_secondary_hits_grid);

		render_data.render_settings.regir_settings.hash_cell_data_secondary_hits = m_hash_cell_data_secondary_hits.to_device();

		render_data.render_settings.regir_settings.non_canonical_pre_integration_factors_secondary_hits = get_non_canonical_factors(false).get_atomic_device_pointer();
		render_data.render_settings.regir_settings.canonical_pre_integration_factors_secondary_hits = get_canonical_factors(false).get_atomic_device_pointer();
	}
}

ReGIRHashGridSoAHost<OrochiBuffer>& ReGIRHashGridStorage::get_initial_grid_buffers(bool primary_hit)
{
	return primary_hit ? m_initial_reservoirs_primary_hits_grid : m_initial_reservoirs_secondary_hits_grid;
}

ReGIRHashGridSoAHost<OrochiBuffer>& ReGIRHashGridStorage::get_spatial_grid_buffers(bool primary_hit)
{
	return primary_hit ? m_spatial_output_primary_hits_grid : m_spatial_output_secondary_hits_grid;
}

ReGIRHashCellDataSoAHost<OrochiBuffer>& ReGIRHashGridStorage::get_hash_cell_data_soa(bool primary_hit)
{
	return primary_hit ? m_hash_cell_data_primary_hits : m_hash_cell_data_secondary_hits;
}

ReGIRHashCellDataSoADevice& ReGIRHashGridStorage::get_hash_cell_data_device_soa(ReGIRSettings& regir_settings, bool primary_hit)
{
	return regir_settings.get_hash_cell_data_soa(primary_hit);
}

OrochiBuffer<float>& ReGIRHashGridStorage::get_non_canonical_factors(bool primary_hit)
{
	return primary_hit ? m_non_canonical_pre_integration_factors_primary_hits : m_non_canonical_pre_integration_factors_secondary_hits;
}

OrochiBuffer<float>& ReGIRHashGridStorage::get_canonical_factors(bool primary_hit)
{
	return primary_hit ? m_canonical_pre_integration_factors_primary_hits : m_canonical_pre_integration_factors_secondary_hits;
}

unsigned int& ReGIRHashGridStorage::get_total_number_of_cells(bool primary_hit)
{
	return primary_hit ? m_total_number_of_cells_primary_hits : m_total_number_of_cells_secondary_hits;
}

unsigned int ReGIRHashGridStorage::get_total_number_of_cells(bool primary_hit) const
{
	return primary_hit ? m_total_number_of_cells_primary_hits : m_total_number_of_cells_secondary_hits;
}

unsigned int ReGIRHashGridStorage::get_supersampling_current_frame() const
{
	return m_supersampling_curent_grid_offset;
}

unsigned int ReGIRHashGridStorage::get_supersampling_frames_available() const
{
	return m_supersampling_frames_available;
}
