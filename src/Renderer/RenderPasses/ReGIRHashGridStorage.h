/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_HASH_GRID_STORAGE_H
#define REGIR_HASH_GRID_STORAGE_H

#include "HostDeviceCommon/RenderData.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRCellsAliasTablesSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRHashGridSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRHashCellDataSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRPresampledLightsSoAHost.h"

class ReGIRRenderPass;

class ReGIRHashGridStorage
{
public:
	static constexpr unsigned int DEFAULT_GRID_CELL_COUNT_PRIMARY_HITS = 500000;
	static constexpr unsigned int DEFAULT_GRID_CELL_COUNT_SECONDARY_HITS = 10000;

	void set_regir_render_pass(ReGIRRenderPass* regir_render_pass);

	std::size_t get_byte_size() const;

	bool pre_render_update(HIPRTRenderData& render_data);
	void post_sample_update_async(HIPRTRenderData& render_data);
	void increment_supersampling_counters(HIPRTRenderData& render_data);
	bool try_rehash(HIPRTRenderData& render_data);
	void reset();
	bool free();

	void clear_pre_integrated_RIS_integral_factors(bool primary_hit);

	void to_device(HIPRTRenderData& render_data);

	ReGIRHashGridSoAHost<OrochiBuffer>& get_initial_grid_buffers(bool primary_hit);
	ReGIRHashGridSoAHost<OrochiBuffer>& get_spatial_grid_buffers(bool primary_hit);
	ReGIRHashGridSoAHost<OrochiBuffer>& get_async_compute_staging_buffer(bool primary_hit);
	ReGIRHashGridSoADevice get_async_compute_staging_buffer_device(bool primary_hit);
	ReGIRHashCellDataSoAHost<OrochiBuffer>& get_hash_cell_data_soa(bool primary_hit);
	ReGIRHashCellDataSoADevice& get_hash_cell_data_device_soa(ReGIRSettings& regir_settings, bool primary_hit);
	OrochiBuffer<float>& get_non_canonical_factors(bool primary_hit);
	OrochiBuffer<float>& get_canonical_factors(bool primary_hit);
	ReGIRCellsAliasTablesSoAHost<OrochiBuffer>& get_cell_alias_tables(bool primary_hit);
	unsigned int& get_total_number_of_cells(bool primary_hit);
	unsigned int get_total_number_of_cells(bool primary_hit) const;

	unsigned int get_supersampling_current_frame() const;
	unsigned int get_supersampling_frames_available() const;

public:
	void reset_internal(bool primary_hit);
	bool pre_render_update_internal(HIPRTRenderData& render_data, bool primary_hit);
	bool try_rehash_internal(HIPRTRenderData& render_data, bool primary_hit);
	bool free_internal(bool primary_hit);

	ReGIRRenderPass* m_regir_render_pass = nullptr;

	ReGIRPresampledLightsSoAHost<OrochiBuffer> m_presampled_lights;

	// Buffer that contains the ReGIR grid. If temporal reuse is enabled,
	// this buffer will contain one more than one grid worth of space to
	// accomodate for the grid of the past frames for temporal reuse
	ReGIRHashGridSoAHost<OrochiBuffer> m_initial_reservoirs_primary_hits_grid;
	ReGIRHashGridSoAHost<OrochiBuffer> m_initial_reservoirs_secondary_hits_grid;
	ReGIRHashGridSoAHost<OrochiBuffer> m_spatial_output_primary_hits_grid;
	ReGIRHashGridSoAHost<OrochiBuffer> m_spatial_output_secondary_hits_grid;

	// For filling the grid asynchronously, we sometimes (depending on the spatial reuse settings etc...)
	// need another buffer to store the results of the async compute without overriding the buffers
	// that the path tracing kernels are currently using to shade
	ReGIRHashGridSoAHost<OrochiBuffer> m_async_compute_staging_buffer_primary_hits;
	ReGIRHashGridSoAHost<OrochiBuffer> m_async_compute_staging_buffer_secondary_hits;

	int m_correlation_reduction_current_grid_offset = 0;
	int m_correlation_reduction_frames_available = 0;
	ReGIRHashGridSoAHost<OrochiBuffer> m_correlation_reduction_grid_primary_hits;

	// Stores the pre-integrated RIS integral for each cell in the grid
	OrochiBuffer<float> m_non_canonical_pre_integration_factors_primary_hits;
	OrochiBuffer<float> m_non_canonical_pre_integration_factors_secondary_hits;
	OrochiBuffer<float> m_canonical_pre_integration_factors_primary_hits;
	OrochiBuffer<float> m_canonical_pre_integration_factors_secondary_hits;

	ReGIRHashCellDataSoAHost<OrochiBuffer> m_hash_cell_data_primary_hits;
	ReGIRHashCellDataSoAHost<OrochiBuffer> m_hash_cell_data_secondary_hits;

	ReGIRCellsAliasTablesSoAHost<OrochiBuffer> m_cells_alias_tables_primary_hits;
	ReGIRCellsAliasTablesSoAHost<OrochiBuffer> m_cells_alias_tables_secondary_hits;

	float m_current_grid_min_cell_size = 0.0f;
	float m_grid_cell_target_projected_size = 0.0f;

	unsigned int m_total_number_of_cells_primary_hits = 0;
	unsigned int m_total_number_of_cells_secondary_hits = 0;
};

#endif
