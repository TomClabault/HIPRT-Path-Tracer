/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_HASH_GRID_STORAGE_H
#define REGIR_HASH_GRID_STORAGE_H

#include "HostDeviceCommon/RenderData.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRHashGridSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRHashCellDataSoAHost.h"

class ReGIRRenderPass;

class ReGIRHashGridStorage
{
public:
	static constexpr unsigned int DEFAULT_GRID_CELL_COUNT = 10000;

	void set_regir_render_pass(ReGIRRenderPass* regir_render_pass);

	unsigned int get_total_number_of_cells() const;
	std::size_t get_byte_size() const;

	bool pre_render_update(HIPRTRenderData& render_data);
	void post_sample_update_async(HIPRTRenderData& render_data);
	bool try_rehash(HIPRTRenderData& render_data);
	void reset();
	bool free();

	void to_device(HIPRTRenderData& render_data);

	ReGIRHashCellDataSoAHost<OrochiBuffer>& get_hash_cell_data_soa();
	unsigned int get_supersampling_current_frame() const;
	unsigned int get_supersampling_frames_available() const;

private:
	ReGIRRenderPass* m_regir_render_pass = nullptr;

	// Buffer that contains the ReGIR grid. If temporal reuse is enabled,
	// this buffer will contain one more than one grid worth of space to
	// accomodate for the grid of the past frames for temporal reuse
	ReGIRHashGridSoAHost<OrochiBuffer> m_grid_buffers;
	ReGIRHashGridSoAHost<OrochiBuffer> m_spatial_reuse_output_grid_buffer;

	int m_supersampling_curent_grid_offset = 0;
	int m_supersampling_frames_available = 0;
	ReGIRHashGridSoAHost<OrochiBuffer> m_supersample_grid;

	ReGIRHashCellDataSoAHost<OrochiBuffer> m_hash_cell_data;

	float m_current_grid_min_cell_size = 0.0f;
	float m_grid_cell_target_projected_size_ratio = 0.0f;

	unsigned int m_total_number_of_cells = 0;
};

#endif
