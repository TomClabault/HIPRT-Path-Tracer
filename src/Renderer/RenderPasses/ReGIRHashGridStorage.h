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
	void set_regir_render_pass(ReGIRRenderPass* regir_render_pass);

	unsigned int get_total_number_of_cells() const;
	std::size_t get_byte_size() const;

	bool pre_render_update(HIPRTRenderData& render_data);
	void reset();
	bool free();

	void to_device(HIPRTRenderData& render_data);

// private:

	ReGIRRenderPass* m_regir_render_pass = nullptr;

	// Buffer that contains the ReGIR grid. If temporal reuse is enabled,
	// this buffer will contain one more than one grid worth of space to
	// accomodate for the grid of the past frames for temporal reuse
	ReGIRHashGridSoAHost<OrochiBuffer> m_grid_buffers;
	ReGIRHashGridSoAHost<OrochiBuffer> m_spatial_reuse_output_grid_buffer;
	ReGIRHashCellDataSoAHost<OrochiBuffer> m_hash_cell_data;

	float3 m_current_grid_resolution = make_float3(ReGIRHashGridSoADevice::DEFAULT_GRID_SIZE, ReGIRHashGridSoADevice::DEFAULT_GRID_SIZE, ReGIRHashGridSoADevice::DEFAULT_GRID_SIZE);
	unsigned int m_total_number_of_cells = 0;
	unsigned int m_hash_grid_current_overallocation_factor = 1000;
};

#endif
