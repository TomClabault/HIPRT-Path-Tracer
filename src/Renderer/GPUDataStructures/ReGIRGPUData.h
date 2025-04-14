/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GPU_DATA_STRUCTURES_REGIR_GPU_DATA_H
#define RENDERER_GPU_DATA_STRUCTURES_REGIR_GPU_DATA_H

#include "Device/includes/ReSTIR/ReGIR/Grid.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

#include "HIPRT-Orochi/OrochiBuffer.h"

struct ReGIRGPUData
{
	int get_number_of_cells() const
	{
		return gpu_grid.grid_resolution.x * gpu_grid.grid_resolution.y * gpu_grid.grid_resolution.z;
	}

	ReGIRGrid gpu_grid;
	ReGIRSettings settings;

	OrochiBuffer<ReGIRReservoir> grid_buffer;
};

#endif
