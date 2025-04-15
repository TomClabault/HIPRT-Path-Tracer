/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GPU_DATA_STRUCTURES_REGIR_GPU_DATA_H
#define RENDERER_GPU_DATA_STRUCTURES_REGIR_GPU_DATA_H

#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/includes/ReSTIR/ReGIR/Reservoir.h"

#include "HIPRT-Orochi/OrochiBuffer.h"

struct ReGIRGPUData
{
	int get_number_of_reservoirs_in_grid(HIPRTRenderData& render_data) const
	{
		ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;
		return regir_settings.grid_resolution.x * regir_settings.grid_resolution.y * regir_settings.grid_resolution.z * regir_settings.reservoirs_count_per_grid_cell;
	}

	OrochiBuffer<ReGIRReservoir> grid_buffer;
};

#endif
