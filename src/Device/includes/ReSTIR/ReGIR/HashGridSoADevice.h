/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_DEVICE_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_DEVICE_H

#include "Device/includes/ReSTIR/ReGIR/ReservoirSoA.h"

struct ReGIRHashGridSoADevice
{
	// These two SoAs are allocated to hold 'number_cells * number_reservoirs_per_cell'
	// So for a given 'hash_grid_cell_index', the cell contains reservoirs and samples going from 
	// reservoirs[hash_grid_cell_index * number_reservoirs_per_cell] to reservoirs[cell_index * number_reservoirs_per_cell + number_reservoirs_per_cell[
	ReGIRReservoirSoADevice reservoirs;
	ReGIRSampleSoADevice samples;

	unsigned int m_total_number_of_cells = 0;
};

#endif // DEVICE_INCLUDES_REGIR_HASH_GRID_SOA_DEVICE_H
