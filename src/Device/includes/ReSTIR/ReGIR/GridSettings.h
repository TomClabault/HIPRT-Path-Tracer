/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_GRID_SETTINGS_H
#define DEVICE_INCLUDES_REGIR_GRID_SETTINGS_H

struct ReGIRGridSettings
{
	float3 grid_origin;
	// "Length" of the grid in each X, Y, Z axis directions
	float3 extents;

	static constexpr int DEFAULT_GRID_SIZE = 32;
	// int3 grid_resolution = make_int3(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE);
	int3 grid_resolution = make_int3(96, 32, 96);

	// Some precomputed stuff
	unsigned int m_total_number_of_cells = 0;
	unsigned int m_total_number_of_reservoirs = 0;
	unsigned int m_number_of_reservoirs_per_cell = 0;
	unsigned int m_number_of_reservoirs_per_grid = 0;
	float3 m_cell_size = make_float3(0.0f, 0.0f, 0.0f);
	float m_cell_diagonal_length = 0.0f;
};

#endif
