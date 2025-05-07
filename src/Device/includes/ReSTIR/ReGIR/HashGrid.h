/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_H

struct ReGIRHashGrid
{
	//HIPRT_DEVICE int get_hash_grid_cell_index_from_world_pos_no_collision_resolve(float3 world_position) const
	//{
	//	float3 position_in_grid = world_position - grid_origin;
	//	float3 position_in_grid_cell_unit = position_in_grid / get_cell_size();

	//	int3 cell_xyz = make_int3(static_cast<int>(position_in_grid_cell_unit.x), static_cast<int>(position_in_grid_cell_unit.y), static_cast<int>(position_in_grid_cell_unit.z));
	//	// If a point is on the very edge of the grid, we're going to have one of the coordinates be 'grid_resolution.XXX'
	//	// exactly, 16 for a grid resolution of 16 for example. 
	//	// 
	//	// But that's out of bounds because our grid cells are in [0, 15] so we're subing 
	//	cell_xyz = hippt::min(cell_xyz, grid_resolution - make_int3(1, 1, 1));

	//	return cell_xyz.x + cell_xyz.y * grid_resolution.x + cell_xyz.z * grid_resolution.x * grid_resolution.y;
	//}

	/*HIPRT_DEVICE float3 get_cell_center_from_world_pos(float3 world_point, float3 camera_position) const
	{
		return get_cell_center_from_hash_grid_cell_index(get_hash_grid_cell_index_from_world_pos_no_collision_resolve(world_point, camera_position));
	}*/

	//HIPRT_DEVICE int get_hash_grid_cell_index_from_xyz(int3 xyz_cell_index) const
	//{
	//	if (xyz_cell_index.x < 0 || xyz_cell_index.x >= grid_resolution.x
	//		|| xyz_cell_index.y < 0 || xyz_cell_index.y >= grid_resolution.y
	//		|| xyz_cell_index.z < 0 || xyz_cell_index.z >= grid_resolution.z)
	//		// Outside of the grid
	//		return -1;

	//	return xyz_cell_index.x + xyz_cell_index.y * grid_resolution.x + xyz_cell_index.z * grid_resolution.x * grid_resolution.y;
	//}

	// float3 grid_origin;
	// // "Length" of the grid in each X, Y, Z axis directions
	// float3 extents;

	static constexpr int DEFAULT_GRID_SIZE = 4;
	int3 grid_resolution = make_int3(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE);

	// Some precomputed stuff
	//unsigned int m_total_number_of_reservoirs = 0;
	//unsigned int m_number_of_reservoirs_per_cell = 0;
	//unsigned int m_number_of_reservoirs_per_grid = 0;
	//float m_cell_diagonal_length = 0.0f;
};

#endif
