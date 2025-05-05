/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_HASH_GRID_H
#define DEVICE_INCLUDES_REGIR_HASH_GRID_H

struct ReGIRHashGrid
{
	HIPRT_DEVICE unsigned int hash(float3 world_position, float3 camera_position) const
	{
		float3 relative_to_camera = world_position - camera_position;

		constexpr unsigned int p1 = 73856093;
		constexpr unsigned int p2 = 19349663;
		constexpr unsigned int p3 = 83492791;

		unsigned int x = relative_to_camera.x / 0.3f;
		unsigned int y = relative_to_camera.y / 0.3f;
		unsigned int z = relative_to_camera.z / 0.3f;

		return ((x * p1) ^ (y * p2) ^ (z * p3)) % m_total_number_of_cells;
	}

	HIPRT_DEVICE unsigned int get_linear_cell_index_from_world_pos(float3 world_position, float3 camera_position) const
	{
		return hash(world_position, camera_position);
	}

	HIPRT_DEVICE float3 get_cell_size(float3 world_position = make_float3(0, 0, 0), float3 camera_position = make_float3(0, 0, 0)) const
	{
		return m_cell_size;
	}

	HIPRT_DEVICE float get_cell_diagonal_length() const
	{
		return m_cell_diagonal_length;
	}

	HIPRT_DEVICE int3 get_xyz_cell_index_from_linear(int linear_cell_index) const
	{
		int index_x = linear_cell_index % grid_resolution.x;
		int index_y = (linear_cell_index % (grid_resolution.x * grid_resolution.y)) / grid_resolution.x;
		int index_z = linear_cell_index / (grid_resolution.x * grid_resolution.y);

		return make_int3(index_x, index_y, index_z);
	}

	HIPRT_DEVICE float3 get_cell_origin_from_linear_cell_index(int linear_cell_index) const
	{
		float3 cell_size = get_cell_size();

		int3 cell_index_xyz = get_xyz_cell_index_from_linear(linear_cell_index);
		float3 cell_index_xyz_float = make_float3(static_cast<float>(cell_index_xyz.x), static_cast<float>(cell_index_xyz.y), static_cast<float>(cell_index_xyz.z));

		return grid_origin + cell_size * cell_index_xyz_float;
	}

	HIPRT_DEVICE float3 get_cell_center_from_linear_cell_index(unsigned int linear_cell_index) const
	{
		float3 cell_size = get_cell_size();

		return get_cell_origin_from_linear_cell_index(linear_cell_index) + cell_size * 0.5f;
	}

	HIPRT_DEVICE float3 jitter_world_position(float3 original_world_position, Xorshift32Generator& rng) const
	{
		return original_world_position + (make_float3(rng(), rng(), rng()) * 2.0f - make_float3(1.0f, 1.0f, 1.0f)) * get_cell_size() * 0.5f;
	}

	HIPRT_DEVICE int get_linear_cell_index_from_world_pos(float3 world_position) const
	{
		float3 position_in_grid = world_position - grid_origin;
		float3 position_in_grid_cell_unit = position_in_grid / get_cell_size();

		int3 cell_xyz = make_int3(static_cast<int>(position_in_grid_cell_unit.x), static_cast<int>(position_in_grid_cell_unit.y), static_cast<int>(position_in_grid_cell_unit.z));
		// If a point is on the very edge of the grid, we're going to have one of the coordinates be 'grid_resolution.XXX'
		// exactly, 16 for a grid resolution of 16 for example. 
		// 
		// But that's out of bounds because our grid cells are in [0, 15] so we're subing 
		cell_xyz = hippt::min(cell_xyz, grid_resolution - make_int3(1, 1, 1));

		return cell_xyz.x + cell_xyz.y * grid_resolution.x + cell_xyz.z * grid_resolution.x * grid_resolution.y;
	}

	HIPRT_DEVICE float3 get_cell_center_from_world_pos(float3 world_point) const
	{
		return get_cell_center_from_linear_cell_index(get_linear_cell_index_from_world_pos(world_point));
	}

	HIPRT_DEVICE int get_linear_cell_index_from_xyz(int3 xyz_cell_index) const
	{
		if (xyz_cell_index.x < 0 || xyz_cell_index.x >= grid_resolution.x
			|| xyz_cell_index.y < 0 || xyz_cell_index.y >= grid_resolution.y
			|| xyz_cell_index.z < 0 || xyz_cell_index.z >= grid_resolution.z)
			// Outside of the grid
			return -1;

		return xyz_cell_index.x + xyz_cell_index.y * grid_resolution.x + xyz_cell_index.z * grid_resolution.x * grid_resolution.y;
	}

	float3 grid_origin;
	// "Length" of the grid in each X, Y, Z axis directions
	float3 extents;

	static constexpr int DEFAULT_GRID_SIZE = 64;
	int3 grid_resolution = make_int3(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE);

	// Some precomputed stuff
	unsigned int m_total_number_of_cells = 0;
	unsigned int m_total_number_of_reservoirs = 0;
	unsigned int m_number_of_reservoirs_per_cell = 0;
	unsigned int m_number_of_reservoirs_per_grid = 0;
	float3 m_cell_size = make_float3(0.0f, 0.0f, 0.0f);
	float m_cell_diagonal_length = 0.0f;
};

#endif
