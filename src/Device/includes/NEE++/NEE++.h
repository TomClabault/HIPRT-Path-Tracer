/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEE_PLUS_PLUS
#define DEVICE_INCLUDES_NEE_PLUS_PLUS

/**
 * Structure that contains the data for the implementation of NEE++.
 * 
 * Reference:
 * [1] [Next Event Estimation++: Visibility Mapping for Efficient Light Transport Simulation]
 */
struct NEEPlusPlus
{
	static constexpr int NEE_PLUS_PLUS_DEFAULT_GRID_SIZE = 24;

	int3 grid_dimensions = make_int3(NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEE_PLUS_PLUS_DEFAULT_GRID_SIZE, NEE_PLUS_PLUS_DEFAULT_GRID_SIZE);

	// Bottom left corner and top right corner of the 3D grid in world space
	float3 grid_origin, grid_max_point;

	// Linear buffer that stores the number of rays that were
	// computed as non-occluded from voxel to voxel in the scene.
	//
	// For example, if 16 rays were shot from one voxel to another
	// and 7 of these rays were found to be unoccluded, then the corresponding
	// entry in the map will contain the value 7
	//
	// Because the visibility map is symmetrical, this is a linear buffer that contains
	// only half of the visibility matrix
	//
	// For the indexing logic, (0, 0) is in the top left corner of the matrix
	AtomicType<int>* visibility_map;

	// Same as 'visibility_map' but stores how many rays were traced in total from one
	// voxel to another. In the example from above, this would contain the value 16.
	//
	// For the indexing logic, (0, 0) is in the top left corner of the matrix
	AtomicType<int>* visibility_map_count;

	// How many elements are in 'visibility_map' and 'visibility_map_count'
	unsigned int map_size = NEE_PLUS_PLUS_DEFAULT_GRID_SIZE * NEE_PLUS_PLUS_DEFAULT_GRID_SIZE * NEE_PLUS_PLUS_DEFAULT_GRID_SIZE / 2;

	HIPRT_HOST_DEVICE void accumulate_visibility(float3 first_world_position, float3 second_world_position, bool visible)
	{
		int matrix_index = get_visibility_map_index(first_world_position, second_world_position);
		if (matrix_index == -1)
			// One of the two points was outside the scene, cannot cache this
			return;

		if (visible)
			hippt::atomic_add(&visibility_map[matrix_index], 1);
		hippt::atomic_add(&visibility_map_count[matrix_index], 1);
	}

private:
	/**
	 * Returns the index of the voxel of the given position in [grid_dimensions.x - 1, grid_dimensions.y - 1, grid_dimensions.z - 1]
	 */
	HIPRT_HOST_DEVICE int3 get_voxel_3D_index(float3 position)
	{
		float3 position_grid_space = position - grid_origin;
		float3 voxel_index_float = position_grid_space / (grid_max_point - grid_origin);
		if (voxel_index_float.x > 1.0f || voxel_index_float.y > 1.0f || voxel_index_float.z > 1.0f)
			// The point is outside the scene
			return make_int3(-1, -1, -1);

		voxel_index_float = hippt::min(0.999f, voxel_index_float);

		int3 voxel_index_int = make_int3(static_cast<int>(voxel_index_float.x * grid_dimensions.x),
										 static_cast<int>(voxel_index_float.y * grid_dimensions.y),
										 static_cast<int>(voxel_index_float.z * grid_dimensions.z));

		return voxel_index_int;
	}

	HIPRT_HOST_DEVICE int get_visibility_map_index(float3 first_world_position, float3 second_world_position)
	{
		int3 first_pos_voxel_3D_index = get_voxel_3D_index(first_world_position);
		int3 second_pos_voxel_3D_index = get_voxel_3D_index(second_world_position);

		if (first_pos_voxel_3D_index.x == -1 || second_pos_voxel_3D_index.x == -1)
			// One of the two points is outside the scene, cannot cache this
			return -1;

		int first_voxel_index = first_pos_voxel_3D_index.x + first_pos_voxel_3D_index.y * grid_dimensions.x + first_pos_voxel_3D_index.z * grid_dimensions.y * grid_dimensions.x;
		int second_voxel_index = second_pos_voxel_3D_index.x + second_pos_voxel_3D_index.y * grid_dimensions.x + second_pos_voxel_3D_index.z * grid_dimensions.y * grid_dimensions.x;

		// Because the visibility matrix is symmetrical, we're only using half of it to save half the VRAM
		// Thus we need to find the index in the lower half of the visiblity matrix
		if (first_voxel_index > second_voxel_index)
		{
			// The first voxel index is used to index the matrix on the X axis
			// 
			// If the first voxel is past the diagonal (from top left corner to bottom right)
			// of the matrix, i.e. in the top right half of the matrix, we're going to flip
			// it about the diagonal because we're only using the bottom left half of the matrix
			// (since the matrix is symmetrical about the diagonal)
			int temp = first_voxel_index;
			first_voxel_index = second_voxel_index;
			second_voxel_index = first_voxel_index;
		}

		return first_voxel_index + second_voxel_index * (map_size * 2);
	}
};

#endif
