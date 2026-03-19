/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_HASH_GRID_HASH_H
#define DEVICE_INCLUDES_HASH_GRID_HASH_H

#include "HostDeviceCommon/HIPRTCamera.h"

/**
 * PCG for the first hash function
 */
HIPRT_DEVICE static unsigned int h1_pcg(unsigned int seed)
{
	unsigned int state = seed * 747796405u + 2891336453u;
	unsigned int word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;

	return (word >> 22u) ^ word;
}

HIPRT_HOST_DEVICE static unsigned int h1_pcg(float seed)
{
	return h1_pcg(hippt::float_as_uint(seed));
}

/**
 * xxhash32 for the second hash function
 */
HIPRT_DEVICE static unsigned int h2_xxhash32(unsigned int seed)
{
	constexpr unsigned int PRIME32_2 = 2246822519U;
	constexpr unsigned int PRIME32_3 = 3266489917U;
	constexpr unsigned int PRIME32_4 = 668265263U;
	constexpr unsigned int PRIME32_5 = 374761393U;

	unsigned int h32 = seed + PRIME32_5;

	h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
	h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
	h32 = PRIME32_3 * (h32 ^ (h32 >> 13));

	return h32 ^ (h32 >> 16);
}

HIPRT_HOST_DEVICE static unsigned int h2_xxhash32(float seed)
{
	return h2_xxhash32(hippt::float_as_uint(seed));
}

/**
 * Reference: SIGGRAPH 2022 - Advances in Spatial Hashing
 */
HIPRT_HOST_DEVICE static float3_t hash_grid_aliasing_fix_periodic_shifting(float3_t base_position, float grid_cell_size)
{
	float scaling = 0.005f * grid_cell_size;

	constexpr float frequency_per_grid_cell			= 5.0f;
	constexpr float frequency_per_grid_cell_inverse = 1.0f / frequency_per_grid_cell;
	const float frequency							= 1.0f / (grid_cell_size * frequency_per_grid_cell_inverse);

	return make_float3(base_position.x + (hippt::intrin_cosf(base_position.z * frequency) + hippt::intrin_cosf(base_position.y * frequency)) * scaling * 0.5f,
					   base_position.y + (hippt::intrin_cosf(base_position.x * frequency) + hippt::intrin_cosf(base_position.z * frequency)) * scaling * 0.5f,
					   base_position.z + (hippt::intrin_cosf(base_position.y * frequency) + hippt::intrin_cosf(base_position.x * frequency)) * scaling * 0.5f);
}

HIPRT_HOST_DEVICE static float3_t hash_grid_aliasing_fix_clamping(float3_t base_position, float grid_cell_size)
{
	float grid_coord_x_frac = hippt::fract(base_position.x / grid_cell_size);
	float grid_coord_y_frac = hippt::fract(base_position.y / grid_cell_size);
	float grid_coord_z_frac = hippt::fract(base_position.z / grid_cell_size);

	// If the position is very close to the border of a cell, clamping the
	// position to the border of the cell
	float3_t new_position = base_position;
	if (grid_coord_x_frac < 1.0e-3f || grid_coord_x_frac > 0.999f)
		new_position.x = roundf(base_position.x / grid_cell_size) * grid_cell_size;
	if (grid_coord_y_frac < 1.0e-3f || grid_coord_y_frac > 0.999f)
		new_position.y = roundf(base_position.y / grid_cell_size) * grid_cell_size;
	if (grid_coord_z_frac < 1.0e-3f || grid_coord_z_frac > 0.999f)
		new_position.z = roundf(base_position.z / grid_cell_size) * grid_cell_size;

	return new_position;
}

/**
 * The 'precision' factor controls the discretization of the normal.
 * Higher values mean more discretization steps mean more precision.
 *
 * 2 is a default good value for 'precision'
 */
HIPRT_HOST_DEVICE static unsigned int hash_quantize_normal(float3_t normal, unsigned int precision)
{
	float precision_f = precision;

	unsigned int x = static_cast<unsigned int>(normal.x * precision_f) << (2 * precision);
	unsigned int y = static_cast<unsigned int>(normal.y * precision_f) << (1 * precision);
	unsigned int z = static_cast<unsigned int>(normal.z * precision_f);

	return x | y | z;
}

/**
 * Reference: [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boisse, 2021]
 */
HIPRT_DEVICE static float compute_adaptive_cell_size(float3_t world_position,
													 const HIPRTCamera& current_camera,
													 float target_projected_size,
													 float grid_cell_min_size)
{
	int width  = current_camera.sensor_width;
	int height = current_camera.sensor_height;

	float cell_size_step = hippt::length(world_position - current_camera.position) *
						   tanf(target_projected_size * current_camera.vertical_fov * hippt::max(1.0f / height, (float)height / hippt::square(width)));
	float log_step = floorf(log2f(cell_size_step / grid_cell_min_size));

	return hippt::max(grid_cell_min_size, grid_cell_min_size * exp2f(log_step));
}

/**
 * Returns the hash cell index of the given world position and camera position. Does not resolve collisions.
 * The hash key for resolving collision is given in 'out_checksum'
 */
HIPRT_DEVICE static unsigned int hash_pos_distance_to_camera(unsigned int total_number_of_cells,
															 float3_t world_position,
															 const HIPRTCamera& current_camera,
															 float target_projected_size,
															 float grid_cell_min_size,
															 unsigned int& out_checksum)
{
	float cell_size = compute_adaptive_cell_size(world_position, current_camera, target_projected_size, grid_cell_min_size);

	// Aliasing fix for the hash grid when our point is very close to the border of a cell
	world_position = hash_grid_aliasing_fix_clamping(world_position, cell_size);

	unsigned int grid_coord_x = static_cast<int>(floorf(world_position.x / cell_size));
	unsigned int grid_coord_y = static_cast<int>(floorf(world_position.y / cell_size));
	unsigned int grid_coord_z = static_cast<int>(floorf(world_position.z / cell_size));

	// Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boisse, 2021]
	out_checksum = h2_xxhash32(cell_size + h2_xxhash32(grid_coord_z + h2_xxhash32(grid_coord_y + h2_xxhash32(grid_coord_x))));

	unsigned int cell_hash = h1_pcg(cell_size + h1_pcg(grid_coord_z + h1_pcg(grid_coord_y + h1_pcg(grid_coord_x)))) % total_number_of_cells;

	return cell_hash;
}

HIPRT_DEVICE static unsigned int hash_double_position_camera(unsigned int total_number_of_cells,
															 float3_t world_position_1,
															 float3_t world_position_2,
															 const HIPRTCamera& current_camera,
															 float target_projected_size,
															 float grid_cell_min_size,
															 unsigned int& out_checksum)
{
	float cell_size_1 = compute_adaptive_cell_size(world_position_1, current_camera, target_projected_size, grid_cell_min_size);
	float cell_size_2 = compute_adaptive_cell_size(world_position_2, current_camera, target_projected_size, grid_cell_min_size);

	// Aliasing fix for the hash grid when our point is very close to the border of a cell
	world_position_1 = hash_grid_aliasing_fix_clamping(world_position_1, cell_size_1);
	world_position_2 = hash_grid_aliasing_fix_clamping(world_position_2, cell_size_2);

	unsigned int grid_coord_x_1 = static_cast<int>(floorf(world_position_1.x / cell_size_1));
	unsigned int grid_coord_y_1 = static_cast<int>(floorf(world_position_1.y / cell_size_1));
	unsigned int grid_coord_z_1 = static_cast<int>(floorf(world_position_1.z / cell_size_1));

	unsigned int grid_coord_x_2 = static_cast<int>(floorf(world_position_2.x / cell_size_2));
	unsigned int grid_coord_y_2 = static_cast<int>(floorf(world_position_2.y / cell_size_2));
	unsigned int grid_coord_z_2 = static_cast<int>(floorf(world_position_2.z / cell_size_2));

	// Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boisse, 2021]
	unsigned int hash_1 = h2_xxhash32(cell_size_1 + h2_xxhash32(grid_coord_z_1 + h2_xxhash32(grid_coord_y_1 + h2_xxhash32(grid_coord_x_1))));
	unsigned int hash_2 = h2_xxhash32(cell_size_2 + h2_xxhash32(grid_coord_z_2 + h2_xxhash32(grid_coord_y_2 + h2_xxhash32(grid_coord_x_2))));
	out_checksum		= h2_xxhash32(hash_1 ^ hash_2);

	unsigned int cell_hash_1 = h1_pcg(cell_size_1 + h1_pcg(grid_coord_z_1 + h1_pcg(grid_coord_y_1 + h1_pcg(grid_coord_x_1))));
	unsigned int cell_hash_2 = h1_pcg(cell_size_2 + h1_pcg(grid_coord_z_2 + h1_pcg(grid_coord_y_2 + h1_pcg(grid_coord_x_2))));
	unsigned int cell_hash	 = h1_pcg(cell_hash_1 ^ cell_hash_2) % total_number_of_cells;

	return cell_hash;
}

HIPRT_DEVICE static unsigned int screen_space_gbuffer_hash(
						int pixel_x, int pixel_y, int grid_cell_size, float3_t world_position, float3_t geometric_normal, const HIPRTCamera& current_camera)
{
	unsigned int grid_coord_x  = pixel_x / grid_cell_size;
	unsigned int grid_coord_y  = pixel_y / grid_cell_size;
	unsigned int hashed_normal = hash_quantize_normal(geometric_normal, 2);

	float distance_to_camera				  = hippt::length(world_position - current_camera.position);
	unsigned int distance_to_camera_quantized = 0; // static_cast<int>(distance_to_camera / 0.3f);

	return h1_pcg(grid_coord_x + h1_pcg(grid_coord_y + h1_pcg(hashed_normal + h1_pcg(distance_to_camera_quantized))));
}

#endif
