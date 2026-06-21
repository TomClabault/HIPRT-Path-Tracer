/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_HASH_GRID_HASH_H
#define DEVICE_INCLUDES_HASH_GRID_HASH_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Xorshift.h"

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

HIPRT_DEVICE static float3_t jitter_normal_in_tangent_plane(float3_t surface_normal, float3_t shading_point, float fuzzy_strength = 0.2f)
{
	// Getting the tangent plane vectors from the normal
	float3_t T, B;
	build_ONB(surface_normal, T, B);

	// Some deterministic random numbers from the position, in [-1, 1]
	float jitter_x = Xorshift32Generator(h2_xxhash32(shading_point.x * static_cast<float>(0xFFFFFFFF)))() * 2.0f - 1.0f;
	float jitter_y = Xorshift32Generator(h2_xxhash32(shading_point.y * static_cast<float>(0xFFFFFFFF)))() * 2.0f - 1.0f;

	// Jittering our normal in the tangent plane
	float3_t jittered = surface_normal + (T * jitter_x + B * jitter_y) * fuzzy_strength;

	// --- Step 4: renormalize ---
	return hippt::normalize(jittered);
}

HIPRT_DEVICE static float3_t jitter_world_position_tangent_plane(
	float3_t original_world_position, float3_t surface_normal, Xorshift32Generator& rng, float jittering_scaling, float jittering_radius = 0.5f)
{
	// Getting the tangent plane vectors from the normal
	float3_t T, B;
	build_ONB(surface_normal, T, B);

	// Offsets X and Y in the tangent plane
	float random_offset_x = rng() * 2.0f - 1.0f;
	float random_offset_y = rng() * 2.0f - 1.0f;

	// Scaling by the grid size
	float scaling = jittering_scaling * jittering_radius;
	random_offset_x *= scaling;
	random_offset_y *= scaling;

	return original_world_position + random_offset_x * T + random_offset_y * B;
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
 *
 * The output is precision * 3 bits, packed in the lower end of the uint
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
HIPRT_DEVICE static unsigned int hash_pos_distance_to_camera(float3_t world_position,
															 float3_t surface_normal,
															 const HIPRTCamera& current_camera,
															 float target_projected_size,
															 float grid_cell_min_size,
															 unsigned int hash_normal_precision,
															 unsigned int& out_checksum)
{
	float cell_size = compute_adaptive_cell_size(world_position, current_camera, target_projected_size, grid_cell_min_size);

	// Aliasing fix for the hash grid when our point is very close to the border of a cell
	world_position = hash_grid_aliasing_fix_clamping(world_position, cell_size);

	unsigned int grid_coord_x = static_cast<int>(floorf(world_position.x / cell_size));
	unsigned int grid_coord_y = static_cast<int>(floorf(world_position.y / cell_size));
	unsigned int grid_coord_z = static_cast<int>(floorf(world_position.z / cell_size));

	unsigned int normal_hashed = hash_quantize_normal(surface_normal, hash_normal_precision);

	// Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boisse, 2021]
	out_checksum = h2_xxhash32(cell_size + h2_xxhash32(grid_coord_z + h2_xxhash32(grid_coord_y + h2_xxhash32(grid_coord_x + h2_xxhash32(normal_hashed)))));

	unsigned int cell_hash = h1_pcg(cell_size + h1_pcg(grid_coord_z + h1_pcg(grid_coord_y + h1_pcg(grid_coord_x + h1_pcg(normal_hashed)))));

	return cell_hash;
}

HIPRT_DEVICE static unsigned int hash_double_position_camera(float3_t world_position_1,
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
	unsigned int cell_hash	 = h1_pcg(cell_hash_1 ^ cell_hash_2);

	return cell_hash;
}

HIPRT_DEVICE static unsigned int screen_space_gbuffer_hash(int pixel_x,
														   int pixel_y,
														   int screen_space_grid_cell_size,
														   float3_t world_position,
														   float3_t geometric_normal,
														   unsigned int normal_hash_precision,
														   float normal_jitter_strength,
														   unsigned int* out_checksum = nullptr)
{
	unsigned int grid_coord_x = pixel_x / screen_space_grid_cell_size;
	unsigned int grid_coord_y = pixel_y / screen_space_grid_cell_size;

	unsigned int hashed_normal =
		hash_quantize_normal(jitter_normal_in_tangent_plane(geometric_normal, world_position, normal_jitter_strength), normal_hash_precision);

	if (out_checksum != nullptr)
		*out_checksum = h2_xxhash32(grid_coord_x + h2_xxhash32(grid_coord_y + h2_xxhash32(hashed_normal)));
	return h1_pcg(grid_coord_x + h1_pcg(grid_coord_y + h1_pcg(hashed_normal)));
}

#endif
