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
HIPRT_DEVICE HIPRT_INLINE unsigned int h1_pcg(unsigned int seed)
{
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    
    return (word >> 22u) ^ word;
}

HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int h1_pcg(float seed)
{
    return h1_pcg(hippt::float_as_uint(seed));
}

/**
 * xxhash32 for the second hash function
 */
HIPRT_DEVICE HIPRT_INLINE unsigned int h2_xxhash32(unsigned int seed)
{
    constexpr unsigned int PRIME32_2 = 2246822519U;
    constexpr unsigned int PRIME32_3 = 3266489917U;
    constexpr unsigned int PRIME32_4 = 668265263U;
    constexpr unsigned int PRIME32_5 = 374761393U;

    unsigned int h32 = seed + PRIME32_5;

    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));

    return h32^(h32 >> 16);
}

HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int h2_xxhash32(float seed)
{
    return h2_xxhash32(hippt::float_as_uint(seed));
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 hash_periodic_shifting(float3 base_position, float grid_cell_size)
{
    float scaling = 0.1f * grid_cell_size;

    constexpr float frequency_per_grid_cell = 5.0f;
    constexpr float frequency_per_grid_cell_inverse = 1.0f / frequency_per_grid_cell;

    return make_float3(
        base_position.x + cosf(base_position.z / (grid_cell_size * frequency_per_grid_cell_inverse)) * scaling,
        base_position.y + cosf(base_position.x / (grid_cell_size * frequency_per_grid_cell_inverse)) * scaling,
        base_position.z + cosf(base_position.y / (grid_cell_size * frequency_per_grid_cell_inverse)) * scaling);
}

 /**
 * Reference: [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boissé, 2021]
 */
HIPRT_DEVICE HIPRT_INLINE float compute_adaptive_cell_size(float3 world_position, const HIPRTCamera& current_camera, float target_projected_size, float grid_cell_min_size)
{
    int width = current_camera.sensor_width;
    int height = current_camera.sensor_height;

    float cell_size_step = hippt::length(world_position - current_camera.position) * tanf(target_projected_size * current_camera.vertical_fov * hippt::max(1.0f / height, (float)height / hippt::square(width)));
    float log_step = floorf(log2f(cell_size_step / grid_cell_min_size));

    return hippt::max(grid_cell_min_size, grid_cell_min_size * exp2f(log_step));
}

/**
 * Returns the hash cell index of the given world position and camera position. Does not resolve collisions.
 * The hash key for resolving collision is given in 'out_checksum'
 */
HIPRT_DEVICE HIPRT_INLINE unsigned int hash_pos_distance_to_camera(unsigned int total_number_of_cells, float3 world_position, const HIPRTCamera& current_camera, float target_projected_size, float grid_cell_min_size, unsigned int& out_checksum)
{
    float cell_size = compute_adaptive_cell_size(world_position, current_camera, target_projected_size, grid_cell_min_size);

    // Periodic shifting to avoid float precision issues when, for example, rays hit a surface
    // that is perfectly at Y=0 (the floor of the scene for example).
    // 
    // In that example, because of float imprecisions, rays hitting the floor will never have
    // a y=0 hit coordinate but rather be slightly negative or slightly positive, depending
    // on float imprecisions and this will actually create some noisy patterns where random rays access the hash 
    // grid cell that has Y-negative and some other randoms rays access the Y-positive hash grid cell
    //
    // Reference: SIGGRAPH 2022 - Advances in Spatial Hashing
    world_position = hash_periodic_shifting(world_position, cell_size);

    unsigned int grid_coord_x = static_cast<int>(floorf(world_position.x / cell_size));
    unsigned int grid_coord_y = static_cast<int>(floorf(world_position.y / cell_size));
    unsigned int grid_coord_z = static_cast<int>(floorf(world_position.z / cell_size));

    // Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boissé, 2021]
    out_checksum = h2_xxhash32(cell_size + h2_xxhash32(grid_coord_z + h2_xxhash32(grid_coord_y + h2_xxhash32(grid_coord_x))));
    
    unsigned int cell_hash = h1_pcg(cell_size + h1_pcg(grid_coord_z + h1_pcg(grid_coord_y + h1_pcg(grid_coord_x)))) % total_number_of_cells;

    return cell_hash;
}

HIPRT_DEVICE HIPRT_INLINE unsigned int hash_double_position_camera(unsigned int total_number_of_cells, float3 world_position_1, float3 world_position_2, const HIPRTCamera& current_camera, float target_projected_size, float grid_cell_min_size, unsigned int& out_checksum)
{
    float cell_size_1 = compute_adaptive_cell_size(world_position_1, current_camera, target_projected_size, grid_cell_min_size);
    float cell_size_2 = compute_adaptive_cell_size(world_position_2, current_camera, target_projected_size, grid_cell_min_size);

    // Periodic shifting to avoid float precision issues when, for example, rays hit a surface
    // that is perfectly at Y=0 (the floor of the scene for example).
    // 
    // In that example, because of float imprecisions, rays hitting the floor will never have
    // a y=0 hit coordinate but rather be slightly negative or slightly positive, depending
    // on float imprecisions and this will actually create some noisy patterns where random rays access the hash 
    // grid cell that has Y-negative and some other randoms rays access the Y-positive hash grid cell
    //
    // Reference: SIGGRAPH 2022 - Advances in Spatial Hashing
    world_position_1 = hash_periodic_shifting(world_position_1, cell_size_1);
    world_position_2 = hash_periodic_shifting(world_position_2, cell_size_2);

    unsigned int grid_coord_x_1 = static_cast<int>(floorf(world_position_1.x / cell_size_1));
    unsigned int grid_coord_y_1 = static_cast<int>(floorf(world_position_1.y / cell_size_1));
    unsigned int grid_coord_z_1 = static_cast<int>(floorf(world_position_1.z / cell_size_1));

    unsigned int grid_coord_x_2 = static_cast<int>(floorf(world_position_2.x / cell_size_2));
    unsigned int grid_coord_y_2 = static_cast<int>(floorf(world_position_2.y / cell_size_2));
    unsigned int grid_coord_z_2 = static_cast<int>(floorf(world_position_2.z / cell_size_2));

    // Using two hash functions as proposed in [WORLD-SPACE SPATIOTEMPORAL RESERVOIR REUSE FOR RAY-TRACED GLOBAL ILLUMINATION, Boissé, 2021]
    unsigned int hash_1 = h2_xxhash32(cell_size_1 + h2_xxhash32(grid_coord_z_1 + h2_xxhash32(grid_coord_y_1 + h2_xxhash32(grid_coord_x_1))));
    unsigned int hash_2 = h2_xxhash32(cell_size_2 + h2_xxhash32(grid_coord_z_2 + h2_xxhash32(grid_coord_y_2 + h2_xxhash32(grid_coord_x_2))));
    out_checksum = h2_xxhash32(hash_1 ^ hash_2);
    
    unsigned int cell_hash_1 = h1_pcg(cell_size_1 + h1_pcg(grid_coord_z_1 + h1_pcg(grid_coord_y_1 + h1_pcg(grid_coord_x_1))));
    unsigned int cell_hash_2 = h1_pcg(cell_size_2 + h1_pcg(grid_coord_z_2 + h1_pcg(grid_coord_y_2 + h1_pcg(grid_coord_x_2))));
    unsigned int cell_hash = h1_pcg(cell_hash_1 ^ cell_hash_2) % total_number_of_cells;

    return cell_hash;
}

#endif
