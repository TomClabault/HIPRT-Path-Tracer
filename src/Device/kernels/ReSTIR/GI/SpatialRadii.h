/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_GI_COMPUTE_SPATIAL_RADII_H
#define KERNELS_RESTIR_GI_COMPUTE_SPATIAL_RADII_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/UtilsSpatial.h"

#include "HostDeviceCommon/RenderData.h"

#define NB_DIRECTIONS 32
#define NB_SAMPLES_PER_DIRECTION 16
#define MINIMUM_RADIUS_SIZE 5

HIPRT_HOST_DEVICE float get_distance_from_center_from_sample_index(int sample_index, float max_spatial_radius)
{
    return MINIMUM_RADIUS_SIZE + (max_spatial_radius - MINIMUM_RADIUS_SIZE) * ((sample_index + 1.0f) / NB_SAMPLES_PER_DIRECTION);
}

HIPRT_HOST_DEVICE int get_spatial_sample_index(int2 center_pixel_coords, int direction_index, int sample_index, float max_spatial_radius, int2 render_resolution, Xorshift32Generator& DEBUG_RNG)
{
    float2 neighbor_offset_in_disk;
    neighbor_offset_in_disk.x = get_distance_from_center_from_sample_index(sample_index, max_spatial_radius);
    neighbor_offset_in_disk.y = 0; // Always 0 for y, we're just generating the position on a line here and then we're going to rotate it around the circle

    // 2D rotation matrix: https://en.wikipedia.org/wiki/Rotation_matrix
    float theta = (direction_index / static_cast<float>(NB_DIRECTIONS)) * M_TWO_PI;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    float2 neighbor_offset_rotated = make_float2(neighbor_offset_in_disk.x * cos_theta - neighbor_offset_in_disk.y * sin_theta, neighbor_offset_in_disk.x * sin_theta + neighbor_offset_in_disk.y * cos_theta);

    int2 neighbor_offset_int = make_int2(static_cast<int>(neighbor_offset_rotated.x), static_cast<int>(neighbor_offset_rotated.y));
	int2 neighbor_pixel_coords = center_pixel_coords + neighbor_offset_int;

	if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= render_resolution.x ||
		neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= render_resolution.y)
		// Rejecting the sample if it's outside of the viewport
		return -1;

	return neighbor_pixel_coords.x + neighbor_pixel_coords.y * render_resolution.x;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_Spatial_Radii(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_Spatial_Radii(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t center_pixel_index = x + y * render_data.render_settings.render_resolution.x;

    if (!render_data.aux_buffers.pixel_active[center_pixel_index])
        // Pixel isn't active because of adaptive sampling or render resolution scaling
        return;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(center_pixel_index + 1);
    else
        seed = wang_hash((center_pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
    Xorshift32Generator random_number_generator(seed);

    // Clearing previous data
    render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask[center_pixel_index] = 0;
    render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius[center_pixel_index] = 0;

    float3 center_shading_point = render_data.g_buffer.primary_hit_position[center_pixel_index];
    float3 center_shading_normal = render_data.g_buffer.shading_normals[center_pixel_index].unpack();

    unsigned char valid_sample_per_direction[NB_DIRECTIONS] = { 0 };
    unsigned char max_radius_per_direction[NB_DIRECTIONS] = { 0 };
    for (int direction_index = 0; direction_index < NB_DIRECTIONS; direction_index++)
    {
        for (int sample_index = 0; sample_index < NB_SAMPLES_PER_DIRECTION; sample_index++)
        {
            int spatial_index = get_spatial_sample_index(make_int2(x, y), direction_index, sample_index, render_data.render_settings.restir_gi_settings.common_spatial_pass.reuse_radius, render_data.render_settings.render_resolution, random_number_generator);
            if (spatial_index == -1)
                continue;

            bool sample_is_similar = check_neighbor_similarity_heuristics<true>(render_data, spatial_index, center_pixel_index, center_shading_point, center_shading_normal);
            if (sample_is_similar)
            {
                valid_sample_per_direction[direction_index]++;
                max_radius_per_direction[direction_index] = sample_index;
            }

        }
    }

    unsigned int valid_directions = 0;
    unsigned int optimized_spatial_radius = 0;
    unsigned int direction_reuse_mask = 0;
    for (int direction_index = 0; direction_index < NB_DIRECTIONS; direction_index++)
    {
        if (valid_sample_per_direction[direction_index] / static_cast<float>(NB_SAMPLES_PER_DIRECTION) > 0.75f)
        {
            direction_reuse_mask |= 1 << direction_index;

            valid_directions++;
            optimized_spatial_radius += get_distance_from_center_from_sample_index(max_radius_per_direction[direction_index], render_data.render_settings.restir_gi_settings.common_spatial_pass.reuse_radius);
        }
    }

    optimized_spatial_radius = roundf(optimized_spatial_radius / (float)valid_directions);

    render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask[center_pixel_index] = direction_reuse_mask;
    render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius[center_pixel_index] = optimized_spatial_radius;
}

#endif
