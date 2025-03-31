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

#include "HostDeviceCommon/KernelOptions/ReSTIRGIOptions.h"
#include "HostDeviceCommon/RenderData.h"

#define MINIMUM_RADIUS_SIZE 5
#define NB_RADIUS 32
#define NB_SAMPLES_PER_RADIUS_INTERNAL ReSTIR_GI_SpatialDirectionalReuseBitCount // CHANGE THIS ONE
#define NB_SAMPLES_PER_RADIUS (NB_SAMPLES_PER_RADIUS_INTERNAL > 64 ? 64 : NB_SAMPLES_PER_RADIUS_INTERNAL) // Max to 64 for unsigned long long int

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_Directional_Reuse_Compute(HIPRTRenderData render_data,
    unsigned int* __restrict__ out_directional_reuse_masks_buffer_u,
    unsigned long long int* __restrict__ out_directional_reuse_masks_buffer_ull,
    unsigned char* __restrict__ out_adaptive_radius_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_Directional_Reuse_Compute(HIPRTRenderData render_data, int x, int y,
    unsigned int* __restrict__ out_directional_reuse_masks_buffer_u,
    unsigned long long int* __restrict__ out_directional_reuse_masks_buffer_ull,
    unsigned char* __restrict__ out_adaptive_radius_buffer)
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
#if NB_SAMPLES_PER_RADIUS > 32
    out_directional_reuse_masks_buffer_ull[center_pixel_index] = 0;
#else
    out_directional_reuse_masks_buffer_u[center_pixel_index] = 0;
#endif
    out_adaptive_radius_buffer[center_pixel_index] = 0;

    float3 center_shading_point = render_data.g_buffer.primary_hit_position[center_pixel_index];
    float3 center_shading_normal = render_data.g_buffer.shading_normals[center_pixel_index].unpack();





    float best_area = 0.0f;
    int best_radius_index = 0;

    // Each long long int in there contains, in each bit, whether or not the direction for that radius is reusable or not
    unsigned long long int valid_samples_per_radius[NB_RADIUS] = { 0 };

    for (int radius_index = 0; radius_index < NB_RADIUS; radius_index++)
    {
        float current_radius = MINIMUM_RADIUS_SIZE + (radius_index / (float)NB_RADIUS) * (ReSTIRSettingsHelper::get_restir_spatial_pass_settings<true>(render_data).reuse_radius - MINIMUM_RADIUS_SIZE);
        float current_radius_circle_area = M_PI * current_radius * current_radius;

        // Now sampling a bunch of neighbors *on* that radius, exactly at that radius distance from the center (i.e. *not* within the disk of that radius)
        float area_at_current_radius = 0.0f;
        for (int sample_index = 0; sample_index < NB_SAMPLES_PER_RADIUS; sample_index++)
        {
            if (radius_index > 0)
                if (!(valid_samples_per_radius[radius_index - 1] & (1ull << sample_index)))
                    // If this direction wasn't accepted at the previous radius
                    continue;

            float theta = sample_index / (float)NB_SAMPLES_PER_RADIUS * M_TWO_PI;
            float x_circle = current_radius * cosf(theta);
            float y_circle = current_radius * sinf(theta);

            int2 neighbor_offset_in_disk = make_int2(static_cast<int>(x_circle), static_cast<int>(y_circle));
            int2 neighbor_pixel_coords = make_int2(x, y) + neighbor_offset_in_disk;
            if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= render_data.render_settings.render_resolution.x ||
                neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= render_data.render_settings.render_resolution.y)
                // Rejecting the sample if it's outside of the viewport
                continue;

            int neighbor_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * render_data.render_settings.render_resolution.x;

            if (!check_neighbor_similarity_heuristics<true>(render_data, neighbor_index, center_pixel_index, center_shading_point, center_shading_normal))
                continue;

            valid_samples_per_radius[radius_index] |= (1ull << sample_index);
            area_at_current_radius += current_radius_circle_area * (1.0f / NB_SAMPLES_PER_RADIUS);
        }

        if (best_area < area_at_current_radius)
        {
            best_area = area_at_current_radius;
            best_radius_index = radius_index;
        }
    }

    // Computing the actual radius from the best radius index
    float best_radius = MINIMUM_RADIUS_SIZE + (best_radius_index / (float)NB_RADIUS) * (ReSTIRSettingsHelper::get_restir_spatial_pass_settings<true>(render_data).reuse_radius - MINIMUM_RADIUS_SIZE);
    if (best_area == 0.0f)
        best_radius = 0.0f;

    out_adaptive_radius_buffer[center_pixel_index] = (unsigned char)best_radius;
#if NB_SAMPLES_PER_RADIUS > 32
    out_directional_reuse_masks_buffer_ull[center_pixel_index] = valid_samples_per_radius[best_radius_index];
#else
    // Extracting the low 32 bits
    out_directional_reuse_masks_buffer_u[center_pixel_index] = (unsigned int)(valid_samples_per_radius[best_radius_index] & 0x00000000FFFFFFFFF);
#endif

    /*if (render_data.render_settings.debug_x == x && render_data.render_settings.debug_y == y)
    {
        for (int i = 0; i < 128; i++)
        {
            int neighbor_index = get_spatial_neighbor_pixel_index<true>(render_data, i, make_int2(x, y), make_float2(1.0f, 0.0f), random_number_generator);
            if (neighbor_index == -1)
                continue;

            path_tracing_accumulate_color(render_data, ColorRGB32F(0.0f, 1000.0f, 0.0f), neighbor_index);
        }
    }

    if (render_data.render_settings.debug_x == x && render_data.render_settings.debug_y == y)
        path_tracing_accumulate_color(render_data, ColorRGB32F(10000000.0f, 0.0, 0.0f), center_pixel_index);*/
}

#endif
