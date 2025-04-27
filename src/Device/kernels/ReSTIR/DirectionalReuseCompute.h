/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_DIRECTIONAL_REUSE_COMPUTE_H
#define KERNELS_RESTIR_DIRECTIONAL_REUSE_COMPUTE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/UtilsSpatial.h"

#include "HostDeviceCommon/RenderData.h"

#define NB_RADIUS 32
#if ComputingSpatialDirectionalReuseForReSTIRGI == KERNEL_OPTION_TRUE
#define NB_SAMPLES_PER_RADIUS_INTERNAL ReSTIR_GI_SpatialDirectionalReuseBitCount // CHANGE THIS ONE
#else
#define NB_SAMPLES_PER_RADIUS_INTERNAL ReSTIR_DI_SpatialDirectionalReuseBitCount // CHANGE THIS ONE
#endif

#define NB_SAMPLES_PER_RADIUS (NB_SAMPLES_PER_RADIUS_INTERNAL > 64 ? 64 : NB_SAMPLES_PER_RADIUS_INTERNAL) // Max to 64 for unsigned long long int

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_Directional_Reuse_Compute(HIPRTRenderData render_data,
    unsigned int* __restrict__ out_directional_reuse_masks_buffer_u,
    unsigned long long int* __restrict__ out_directional_reuse_masks_buffer_ull,
    unsigned char* __restrict__ out_adaptive_radius_buffer)
#else
template <bool IsReSTIRGI>
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_Directional_Reuse_Compute(HIPRTRenderData render_data, int x, int y,
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

//     #if NB_SAMPLES_PER_RADIUS > 32
//     out_directional_reuse_masks_buffer_ull[center_pixel_index] = center_pixel_index;// valid_samples_per_radius[best_radius_index];
// #else
//     // Extracting the low 32 bits
//     out_directional_reuse_masks_buffer_u[center_pixel_index] = (unsigned int)(valid_samples_per_radius[best_radius_index] & 0x00000000FFFFFFFFF);
// #endif
// return;




#ifdef __KERNELCC__
    // If on the GPU, using the 'ComputingSpatialDirectionalReuseForReSTIRGI' macro
    // (that is passed to the compiler in the ReSTIRDI/GI RenderPass.cpp)
    //
    // To get the settings
    ReSTIRCommonSpatialPassSettings spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ComputingSpatialDirectionalReuseForReSTIRGI>(render_data);
#else
    // On the CPU, it is the template argument that dictates whether this is for ReSTIR DI or GI
    ReSTIRCommonSpatialPassSettings spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
#endif

    float3 center_shading_point = render_data.g_buffer.primary_hit_position[center_pixel_index];
#ifdef __KERNELCC__
    float3 center_normal = ReSTIRSettingsHelper::get_restir_neighbor_similarity_settings<ComputingSpatialDirectionalReuseForReSTIRGI>(render_data).reject_using_geometric_normals ? render_data.g_buffer.geometric_normals[center_pixel_index].unpack() : render_data.g_buffer.shading_normals[center_pixel_index].unpack();
#else
    float3 center_normal = ReSTIRSettingsHelper::get_restir_neighbor_similarity_settings<IsReSTIRGI>(render_data).reject_using_geometric_normals ? render_data.g_buffer.geometric_normals[center_pixel_index].unpack() : render_data.g_buffer.shading_normals[center_pixel_index].unpack();
#endif

    float best_area = 0.0f;
    int best_radius_index = 0;
    // Each long long int in there contains, in each bit, whether or not the direction for that radius is reusable or not
    unsigned long long int valid_samples_per_radius[NB_RADIUS] = { 0 };
    for (int radius_index = 0; radius_index < NB_RADIUS; radius_index++)
    {
        float current_radius = spatial_pass_settings.minimum_per_pixel_reuse_radius + (radius_index / (float)NB_RADIUS) * (spatial_pass_settings.reuse_radius - spatial_pass_settings.minimum_per_pixel_reuse_radius);
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

            int2 neighbor_offset_in_disk = make_int2(static_cast<int>(roundf(x_circle)), static_cast<int>(roundf(y_circle)));
            int2 neighbor_pixel_coords = make_int2(x, y) + neighbor_offset_in_disk;
            if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= render_data.render_settings.render_resolution.x ||
                neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= render_data.render_settings.render_resolution.y)
                // Rejecting the sample if it's outside of the viewport
                continue;

            int neighbor_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * render_data.render_settings.render_resolution.x;

#ifdef __KERNELCC__
            // If on the GPU, using the 'ComputingSpatialDirectionalReuseForReSTIRGI' macro
            // (that is passed to the compiler in the ReSTIRDI/GI RenderPass.cpp)
            //
            // To determine whether this is for ReSTIR DI or GI
            if (!check_neighbor_similarity_heuristics<ComputingSpatialDirectionalReuseForReSTIRGI>(render_data, neighbor_index, center_pixel_index, center_shading_point, center_normal))
                continue;
#else
            // On the CPU, it is the template argument that dictates whether this is for ReSTIR DI or GI
            if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data, neighbor_index, center_pixel_index, center_shading_point, center_normal))
                continue;
#endif

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
    float best_radius = spatial_pass_settings.minimum_per_pixel_reuse_radius + (best_radius_index / (float)NB_RADIUS) * (spatial_pass_settings.reuse_radius - spatial_pass_settings.minimum_per_pixel_reuse_radius);
    if (best_area == 0.0f)
        best_radius = 0.0f;

    out_adaptive_radius_buffer[center_pixel_index] = (unsigned char)best_radius;
#if NB_SAMPLES_PER_RADIUS > 32
    out_directional_reuse_masks_buffer_ull[center_pixel_index] = valid_samples_per_radius[best_radius_index];
#else
    // Extracting the low 32 bits
    out_directional_reuse_masks_buffer_u[center_pixel_index] = (unsigned int)(valid_samples_per_radius[best_radius_index] & 0x00000000FFFFFFFFF);
#endif
}

#endif
