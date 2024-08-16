/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_DI_INITIAL_CANDIDATES_H
#define KERNELS_RESTIR_DI_INITIAL_CANDIDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/RIS.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data, int2 res, HIPRTCamera camera, int x, int y)
#endif
{
    if (render_data.buffers.emissive_triangles_count == 0)
        // No initial candidates to sample since no lights
        // TODO this is incorrect for the envmap since ReSTIR should also sample the envmap
        return;

#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    uint32_t pixel_index = (x + y * res.x);
    if (pixel_index >= res.x * res.y)
        return;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1));

    Xorshift32Generator random_number_generator(seed);

    HitInfo hit_info;
    hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index];
    hit_info.shading_normal = render_data.g_buffer.shading_normals[pixel_index];
    hit_info.inter_point = render_data.g_buffer.first_hits[pixel_index];

    // TODO replace this by using the simplified material directly in sample_light_RIS_reservoir instead of converting to RendererMaterial
    // TODO storing in g_buffer.materials with SimplifiedRendererMaterial loses the texutres information
    SimplifiedRendererMaterial simplified_mat = render_data.g_buffer.materials[pixel_index];
    RendererMaterial material = RendererMaterial(simplified_mat);

    float3 view_direction = render_data.g_buffer.view_directions[pixel_index];

    RayPayload ray_payload;
    ray_payload.material = material;
    ray_payload.volume_state = render_data.g_buffer.ray_volume_states[pixel_index];

    int light_candidates = render_data.render_settings.restir_di_render_settings.number_of_initial_light_candidates;
    int bsdf_candidates = render_data.render_settings.restir_di_render_settings.number_of_initial_bsdf_candidates;
    // Producing and storing the reservoir
    render_data.aux_buffers.initial_reservoirs[pixel_index] = sample_bsdf_and_lights_RIS_reservoir(render_data, ray_payload, hit_info, view_direction, light_candidates, bsdf_candidates, random_number_generator);
}

#endif
