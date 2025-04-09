/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_GI_SHADING_H
#define KERNELS_RESTIR_GI_SHADING_H

#include "Device/includes/Envmap.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Lights.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/ShadeReservoir.h"
#include "Device/includes/ReSTIR/GI/TargetFunction.h"
#include "Device/includes/ReSTIR/UtilsSpatial.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

 // ReSTIR GI shading/resampling is still a bit broken, there's still some brightening bias coming from
 // I don't know where, supposedly when the BRDF starts to include smooth/glossy BRDFs
 // 
 // This manifests the most on specular 1 + roughness 0 everywhere in the scene
 // Maybe the bias is also there with a Lambertian BRDF but I could never see it. Maybe it's there but it's just so subtle that it's invisible
 // 
 // -------------------------- WHAT WE KNOW --------------------------
 // - Still biased with no alpha tests
 // - Do we absolutely have correct convergence on Lambertian & Oren Nayar? --> hard to verify, looks like it?
 // - Is it the glass that is biased? -------> No
 // - 1/Z is also biased, even without the jacobian rejection heuristic
 // - It's not the adaptive sampling that is messed up
 // - Definitely has some bias (very little but there) with everything using a metallic BRDF, roughness 0.1, 50 bounces. Contemporary bedroom
 // - There is some bias in the contemporary bedroom at 1 bounce, everything specular, 0 roughness, with RIS light sampling + envmap sampling
 // - Can't see any bias with lambertian/oren nayar contemporary bedroom + NEE + Envmap
 // - There is still some bias with a roughness 1.0f metallic
 // - Not a normal mapping issue?
 // - With everything specular at IOR 1.0f but roughness 1.0f, there's basically no bias. Even though the specular layer has no effect because of IOR 1.0f. So if the roughness of an inexistant layer changes the bias, it can only be a PDF issue?
 // - Because there is no bias on full Lambertian, this isn't a jacobian issue?
 // 
 // With a specular IOR 1 + diffuse lobe setup
 //     - increasing the roughness of the specular (still at IOR 1) reduces the bias
 //     - artificially fixing the proba of sampling the specular to 90 % and diffuse 10 % increases the bias quiiite a lot(but still converges correctly without ReSTIR and when reusing 0 neighbor).Even more so when approaching 100 % specular(but not quite 100 %)
 //     - sampling the specular & diffuse lobe based on the fresnel reflectance yields a different bias vs.sampling 50 / 50 (or 90 / 10).
 //     - resampling more spatial neighbors makes things a bit worse.But only up to a certain point.For example, 6 spatial reuse passes with 16 candidates each(which is huge) is barely worse than 1 spatial pass @ 16 candidates
 //     - with 0 spatial neighbor reuse, it converges correctly, no matter the BSDF / sampling probas / ...
 //     - there seems to be no bias with only 1 bounce(i.e.with paths, being at most : "camera -> first hit -> second hit").Bias only comes in with # of bounce >= 2
 // -------------------------- WHAT WE KNOW --------------------------
 // 
 // -------------------------- DIRTY FIX RIGHT NOW --------------------------
 // - No double BSDF shading
 // - No double BSDF in target function
 // - Reuse on specular is ok
 // - Using rejection heuristics is better
 // -------------------------- DIRTY FIX RIGHT NOW --------------------------

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_GI_Shading(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_GI_Shading(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
        return;

    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    if (!render_data.aux_buffers.pixel_active[pixel_index])
        return;

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_number);
    Xorshift32Generator random_number_generator(seed);

    hiprtRay ray;
    ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

    HitInfo closest_hit_info;
    closest_hit_info.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];
    if (closest_hit_info.primitive_index == -1)
    {
        // Geometry miss, directly into the envmap
        ColorRGB32F envmap_radiance = path_tracing_miss_gather_envmap(render_data, ColorRGB32F(1.0f), ray.direction, 0, pixel_index);
        path_tracing_accumulate_color(render_data, envmap_radiance, pixel_index);

        return;
    }
    closest_hit_info.inter_point = render_data.g_buffer.primary_hit_position[pixel_index];
    closest_hit_info.shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();

    // Initializing the ray with the information from the camera ray pass
    RayPayload ray_payload;
    // Loading the first hit in the ray payload
    ray_payload.material = render_data.g_buffer.materials[pixel_index].unpack();
    ray_payload.volume_state.reconstruct_first_hit(
        ray_payload.material,
        render_data.buffers.material_indices,
        closest_hit_info.primitive_index,
        random_number_generator);

    float3 view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

    ColorRGB32F shaded_reservoir_color;
    if (render_data.render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass && ReSTIR_GI_DoSpatialNeighborsDecoupledShading == KERNEL_OPTION_TRUE)
        shaded_reservoir_color = render_data.render_settings.restir_gi_settings.common_spatial_pass.decoupled_shading_reuse_buffer[pixel_index];
    else
        shaded_reservoir_color = shade_ReSTIR_GI_reservoir(render_data,
            render_data.render_settings.restir_gi_settings.restir_output_reservoirs[pixel_index],
            view_direction, ray_payload, closest_hit_info, x, y, random_number_generator);

    path_tracing_accumulate_color(render_data, shaded_reservoir_color, pixel_index);
}

#endif
