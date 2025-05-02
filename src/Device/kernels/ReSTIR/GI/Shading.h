/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_GI_SHADING_H
#define KERNELS_RESTIR_GI_SHADING_H

#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/Lights.h"
#include "Device/includes/LightSampling/LightUtils.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
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
    ray_payload.next_ray_state = RayState::BOUNCE;
    // Loading the first hit in the ray payload
    ray_payload.material = render_data.g_buffer.materials[pixel_index].unpack();
    ray_payload.volume_state.reconstruct_first_hit(
        ray_payload.material,
        render_data.buffers.material_indices,
        closest_hit_info.primitive_index,
        random_number_generator);

    float3 view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

    // Dummy mis_reuse variable
    MISBSDFRayReuse mis_reuse;
    ColorRGB32F camera_outgoing_radiance;
    if (render_data.render_settings.enable_direct)
        // Adding the direct lighting contribution at the first hit in the direction of the camera
        camera_outgoing_radiance += estimate_direct_lighting(render_data, ray_payload, closest_hit_info, view_direction, x, y, mis_reuse, random_number_generator);

    ReSTIRGIReservoir resampling_reservoir = render_data.render_settings.restir_gi_settings.restir_output_reservoirs[pixel_index];
    if (render_data.render_settings.nb_bounces > 0)
    {
        // Only doing the ReSTIR GI stuff if we have more than 1 bounce

        if (resampling_reservoir.UCW > 0.0f)
        {
            // Only doing the shading if we do actually have a sample

            float3 geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();

            float3 restir_resampled_indirect_direction;
            if (resampling_reservoir.sample.is_envmap_path())
                restir_resampled_indirect_direction = resampling_reservoir.sample.sample_point;
            else
                restir_resampled_indirect_direction = hippt::normalize(resampling_reservoir.sample.sample_point - closest_hit_info.inter_point);

            // Computing the BSDF throughput at the first hit
            //  - view direction: towards the camera
            //  - incident light direction: towards the sample point
            float bsdf_pdf_first_hit;
            BSDFContext bsdf_first_hit_context(view_direction, closest_hit_info.shading_normal, geometric_normal, restir_resampled_indirect_direction, resampling_reservoir.sample.incident_light_info_at_visible_point, ray_payload.volume_state, false, ray_payload.material, 0, 0.0f);
            ColorRGB32F bsdf_color_first_hit = bsdf_dispatcher_eval(render_data, bsdf_first_hit_context, bsdf_pdf_first_hit, random_number_generator);

            ColorRGB32F first_hit_throughput;
            if (bsdf_pdf_first_hit > 0.0f)
                first_hit_throughput = bsdf_color_first_hit * hippt::abs(hippt::dot(restir_resampled_indirect_direction, closest_hit_info.shading_normal)) * resampling_reservoir.UCW;

            if (resampling_reservoir.sample.is_envmap_path())
                camera_outgoing_radiance += path_tracing_miss_gather_envmap(render_data, first_hit_throughput, restir_resampled_indirect_direction, 1, pixel_index);
            else
                camera_outgoing_radiance += first_hit_throughput * resampling_reservoir.sample.incoming_radiance_to_visible_point;
        }
    }

        // Setting the 'camera_outgoing_radiance' into the ray color just for the call to 'sanity_check'
    ray_payload.ray_color = camera_outgoing_radiance;
    if (!sanity_check(render_data, ray_payload.ray_color, x, y))
        return;

    if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::FINAL_RESERVOIR_UCW)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.UCW) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::TARGET_FUNCTION)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.sample.target_function) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::WEIGHT_SUM)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.weight_sum) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::M_COUNT)
        path_tracing_accumulate_color(render_data, ColorRGB32F(resampling_reservoir.M) * render_data.render_settings.restir_gi_settings.debug_view_scale_factor, pixel_index);
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::PER_PIXEL_REUSE_RADIUS && render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius != nullptr)
    {
        float radius_percentage = (render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius[pixel_index] / (float)render_data.render_settings.restir_gi_settings.common_spatial_pass.reuse_radius);
        ColorRGB32F debug_color = hippt::lerp(ColorRGB32F(2.0f, 0.0f, 0.0f), ColorRGB32F(0.0f, 2.0f, 0.0f), radius_percentage);

        debug_set_final_color(render_data, x, y, debug_color);
    }
    else if (render_data.render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::PER_PIXEL_VALID_DIRECTIONS_PERCENTAGE && render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius != nullptr)
    {
        unsigned char accepted_directions = hippt::popc(ReSTIRSettingsHelper::get_spatial_reuse_direction_mask_ull<true>(render_data, pixel_index));
        float accepted_percentage = accepted_directions / 32.0f;
        ColorRGB32F debug_color = hippt::lerp(ColorRGB32F(2.0f, 0.0f, 0.0f), ColorRGB32F(0.0f, 2.0f, 0.0f), accepted_percentage);

        debug_set_final_color(render_data, x, y, debug_color);
    }
    else
        // Regular output
        path_tracing_accumulate_color(render_data, camera_outgoing_radiance, pixel_index);
}

#endif
