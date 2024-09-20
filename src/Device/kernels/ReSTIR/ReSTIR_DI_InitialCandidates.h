/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_DI_INITIAL_CANDIDATES_H
#define KERNELS_RESTIR_DI_INITIAL_CANDIDATES_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/DI/Utils.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDIReservoir sample_initial_candidates(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // Pushing the intersection point outside the surface (if we're already outside)
    // or inside the surface (if we're inside the surface)
    // We'll use that intersection point as the origin of our shadow rays
    bool inside_surface = false;// hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    // If we're rendering at low resolution, only doing 1 candidate of each
    // for better interactive framerates
    int initial_nb_light_cand = render_data.render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates;
    int initial_nb_bsdf_cand = render_data.render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates;

    int nb_light_candidates = render_data.render_settings.do_render_low_resolution() ? hippt::min(1, initial_nb_light_cand) : initial_nb_light_cand;
    int nb_bsdf_candidates = render_data.render_settings.do_render_low_resolution() ? hippt::min(1, initial_nb_bsdf_cand) : initial_nb_bsdf_cand;
    float envmap_candidate_probability = 0.0f;
    if (render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
        envmap_candidate_probability = render_data.render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability;

    // Sampling candidates with weighted reservoir sampling
    ReSTIRDIReservoir reservoir;
    for (int i = 0; i < nb_light_candidates; i++)
    {
        float target_function = 0.0f;
        float candidate_weight = 0.0f;
        ReSTIRDISample light_RIS_sample;

        ColorRGB32F sample_radiance;
        float sample_cosine_term = 0.0f;
        float sample_pdf = 0.0f;
        if (random_number_generator() > envmap_candidate_probability)
        {
            // Light sample

            LightSourceInformation light_source_info;
            light_RIS_sample.point_on_light_source = sample_one_emissive_triangle(render_data, random_number_generator, sample_pdf, light_source_info);
            light_RIS_sample.emissive_triangle_index = light_source_info.emissive_triangle_index;

            if (sample_pdf > 0.0f)
            {
                // It can happen that the light PDF returned by the emissive triangle
                // sampling function is 0 because of emissive triangles that are so
                // small that we cannot compute their normal and their area (the cross
                // product of their edges gives a quasi-null vector --> length of 0.0f --> area of 0)

                float distance_to_light;
                float3 to_light_direction = light_RIS_sample.point_on_light_source - evaluated_point;
                to_light_direction = to_light_direction / (distance_to_light = hippt::length(to_light_direction)); // Normalization

                // Multiplying by the inside_surface_multiplier here because if we're inside the surface, we want to flip the normal
                // for the dot product to be "properly" oriented.
                sample_cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal * inside_surface_multiplier, to_light_direction));

                float cosine_at_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -to_light_direction));
                // Converting the PDF from area measure to solid angle measure requires dividing by
                // cos(theta) / dist^2. Dividing by that factor is equal to multiplying by the inverse
                // which is what we're doing here
                sample_pdf *= distance_to_light * distance_to_light;
                sample_pdf /= cosine_at_light_source;

                // Accounting for the probability of sampling a light, not the envmap
                // (which has probability 'envmap_candidate_probability')
                sample_pdf *= (1.0f - envmap_candidate_probability);

                sample_radiance = light_source_info.emission;
            }
        }
        else
        {
            // Envmap sample

            float3 envmap_sampled_direction;
            sample_radiance = envmap_sample(render_data, envmap_sampled_direction, sample_pdf, random_number_generator);
            float opdf;
            ColorRGB32F radiance_2 = envmap_eval(render_data, envmap_sampled_direction, opdf);

            // Taking into account the fact that we only have a 1 in 'envmap_candidate_probability' chance to sample
            // the envmap
            sample_pdf *= envmap_candidate_probability;

            sample_cosine_term = hippt::max(0.0f, hippt::dot(envmap_sampled_direction, closest_hit_info.shading_normal));

            light_RIS_sample.emissive_triangle_index = -1;
            light_RIS_sample.point_on_light_source = envmap_sampled_direction;
            light_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE;
        }

        float distance_to_light;
        float3 to_light_direction;
        if (light_RIS_sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
        {
            to_light_direction = light_RIS_sample.point_on_light_source;
            distance_to_light = 1.0e35f;
        }
        else
        {
            to_light_direction = light_RIS_sample.point_on_light_source - evaluated_point;
            to_light_direction = to_light_direction / (distance_to_light = hippt::length(to_light_direction)); // Normalization
        }

        if (sample_cosine_term > 0.0f)
        {
            float bsdf_pdf;
            RayVolumeState volume_state = ray_payload.volume_state;
            ColorRGB32F bsdf_contribution = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, volume_state, view_direction, closest_hit_info.shading_normal, to_light_direction, bsdf_pdf);

            float mis_weight = power_heuristic(sample_pdf, nb_light_candidates, bsdf_pdf, nb_bsdf_candidates);

            float target_function = (bsdf_contribution * sample_radiance * sample_cosine_term).luminance();

            candidate_weight = mis_weight * target_function / sample_pdf;

            light_RIS_sample.target_function = target_function;
        }

#if ReSTIR_DI_TargetFunctionVisibility == KERNEL_OPTION_TRUE
        if (!render_data.render_settings.do_render_low_resolution() && light_RIS_sample.target_function > 0.0f)
        {
            // Only doing visiblity if we're render at low resolution
            // (meaning we're moving the camera) for better movement framerates
            // Also, only testing visibility if we got a valid sample

            hiprtRay shadow_ray;
            shadow_ray.origin = evaluated_point;
            shadow_ray.direction = to_light_direction;

            bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);
            if (!visible)
            {
                // Sample occluded, it is not going to be resampled anyways because it is
                // going to have a 0 contribution so we just take it into account in the
                // reservoir (because even if it has zero-contribution, this is still a resampled sample)
                reservoir.M++;

                // And we go onto the next sample
                continue;
            }

            // We are now sure that if the sample survived, it is unoccluded
            light_RIS_sample.flags |= RESTIR_DI_FLAGS_UNOCCLUDED;
        }
#endif

        reservoir.add_one_candidate(light_RIS_sample, candidate_weight, random_number_generator);
        reservoir.sanity_check(make_int2(-1, -1));
    }

    // Sampling the BSDF candidates
    for (int i = 0; i < nb_bsdf_candidates; i++)
    {
        float bsdf_sample_pdf = 0.0f;
        float3 sampled_direction;

        RayVolumeState trash_ray_volume_state = ray_payload.volume_state;
        ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, ray_payload.material, trash_ray_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_direction, bsdf_sample_pdf, random_number_generator);

        float3 shadow_ray_origin = evaluated_point;
        bool refraction_sampled = hippt::dot(sampled_direction, closest_hit_info.shading_normal * inside_surface_multiplier) < 0;
        if (refraction_sampled)
        {
            // If we sampled a refraction, we're pushing the origin of the shadow ray "through"
            // the surface so that our ray can refract inside the surface
            //
            // If we don't do that and we just use the 'evaluated_point' computed at the very
            // beginning of the function, we're going to intersect ourselves

            shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier * -1.0f;
        }

        hiprtRay bsdf_ray;
        if (bsdf_sample_pdf > 0.0f)
        {
            bsdf_ray.origin = shadow_ray_origin;
            bsdf_ray.direction = sampled_direction;

            ShadowLightRayHitInfo shadow_light_ray_hit_info;
            bool hit_found = evaluate_shadow_light_ray(render_data, bsdf_ray, 1.0e35f, shadow_light_ray_hit_info);
            if (hit_found && !shadow_light_ray_hit_info.hit_emission.is_black())
            {
                // If we intersected an emissive material, compute the weight. 
                // Otherwise, the weight is 0 because of the emision being 0 so we just don't compute it

                // Using abs here because we want the dot product to be positive.
                // You may be thinking that if we're doing this, then we're not going to discard BSDF
                // sampled direction that are below the surface (whereas we should discard them).
                // That would be correct but bsdf_dispatcher_sample return a PDF == 0.0f if a bad
                // direction was sampled and if the PDF is 0.0f, we never get to this line of code
                // you're reading. If we are here, this is because we sampled a direction that is
                // correct for the BSDF. Even if the direction is correct, the dot product may be
                // negative in the case of refractions / total internal reflections and so in this case,
                // we'll need to negative the dot product for it to be positive
                float cosine_at_evaluated_point = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_direction));

                float target_function = (bsdf_color * shadow_light_ray_hit_info.hit_emission * cosine_at_evaluated_point).luminance();

                float light_pdf = 0.0f;
                if (!refraction_sampled)
                    // Only computing the light PDF if we're not refracting
                    // 
                    // Why?
                    // 
                    // Because right now, we allow sampling BSDF refractions. This means that we can sample a light
                    // that is inside an object with a *BSDF sample*. However, a *light sample* to the same light cannot
                    // be sampled because there's is going to be the surface of the object we're currently on in-between.
                    // Basically, we are not allowing light sample refractions and so they should have a MIS weight of 0 which
                    // is what we're doing here: the pdf of a *light sample* that refracts through a surface is 0.
                    //
                    // If not doing that, we're going to have bad MIS weights that don't sum up to 1
                    // (because the BSDF sample, that should have weight 1 [or to be precise: 1 / nb_bsdf_samples]
                    // will have weight 1 / (1 + nb_light_samples) [or to be precise: 1 / (nb_bsdf_samples + nb_light_samples)]
                    // and this is going to cause darkening as the number of light samples grows)
                    light_pdf = pdf_of_emissive_triangle_hit(render_data, shadow_light_ray_hit_info, sampled_direction);
                // Our light sampler is only chosen with probability '1.0f - envmap_candidate_probability'
                // so we multiply that here to take that into account
                light_pdf *= (1.0f - envmap_candidate_probability);

                float mis_weight = power_heuristic(bsdf_sample_pdf, nb_bsdf_candidates, light_pdf, nb_light_candidates);
                float candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

                ReSTIRDISample bsdf_RIS_sample;
                bsdf_RIS_sample.emissive_triangle_index = shadow_light_ray_hit_info.hit_prim_index;
                bsdf_RIS_sample.point_on_light_source = bsdf_ray.origin + bsdf_ray.direction * shadow_light_ray_hit_info.hit_distance;
                bsdf_RIS_sample.target_function = target_function;
                bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;

                // TODO optimize here and if we keep the sample of the BSDF, we don't have to re-test for visibility when evaluating the reservoir
                // because a BSDF sample can only be chosen if it's unoccluded (otherwise its weight is 0)
                reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
                reservoir.sanity_check(make_int2(-1, -1));
            }
            else if (!hit_found && render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP)
            {
                // Envmap hit, this becomes an envmap sample
                
                // max(0.0f) here because we're not allowing envmap samples below the surface anyways (refraction envmap samples)
                float cosine_at_evaluated_point = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_direction));

                if (cosine_at_evaluated_point > 0.0f)
                {
                    float envmap_pdf;
                    ColorRGB32F envmap_radiance = envmap_eval(render_data, sampled_direction, envmap_pdf);
                    // We're evaluating the probability of choosing that BSDF-sample direction with the envmap sampler.
                    // Because our envmap sampler is chosen only with probability 'envmap_candidate_probability', we multiply
                    // that here to account for that
                    envmap_pdf *= envmap_candidate_probability;

                    float target_function = (bsdf_color * envmap_radiance * cosine_at_evaluated_point).luminance();
                    // Not taking the light sampling PDF into account in the balance heuristic because a envmap hit
                    // (not a light surface hit) can never be sampled by a light-surface sampler and so the PDF
                    // of the current envmap sample is always 0 for a light sampler.
                    float mis_weight = power_heuristic(bsdf_sample_pdf, nb_bsdf_candidates, envmap_pdf, nb_light_candidates);
                    float candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

                    ReSTIRDISample bsdf_RIS_sample;
                    bsdf_RIS_sample.emissive_triangle_index = -1;
                    bsdf_RIS_sample.point_on_light_source = sampled_direction;
                    bsdf_RIS_sample.target_function = target_function;
                    bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE;
                    bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;

                    // TODO optimize here and if we keep the sample of the BSDF, we don't have to re-test for visibility when evaluating the reservoir
                    // because a BSDF sample can only be chosen if it's unoccluded (otherwise its weight is 0)
                    reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
                    reservoir.sanity_check(make_int2(-1, -1));
                }
            }
        }
    }

    reservoir.end();
    reservoir.sanity_check(make_int2(-1, -1));
    // There's no need to keep M > 1 here, if you have 4 light candidates and 1 BSDF candidates, that's 5 samples.
    // But if you divide everyone by 5, everything stays correct. That allows manipulating the M-cap without having
    // to take the number of initial candidates into account
    reservoir.M = 1;

    return reservoir;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data, int2 res, int x, int y)
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
    if (x >= res.x || y >= res.y)
        return;

    uint32_t pixel_index = (x + y * res.x);

    unsigned int seed;
    if (render_data.render_settings.freeze_random)
        seed = wang_hash(pixel_index + 1);
    else
        seed = wang_hash((pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_seed);

    Xorshift32Generator random_number_generator(seed);

    if (!render_data.aux_buffers.pixel_active[pixel_index] || !render_data.g_buffer.camera_ray_hit[pixel_index])
        // Pixel inactive because of adaptive sampling, returning
        return;

    HitInfo hit_info;
    hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index];
    hit_info.shading_normal = render_data.g_buffer.shading_normals[pixel_index];
    hit_info.inter_point = render_data.g_buffer.first_hits[pixel_index];

    SimplifiedRendererMaterial material = render_data.g_buffer.materials[pixel_index];

    float3 view_direction = render_data.g_buffer.view_directions[pixel_index];

    RayPayload ray_payload;
    ray_payload.material = material;
    ray_payload.volume_state = render_data.g_buffer.ray_volume_states[pixel_index];

    // Producing and storing the reservoir
    ReSTIRDIReservoir initial_candidates_reservoir = sample_initial_candidates(render_data, ray_payload, hit_info, view_direction, random_number_generator);

#if ReSTIR_DI_DoVisibilityReuse == KERNEL_OPTION_TRUE
    ReSTIR_DI_visibility_reuse(render_data, initial_candidates_reservoir, hit_info.inter_point + hit_info.shading_normal * 1.0e-4f);
#endif

    render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[pixel_index] = initial_candidates_reservoir;
}

#endif
