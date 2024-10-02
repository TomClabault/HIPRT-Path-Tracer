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
#include "Device/includes/ReSTIR/DI/PresampledLight.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Reference: https://en.wikipedia.org/wiki/Pairing_function
 */
HIPRT_HOST_DEVICE HIPRT_INLINE int cantor_pairing_function(int x, int y)
{
    return (x + y + 1) * (x + y) / 2 + y;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDISample use_presampled_light_candidate(const HIPRTRenderData& render_data, const int2& pixel_coords,
    const float3& evaluated_point, const float3& shading_normal,
    ColorRGB32F& out_sample_radiance, float& out_sample_cosine_term, float& out_sample_pdf, float& out_distance_to_light, float3& out_to_light_direction,
    Xorshift32Generator& random_number_generator)
{
    ReSTIRDISample light_sample;
    const LightPresamplingSettings& light_presampling_settings = render_data.render_settings.restir_di_settings.light_presampling;

    // We want all threads in a block of light_presampling_settings.tile_size * light_presampling_settings.tile_size
    // pixels to sample from the same random subset of lights.
    // We compute a unique number per each light_presampling_settings.tile_size * light_presampling_settings.tile_size
    // tile of pixels and use that unique number as seed for our random number generator
    int tile_index_seed = cantor_pairing_function(pixel_coords.x / light_presampling_settings.tile_size, pixel_coords.y / light_presampling_settings.tile_size);

    Xorshift32Generator subset_rng(render_data.random_seed * (tile_index_seed + 1));
    int random_subset_index = subset_rng.random_index(light_presampling_settings.number_of_subsets);
    int random_light_index_in_subset = random_number_generator.random_index(light_presampling_settings.subset_size);
    int light_sample_index = random_subset_index * light_presampling_settings.subset_size + random_light_index_in_subset;

    ReSTIRDIPresampledLight presampled_light_sample = light_presampling_settings.light_samples[light_sample_index];

    light_sample.emissive_triangle_index = presampled_light_sample.emissive_triangle_index;
    light_sample.point_on_light_source = presampled_light_sample.point_on_light_source;
    light_sample.flags = presampled_light_sample.flags;

    out_sample_radiance = presampled_light_sample.radiance;
    out_sample_pdf = presampled_light_sample.pdf;

    if (light_sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
    {
        out_to_light_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, light_sample.point_on_light_source);
        out_distance_to_light = 1.0e35f;
    }
    else
    {
        out_to_light_direction = light_sample.point_on_light_source - evaluated_point;
        out_to_light_direction = out_to_light_direction / (out_distance_to_light = hippt::length(out_to_light_direction)); // Normalization
    }

    out_sample_cosine_term = hippt::dot(shading_normal, out_to_light_direction);

    if (!(light_sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE))
    {
        // To solid angle conversion if not envmap sample (already in solid angle)
        float cosine_at_light_source = hippt::abs(hippt::dot(presampled_light_sample.light_source_normal, -out_to_light_direction));
        out_sample_pdf *= out_distance_to_light * out_distance_to_light;
        out_sample_pdf /= cosine_at_light_source;

        bool contributes_enough = check_minimum_light_contribution(render_data.render_settings.minimum_light_contribution, out_sample_radiance * out_sample_cosine_term / out_sample_pdf);
        if (!contributes_enough)
        {
            // Early check that the light contributes enough to the point, and if it doesn't, skip that light sample

            // Setting it to -42.0f so that we know that the sample is invalid when the caller of this
            // function will look at the target function's value
            light_sample.target_function = -42.0f;

            return light_sample;
        }
    }

    return light_sample;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDISample sample_fresh_light_candidate(const HIPRTRenderData& render_data, float envmap_candidate_probability, const HitInfo& closest_hit_info, ColorRGB32F& out_sample_radiance, float& out_sample_cosine_term, float& out_sample_pdf, Xorshift32Generator& random_number_generator)
{
    ReSTIRDISample light_sample;

    bool inside_surface = false;// hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    if (random_number_generator() > envmap_candidate_probability)
    {
        // Light sample

        LightSourceInformation light_source_info;
        light_sample.point_on_light_source = uniform_sample_one_emissive_triangle(render_data, random_number_generator, out_sample_pdf, light_source_info);
        light_sample.emissive_triangle_index = light_source_info.emissive_triangle_index;

        if (out_sample_pdf > 0.0f)
        {
            // It can happen that the light PDF returned by the emissive triangle
            // sampling function is 0 because of emissive triangles that are so
            // small that we cannot compute their normal and their area (the cross
            // product of their edges gives a quasi-null vector --> length of 0.0f --> area of 0)

            float distance_to_light;
            float3 to_light_direction = light_sample.point_on_light_source - evaluated_point;
            to_light_direction = to_light_direction / (distance_to_light = hippt::length(to_light_direction)); // Normalization

            // Multiplying by the inside_surface_multiplier here because if we're inside the surface, we want to flip the normal
            // for the dot product to be "properly" oriented.
            out_sample_cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal * inside_surface_multiplier, to_light_direction));

            float cosine_at_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -to_light_direction));
            // Converting the PDF from area measure to solid angle measure requires dividing by
            // cos(theta) / dist^2. Dividing by that factor is equal to multiplying by the inverse
            // which is what we're doing here
            out_sample_pdf *= distance_to_light * distance_to_light;
            out_sample_pdf /= cosine_at_light_source;

            bool contributes_enough = check_minimum_light_contribution(render_data.render_settings.minimum_light_contribution, light_source_info.emission * out_sample_cosine_term / out_sample_pdf);
            if (!contributes_enough)
            {
                // Early check that the light contributes enough to the point, and if it doesn't, skip that light sample

                // Setting it to -42.0f so that we know that the sample is invalid when the caller of this
                // function will look at the target function's value
                light_sample.target_function = -42.0f;

                return light_sample;
            }

            // Accounting for the probability of sampling a light, not the envmap
            // (which has probability 'envmap_candidate_probability')
            out_sample_pdf *= (1.0f - envmap_candidate_probability);

            out_sample_radiance = light_source_info.emission;
        }
    }
    else
    {
        // Envmap sample

        float3 envmap_sampled_direction;
        out_sample_radiance = envmap_sample(render_data.world_settings, envmap_sampled_direction, out_sample_pdf, random_number_generator);
        out_sample_cosine_term = hippt::max(0.0f, hippt::dot(envmap_sampled_direction, closest_hit_info.shading_normal));

        bool contributes_enough = check_minimum_light_contribution(render_data.render_settings.minimum_light_contribution, out_sample_radiance * out_sample_cosine_term / out_sample_pdf);
        if (!contributes_enough)
        {
            // Early check that the envmap sample contributes enough to the point, and if it doesn't, skip it

            // Setting it to -42.0f so that we know that the sample is invalid when the caller of this
            // function will look at the target function's value
            light_sample.target_function = -42.0f;

            return light_sample;

        }

        // Taking into account the fact that we only have a 1 in 'envmap_candidate_probability' chance to sample
        // the envmap
        out_sample_pdf *= envmap_candidate_probability;

        light_sample.emissive_triangle_index = -1;
        // Storing in envmap space
        light_sample.point_on_light_source = matrix_X_vec(render_data.world_settings.world_to_envmap_matrix, envmap_sampled_direction);
        light_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE;
    }

    return light_sample;
}

// Try passing only volume state in here, not ray payload
HIPRT_HOST_DEVICE HIPRT_INLINE void sample_light_candidates(const HIPRTRenderData& render_data, const int2& pixel_coords, ReSTIRDIReservoir& reservoir, int nb_light_candidates, int nb_bsdf_candidates, float envmap_candidate_probability, const float3& view_direction, const HitInfo& closest_hit_info, const RayPayload& ray_payload, Xorshift32Generator& random_number_generator)
{
    bool inside_surface = false;// hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    for (int i = 0; i < nb_light_candidates; i++)
    {
        ColorRGB32F sample_radiance;
        float sample_cosine_term = 0.0f;
        float sample_pdf = 0.0f;

        float distance_to_light = 0.0f;
        float3 to_light_direction{ 0.0f, 0.0f, 0.0f };
#if ReSTIR_DI_DoLightsPresampling == KERNEL_OPTION_TRUE
        ReSTIRDISample light_sample = use_presampled_light_candidate(render_data, pixel_coords, 
            evaluated_point, closest_hit_info.shading_normal * inside_surface_multiplier, 
            sample_radiance, sample_cosine_term, sample_pdf, distance_to_light, to_light_direction, 
            random_number_generator);
#else
        ReSTIRDISample light_sample = sample_fresh_light_candidate(render_data, envmap_candidate_probability, closest_hit_info, sample_radiance, sample_cosine_term, sample_pdf, random_number_generator);

        if (light_sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE)
        {
            to_light_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, light_sample.point_on_light_source);
            distance_to_light = 1.0e35f;
        }
        else
        {
            to_light_direction = light_sample.point_on_light_source - evaluated_point;
            to_light_direction = to_light_direction / (distance_to_light = hippt::length(to_light_direction)); // Normalization
        }
#endif

        if (light_sample.target_function == -42.0f)
        {
            // Special value to indicate that this sample should be skipped

            reservoir.M++;
            continue;
        }

        float candidate_weight = 0.0f;
        if (sample_cosine_term > 0.0f && sample_pdf > 0.0f)
        {
            float bsdf_pdf;
            RayVolumeState volume_state = ray_payload.volume_state;
            ColorRGB32F bsdf_contribution = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, volume_state, view_direction, closest_hit_info.shading_normal, to_light_direction, bsdf_pdf);

            ColorRGB32F light_contribution = bsdf_contribution * sample_radiance * sample_cosine_term;
            float target_function = light_contribution.luminance();

            if (!check_minimum_light_contribution(render_data.render_settings.minimum_light_contribution, light_contribution / sample_pdf / bsdf_pdf))
                target_function = 0.0f;
            else
            {
                float mis_weight = power_heuristic(sample_pdf, nb_light_candidates, bsdf_pdf, nb_bsdf_candidates);
                candidate_weight = mis_weight * target_function / sample_pdf;

                light_sample.target_function = target_function;
            }
        }

#if ReSTIR_DI_InitialTargetFunctionVisibility == KERNEL_OPTION_TRUE
        if (!render_data.render_settings.do_render_low_resolution() && light_sample.target_function > 0.0f)
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
            light_sample.flags |= RESTIR_DI_FLAGS_UNOCCLUDED;
        }
#endif

        reservoir.add_one_candidate(light_sample, candidate_weight, random_number_generator);
        reservoir.sanity_check(make_int2(-1, -1));
    }
}

HIPRT_HOST_DEVICE HIPRT_INLINE void sample_bsdf_candidates(const HIPRTRenderData& render_data, ReSTIRDIReservoir& reservoir, int nb_light_candidates, int nb_bsdf_candidates, float envmap_candidate_probability, const float3& view_direction, const HitInfo& closest_hit_info, const RayPayload& ray_payload, Xorshift32Generator& random_number_generator)
{
    bool inside_surface = false;// hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

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
            bool hit_found = evaluate_shadow_light_ray(render_data, bsdf_ray, 1.0e35f, shadow_light_ray_hit_info, random_number_generator);
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

                ColorRGB32F light_contribution = bsdf_color * shadow_light_ray_hit_info.hit_emission * cosine_at_evaluated_point;
                float target_function = light_contribution.luminance();

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

                if (!check_minimum_light_contribution(render_data.render_settings.minimum_light_contribution, light_contribution / light_pdf / bsdf_sample_pdf))
                {
                    // Skipping if the light doesn't contribute enough
                    reservoir.M++;

                    continue;
                }

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

                    ColorRGB32F envmap_contribution = bsdf_color * envmap_radiance * cosine_at_evaluated_point;
                    // Not taking the light sampling PDF into account in the balance heuristic because a envmap hit
                    // (not a light surface hit) can never be sampled by a light-surface sampler and so the PDF
                    // of the current envmap sample is always 0 for a light sampler.
                    if (!check_minimum_light_contribution(render_data.render_settings.minimum_light_contribution, envmap_contribution / envmap_pdf / bsdf_sample_pdf))
                    {
                        // Skipping if the light doesn't contribute enough
                        reservoir.M++;

                        continue;
                    }

                    float target_function = envmap_contribution.luminance();

                    // We're evaluating the probability of choosing that BSDF-sample direction with the envmap sampler.
                    // Because our envmap sampler is chosen only with probability 'envmap_candidate_probability', we multiply
                    // that here to account for that
                    envmap_pdf *= envmap_candidate_probability;
                    float mis_weight = power_heuristic(bsdf_sample_pdf, nb_bsdf_candidates, envmap_pdf, nb_light_candidates);
                    float candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

                    ReSTIRDISample bsdf_RIS_sample;
                    bsdf_RIS_sample.emissive_triangle_index = -1;
                    // Storing in envmap space
                    bsdf_RIS_sample.point_on_light_source = matrix_X_vec(render_data.world_settings.world_to_envmap_matrix, sampled_direction);
                    bsdf_RIS_sample.target_function = target_function;
                    bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE;
                    bsdf_RIS_sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;

                    reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
                    reservoir.sanity_check(make_int2(-1, -1));
                }
            }
        }
    }
}

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDIReservoir sample_initial_candidates(const HIPRTRenderData& render_data, const int2& pixel_coords, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
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
    {
        if (render_data.buffers.emissive_triangles_count == 0)
            // Only the envmap to sample
            envmap_candidate_probability = 1.0f;
        else
            envmap_candidate_probability = render_data.render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability;
    }

    // Sampling candidates with weighted reservoir sampling
    ReSTIRDIReservoir reservoir;

    sample_light_candidates(render_data, pixel_coords, reservoir, nb_light_candidates, nb_bsdf_candidates, envmap_candidate_probability, view_direction, closest_hit_info, ray_payload, random_number_generator);
    sample_bsdf_candidates(render_data, reservoir, nb_light_candidates, nb_bsdf_candidates, envmap_candidate_probability, view_direction, closest_hit_info, ray_payload, random_number_generator);

    reservoir.end();
    reservoir.sanity_check(pixel_coords);
    // There's no need to keep M > 1 here, if you have 4 light candidates and 1 BSDF candidates, that's 5 samples.
    // But if you divide everyone by 5, everything stays correct. That allows manipulating the M-cap without having
    // to take the number of initial candidates into account
    reservoir.M = 1;

    return reservoir;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_InitialCandidates(HIPRTRenderData render_data, int2 res, int x, int y)
#endif
{
    if (render_data.buffers.emissive_triangles_count == 0 && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
        // No initial candidates to sample since no lights
        return;

#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (x >= res.x || y >= res.y)
        return;

    uint32_t pixel_index = (x + y * res.x);
    SimplifiedRendererMaterial material = render_data.g_buffer.materials[pixel_index];

    if (material.is_emissive())
        // If this pixel is on an emissive material, indicating that the reservoir is emissive
        // with the flag so that the temporal and spatial reuse can avoir resampling on those.
        // We're not resampling on emissive materials because there is no point, we're not trying
        // to shade emissive materials so we don't need light samples on emissive materials
        return;

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


    float3 view_direction = render_data.g_buffer.view_directions[pixel_index];

    RayPayload ray_payload;
    ray_payload.material = material;
    ray_payload.volume_state = render_data.g_buffer.ray_volume_states[pixel_index];

    // Producing and storing the reservoir
    ReSTIRDIReservoir initial_candidates_reservoir = sample_initial_candidates(render_data, make_int2(x, y), ray_payload, hit_info, view_direction, random_number_generator);

#if ReSTIR_DI_DoVisibilityReuse == KERNEL_OPTION_TRUE
    ReSTIR_DI_visibility_reuse(render_data, initial_candidates_reservoir, hit_info.inter_point + hit_info.shading_normal * 1.0e-4f, random_number_generator);
#endif

    render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[pixel_index] = initial_candidates_reservoir;
}

#endif
