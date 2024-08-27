/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RIS_H
#define DEVICE_RIS_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/RIS/RIS_Reservoir.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

// TODO make some simplification assuming that ReSTIR DI is never inside a surface (the camera being inside a surface may be an annoying case to handle)
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F evaluate_reservoir_sample(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const float3& shading_point, const float3& shading_normal, const float3& view_direction, const RISReservoir& reservoir)
{
    ColorRGB32F final_color;

    if (reservoir.weight_sum == 0.0f)
        // No valid sample means no light contribution
        return ColorRGB32F(0.0f);

    RISSample sample = reservoir.sample;

    bool in_shadow;
    float distance_to_light;
    float3 evaluated_point = shading_point + shading_normal * 1.0e-4f;
    float3 shadow_ray_direction = sample.point_on_light_source - evaluated_point;
    float3 shadow_ray_direction_normalized = shadow_ray_direction / (distance_to_light = hippt::length(shadow_ray_direction));
    if (sample.is_bsdf_sample)
        // A BSDF sample that has been picked by RIS cannot be occluded otherwise
        // it would have a weight of 0 and would never be picked by RIS
        in_shadow = false;
    else
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = evaluated_point;
        shadow_ray.direction = shadow_ray_direction_normalized;

        in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);
    }

    if (!in_shadow)
    {
        float bsdf_pdf;
        float cosine_at_evaluated_point;
        ColorRGB32F bsdf_color;
        RayVolumeState trash_volume_state = ray_payload.volume_state;

        if (sample.is_bsdf_sample)
        {
            // If we picked a BSDF sample, we're using the already computed cosine term and color
            // because it's annoying to recompute it (we have to know if the BSDF is a refraction
            // sample or not)
            bsdf_color = sample.bsdf_sample_contribution;
            cosine_at_evaluated_point = sample.bsdf_sample_cosine_term;
        }
        else
        {
            bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, shading_normal, shadow_ray_direction_normalized, bsdf_pdf);

            cosine_at_evaluated_point = hippt::max(0.0f, hippt::dot(shading_normal, shadow_ray_direction_normalized));
        }

        if (cosine_at_evaluated_point > 0.0f)
        {
            int material_index = render_data.buffers.material_indices[sample.emissive_triangle_index];
            ColorRGB32F sample_emission = render_data.buffers.materials_buffer[material_index].emission;

            final_color = bsdf_color * reservoir.UCW * sample_emission * cosine_at_evaluated_point;
        }
    }

    return final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE RISReservoir sample_bsdf_and_lights_RIS_reservoir(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // Pushing the intersection point outside the surface (if we're already outside)
    // or inside the surface (if we're inside the surface)
    // We'll use that intersection point as the origin of our shadow rays
    bool inside_surface = hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    // If we're rendering at low resolution, only doing 1 candidate of each
    // for better interactive framerates
    int nb_light_candidates = render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.ris_settings.number_of_light_candidates;
    int nb_bsdf_candidates = render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.ris_settings.number_of_bsdf_candidates;

    // Sampling candidates with weighted reservoir sampling
    RISReservoir reservoir;
    for (int i = 0; i < nb_light_candidates; i++)
    {
        float light_sample_pdf;
        float distance_to_light;
        float cosine_at_light_source;
        float cosine_at_evaluated_point;
        LightSourceInformation light_source_info;

        ColorRGB32F bsdf_color;
        float target_function = 0.0f;
        float candidate_weight = 0.0f;
        float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);
        if (light_sample_pdf > 0.0f)
        {
            // It can happen that the light PDF returned by the emissive triangle
            // sampling function is 0 because of emissive triangles that are so
            // small that we cannot compute their normal and their area (the cross
            // product of their edges gives a quasi-null vector --> length of 0.0f --> area of 0)

            float3 to_light_direction;
            to_light_direction = random_light_point - evaluated_point;
            distance_to_light = hippt::length(to_light_direction);
            to_light_direction = to_light_direction / distance_to_light; // Normalization
            cosine_at_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -to_light_direction));
            // Multiplying by the inside_surface_multiplier here because if we're inside the surface, we want to flip the normal
            // for the dot product to be "properly" oriented.
            cosine_at_evaluated_point = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal * inside_surface_multiplier, to_light_direction));
            if (cosine_at_evaluated_point > 0.0f)
            {
                float bsdf_pdf;
                RayVolumeState trash_ray_volume_state = ray_payload.volume_state;
                bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, trash_ray_volume_state, view_direction, closest_hit_info.shading_normal, to_light_direction, bsdf_pdf);

                float geometry_term = 1.0f;
                if (render_data.render_settings.ris_settings.geometry_term_in_target_function)
                    geometry_term = cosine_at_light_source / (distance_to_light * distance_to_light);
      
                target_function = (bsdf_color * light_source_info.emission * cosine_at_evaluated_point * geometry_term).luminance();

#if RISUseVisiblityTargetFunction == KERNEL_OPTION_TRUE
                if (!render_data.render_settings.do_render_low_resolution())
                {
                    // Only doing visiblity if we're not rendering at low resolution
                    // (meaning we're moving the camera) for better interaction framerates

                    hiprtRay shadow_ray;
                    shadow_ray.origin = evaluated_point;
                    shadow_ray.direction = to_light_direction;

                    bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

                    target_function *= visible;
                }
#endif

                // Converting the PDF from area measure to solid angle measure requires dividing by
                // cos(theta) / dist^2. Dividing by that factor is equal to multiplying by the inverse
                // which is what we're doing here
                light_sample_pdf *= distance_to_light * distance_to_light;
                light_sample_pdf /= cosine_at_light_source;

                float mis_weight = balance_heuristic(light_sample_pdf, nb_light_candidates, bsdf_pdf, nb_bsdf_candidates);
                candidate_weight = mis_weight * target_function / light_sample_pdf;
            }
        }

        RISSample light_RIS_sample;
        light_RIS_sample.is_bsdf_sample = false;
        light_RIS_sample.point_on_light_source = random_light_point;
        light_RIS_sample.target_function = target_function;
        light_RIS_sample.emissive_triangle_index = light_source_info.emissive_triangle_index;

        reservoir.add_one_candidate(light_RIS_sample, candidate_weight, random_number_generator);
    }

    // Whether or not a BSDF sample has been retained by the reservoir
    for (int i = 0; i < nb_bsdf_candidates; i++)
    {
        float bsdf_sample_pdf = 0.0f;
        float target_function = 0.0f;
        float candidate_weight = 0.0f;
        float3 sampled_direction;
        float3 shadow_ray_origin = evaluated_point;
        RayVolumeState trash_ray_volume_state = ray_payload.volume_state;
        ColorRGB32F bsdf_color;

        bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, ray_payload.material, trash_ray_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_direction, bsdf_sample_pdf, random_number_generator);

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

        float cosine_at_evaluated_point = 0.0f;
        RISSample bsdf_RIS_sample;
        hiprtRay bsdf_ray;
        if (bsdf_sample_pdf > 0.0f)
        {
            bsdf_ray.origin = shadow_ray_origin;
            bsdf_ray.direction = sampled_direction;

            HitInfo bsdf_ray_hit_info;
            RayPayload bsdf_ray_payload;
            bool hit_found = trace_ray(render_data, bsdf_ray, bsdf_ray_payload, bsdf_ray_hit_info);
            if (hit_found && bsdf_ray_payload.material.is_emissive())
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
                cosine_at_evaluated_point = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_direction));

                float geometry_term = 1.0f;
                if (render_data.render_settings.restir_di_settings.target_function.geometry_term_in_target_function)
                {
                    float cosine_at_light_source = hippt::abs(hippt::dot(sampled_direction, bsdf_ray_hit_info.shading_normal));
                    geometry_term = cosine_at_light_source / (bsdf_ray_hit_info.t * bsdf_ray_hit_info.t);
                }

                target_function = (bsdf_color * bsdf_ray_payload.material.emission * cosine_at_evaluated_point * geometry_term).luminance();

                float light_pdf = pdf_of_emissive_triangle_hit(render_data, bsdf_ray_hit_info, sampled_direction); 
                // If we refracting, drop the light PDF to 0
                // 
                // Why?
                // 
                // Because right now, we allow sampling BSDF refractions. This means that we can sample a light
                // that is inside an object with a BSDF sample. However, a light sample to the same light cannot
                // be sampled because there's is going to be the surface of the object we're currently on in-between.
                // Basically, we are not allowing light sample refractions and so they should have a weight of 0 which
                // is what we're doing here: the pdf of a light sample that refracts through a surface is 0.
                //
                // If not doing that, we're going to have bad MIS weights that don't sum up to 1
                // (because the BSDF sample, that should have weight 1 [or to be precise: 1 / nb_bsdf_samples]
                // will have weight 1 / (1 + nb_light_samples) [or to be precise: 1 / (nb_bsdf_samples + nb_light_samples)]
                // and this is going to cause darkening as the number of light samples grows)
                light_pdf *= !refraction_sampled;

                float mis_weight = balance_heuristic(bsdf_sample_pdf, nb_bsdf_candidates, light_pdf, nb_light_candidates);
                candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

                bsdf_RIS_sample.emissive_triangle_index = bsdf_ray_hit_info.primitive_index;
                bsdf_RIS_sample.point_on_light_source = bsdf_ray_hit_info.inter_point;
                bsdf_RIS_sample.is_bsdf_sample = true;
                bsdf_RIS_sample.bsdf_sample_contribution = bsdf_color;
                bsdf_RIS_sample.bsdf_sample_cosine_term = cosine_at_evaluated_point;
                bsdf_RIS_sample.target_function = target_function;
            }
        }

        // TODO optimize here and if we keep the sample of the BSDF, we don't have to re-test for visibility at the end of the function
        // because a BSDF sample can only be chosen if it's unoccluded
        reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
    }

    reservoir.end();
    return reservoir;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_lights_RIS(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    RISReservoir reservoir = sample_bsdf_and_lights_RIS_reservoir(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);

    return evaluate_reservoir_sample(render_data, ray_payload, closest_hit_info.inter_point, closest_hit_info.shading_normal, view_direction, reservoir);
}

#endif
