/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHTS_H
#define DEVICE_LIGHTS_H

#include "HostDeviceCommon/KernelOptions.h"
#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

/**
 *  -------------------------------------------------
 *  State of caustics rendering so far:
 *     - No proper caustic solver. This means that:
 *          - For a refractive sphere with a light inside and the camera viewing everything from the outside
 *              the camera ray that hits the sphere may be able to sample the light by sampling a refraction with the BSDF ray.
 *              That BSDF can then hit the light which makes for a successful BSDF sampling --> we can somewhat sample the light
 *              from the outside. However, from the camera ray first hit, light sampling will not be able to sample the light
 *              that is inside the surface because, well, there's the surface in between. This is where we need a proper caustic solver.
 *              In the meantime, this situation kind of breaks Multiple Importance Sampling (and Resampled Importance Sampling with MIS)
 *              because BSDF sampling can sample the light (in this situation at least) where light sampling cannot. This biases MIS weights
 *              and gives darker and darker results as the number of light samples increases relative to the number of BSDF samples (with RIS
 *              for example). 
 *              Although this is all biased and a proper caustic solver is required, this is still better than nothing (otherwise we can
 *              hardly have caustics at all) so we'll keep it like that waiting for some MNEE/SMS/...
 *  -------------------------------------------------
 */

HIPRT_HOST_DEVICE HIPRT_INLINE float3 sample_one_emissive_triangle(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    int random_index = random_number_generator.random_index(render_data.buffers.emissive_triangles_count);
    int triangle_index = render_data.buffers.emissive_triangles_indices[random_index];

    float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 2]];

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_r1 = sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;

    float3 random_point_on_triangle = vertex_A + AB * u + AC * v;

    float3 normal = hippt::cross(AB, AC);
    float length_normal = hippt::length(normal);
    if (length_normal <= 1.0e-6f)
    {
        // Can happen with very small triangles
        pdf = 0.0f;

        return make_float3(0, 0, 0);
    }


    light_info.emissive_triangle_index = triangle_index;
    light_info.light_source_normal = normal / length_normal; // Normalization
    light_info.light_area = length_normal * 0.5f;
    light_info.emission = render_data.buffers.materials_buffer[render_data.buffers.material_indices[triangle_index]].emission;

    pdf = 1.0f / light_info.light_area;

    return random_point_on_triangle;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float triangle_area(const HIPRTRenderData& render_data, int triangle_index)
{
    float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 0]];
    float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 1]];
    float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index * 3 + 2]];

    float3 AB = vertex_B - vertex_A;
    float3 AC = vertex_C - vertex_A;

    return hippt::length(hippt::cross(AB, AC)) / 2.0f;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_no_MIS(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    float light_sample_pdf;
    LightSourceInformation light_source_info;
    ColorRGB32F light_source_radiance;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);
    if (!(light_sample_pdf > 0.0f))
        // Can happen for very small triangles
        return ColorRGB32F(0.0f);

    float3 shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
    float3 shadow_ray_direction = random_light_point - shadow_ray_origin;
    float distance_to_light = hippt::length(shadow_ray_direction);
    float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = shadow_ray_origin;
    shadow_ray.direction = shadow_ray_direction_normalized;

    // abs() here to allow backfacing light sources
    float dot_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -shadow_ray.direction));
    if (dot_light_source > 0.0f)
    {
        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

        if (!in_shadow)
        {
            // Conversion to solid angle from surface area measure
            light_sample_pdf *= distance_to_light * distance_to_light;
            light_sample_pdf /= dot_light_source;
            light_sample_pdf /= render_data.buffers.emissive_triangles_count;

            float brdf_pdf;
            RayVolumeState trash_volume_state = ray_payload.volume_state;
            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, brdf_pdf);
            if (brdf_pdf != 0.0f)
            {
                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance = light_source_info.emission * cosine_term * bsdf_color / light_sample_pdf;
            }
        }
    }

    return light_source_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_bsdf(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // Pushing the intersection point outside the surface (if we're already outside)
    // or inside the surface (if we're inside the surface)
    // We'll use that intersection point as the origin of our shadow rays
    bool inside_surface = hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    ColorRGB32F bsdf_radiance = ColorRGB32F(0.0f);

    float direction_pdf;
    float3 sampled_brdf_direction;
    RayVolumeState trash_volume_state = ray_payload.volume_state;
    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);

    bool refraction_sampled = hippt::dot(sampled_brdf_direction, closest_hit_info.shading_normal * inside_surface_multiplier) < 0;
    if (direction_pdf > 0.0f)
    {
        hiprtRay new_ray;
        new_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
        new_ray.direction = sampled_brdf_direction;

        if (refraction_sampled)
        {
            // If we sampled a refraction, we're pushing the origin of the shadow ray "through"
            // the surface so that our ray can refract inside the surface
            //
            // If we don't do that and we just use the 'evaluated_point' computed at the very
            // beginning of the function, we're going to intersect ourselves

            new_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier * -1.0f;
        }

        HitInfo new_ray_hit_info;
        RayPayload bsdf_ray_payload;
        bool inter_found = trace_ray(render_data, new_ray, bsdf_ray_payload, new_ray_hit_info);

        // Checking that we did hit something and if we hit something,
        // it needs to be the light that we're currently sampling
        if (inter_found && bsdf_ray_payload.material.is_emissive())
        {
            float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            bsdf_radiance = bsdf_color * cosine_term * bsdf_ray_payload.material.emission / direction_pdf;
        }
    }

    return bsdf_radiance;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light_MIS(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // Pushing the intersection point outside the surface (if we're already outside)
    // or inside the surface (if we're inside the surface)
    // We'll use that intersection point as the origin of our shadow rays
    bool inside_surface = hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    float light_sample_pdf;
    ColorRGB32F light_source_radiance_mis;
    LightSourceInformation light_source_info;
    float3 random_light_point = sample_one_emissive_triangle(render_data, random_number_generator, light_sample_pdf, light_source_info);
    if (light_sample_pdf <= 0.0f)
        // Can happen for very small triangles
        return ColorRGB32F(0.0f);

    float3 shadow_ray_direction = random_light_point - evaluated_point;
    float distance_to_light = hippt::length(shadow_ray_direction);
    float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

    hiprtRay shadow_ray;
    shadow_ray.origin = evaluated_point;
    shadow_ray.direction = shadow_ray_direction_normalized;

    // abs() here to allow backfacing light sources
    float dot_light_source = hippt::abs(hippt::dot(light_source_info.light_source_normal, -shadow_ray.direction));
    if (dot_light_source > 0.0f)
    {
        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

        if (!in_shadow)
        {
            // Conversion to solid angle from surface area measure
            light_sample_pdf *= distance_to_light * distance_to_light;
            light_sample_pdf /= dot_light_source;
            light_sample_pdf /= render_data.buffers.emissive_triangles_count;

            float bsdf_pdf;
            RayVolumeState trash_volume_state = ray_payload.volume_state;
            ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, bsdf_pdf);
            if (bsdf_pdf != 0.0f)
            {
                float mis_weight = power_heuristic(light_sample_pdf, bsdf_pdf);

                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
                light_source_radiance_mis = bsdf_color * cosine_term * light_source_info.emission * mis_weight / light_sample_pdf;
            }
        }
    }

    ColorRGB32F bsdf_radiance_mis;

    float direction_pdf;
    float3 sampled_brdf_direction;
    float3 bsdf_shadow_ray_origin = evaluated_point;
    RayVolumeState trash_volume_state = ray_payload.volume_state;
    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);
    bool refraction_sampled = hippt::dot(sampled_brdf_direction, closest_hit_info.shading_normal * inside_surface_multiplier) < 0;
    if (refraction_sampled)
    {
        // If we sampled a refraction, we're pushing the origin of the shadow ray "through"
        // the surface so that our ray can refract inside the surface
        //
        // If we don't do that and we just use the 'evaluated_point' computed at the very
        // beginning of the function, we're going to intersect ourselves

        bsdf_shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier * -1.0f;
    }

    if (direction_pdf > 0)
    {
        hiprtRay new_ray;
        new_ray.origin = bsdf_shadow_ray_origin;
        new_ray.direction = sampled_brdf_direction;

        HitInfo new_ray_hit_info;
        RayPayload bsdf_ray_payload;
        bool inter_found = trace_ray(render_data, new_ray, bsdf_ray_payload, new_ray_hit_info);

        // Checking that we did hit something and if we hit something,
        // it needs to be the light that we're currently sampling
        if (inter_found && bsdf_ray_payload.material.is_emissive())
        {
            // abs() here to allow double sided emissive geometry.
            // Without abs() here:
            //  - We could be hitting the back of an emissive triangle 
            //  --> triangle normal not facing the same way 
            //  --> cos_angle negative
            float cos_angle_light = hippt::abs(hippt::dot(new_ray_hit_info.shading_normal, -sampled_brdf_direction));

            float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
            float light_pdf = distance_squared / (light_source_info.light_area * cos_angle_light) / render_data.buffers.emissive_triangles_count;
            float mis_weight = power_heuristic(direction_pdf, light_pdf);

            // Using abs here because we want the dot product to be positive.
            // You may be thinking that if we're doing this, then we're not going to discard BSDF
            // sampled direction that are below the surface (whereas we should discard them).
            // That would be correct but bsdf_dispatcher_sample return a PDF == 0.0f if a bad
            // direction was sampled and if the PDF is 0.0f, we never get to this line of code
            // you're reading. If we are here, this is because we sampled a direction that is
            // correct for the BSDF. Even if the direction is correct, the dot product may be
            // negative in the case of refractions / total internal reflections and so in this case,
            // we'll need to negative the dot product for it to be positive
            float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            //float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
            bsdf_radiance_mis = bsdf_color * cosine_term * light_source_info.emission * mis_weight / direction_pdf;
        }
    }

    // Because we're sampling only 1 light out of all the lights of the
    // scene, the probability of having chosen that light is: 1 / numberOfLights
    // This must be factored in the PDF of sampling that light which means that we must
    // divide by 1 / numberOfLights => multiply by numberOfLights
    return light_source_radiance_mis + bsdf_radiance_mis;
}

struct ReservoirSample
{
    // Light sample
    float3 point_on_light_source = { 0, 0, 0 };
    ColorRGB32F emission = { 0.0f, 0.0f, 0.0f };
};

struct Reservoir
{
    HIPRT_HOST_DEVICE bool update(ReservoirSample new_sample, float weight, Xorshift32Generator& random_number_generator)
    {
        M++;
        weight_sum += weight;

        if (random_number_generator() < weight / weight_sum)
        {
            sample = new_sample;

            return true;
        }

        return false;
    }

    HIPRT_HOST_DEVICE ReservoirSample get_sample()
    {
        return sample;
    }

    unsigned int M = 0;
    float weight_sum = 0.0f;

    ReservoirSample sample;
};

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_bsdf_and_lights_RIS(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // Pushing the intersection point outside the surface (if we're already outside)
    // or inside the surface (if we're inside the surface)
    // We'll use that intersection point as the origin of our shadow rays
    bool inside_surface = hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0;
    float inside_surface_multiplier = inside_surface ? -1.0f : 1.0f;
    float3 evaluated_point = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f * inside_surface_multiplier;

    // If we're rendering at low resolution, only doing 1 candidate of each for better framerates
    int nb_light_candidates = render_data.render_settings.render_low_resolution ? 1 : render_data.render_settings.ris_number_of_light_candidates;
    int nb_bsdf_candidates = render_data.render_settings.render_low_resolution ? 1 : render_data.render_settings.ris_number_of_bsdf_candidates;

    // Sampling candidates with weighted reservoir sampling
    Reservoir reservoir;
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
                // Converting the PDF from area measure to solid angle measure requires dividing by
                // cos(theta) / dist^2. Dividing by that factor is equal to multiplying by the inverse
                // which is what we're doing here
                light_sample_pdf *= distance_to_light * distance_to_light;
                light_sample_pdf /= cosine_at_light_source;
                light_sample_pdf /= render_data.buffers.emissive_triangles_count;

                float geometry_term = 1.0f / (distance_to_light * distance_to_light) * cosine_at_light_source * cosine_at_evaluated_point;
                target_function = bsdf_color.length() * light_source_info.emission.length() * cosine_at_evaluated_point;

#if RISUseVisiblityTargetFunction == RIS_USE_VISIBILITY_TRUE
                if (!render_data.render_settings.render_low_resolution)
                {
                    // Only doing visiblity if we're render at low resolution
                    // (meaning we're moving the camera) for better movement framerates

                    hiprtRay shadow_ray;
                    shadow_ray.origin = evaluated_point;
                    shadow_ray.direction = to_light_direction;

                    bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

                    target_function *= visible;
                }
#endif

                float mis_weight = balance_heuristic(light_sample_pdf, nb_light_candidates, bsdf_pdf, nb_bsdf_candidates);
                candidate_weight = mis_weight * target_function / light_sample_pdf;
            }
        }

        ReservoirSample sample;
        sample.point_on_light_source = random_light_point;
        sample.emission = light_source_info.emission;

        reservoir.update(sample, candidate_weight, random_number_generator);
    }

    // Whether or not a BSDF sample has been retained by the reservoir
    bool bsdf_sample_picked = false;
    float bsdf_sample_cosine_at_evaluated_point = 0.0f;
    ColorRGB32F bsdf_sample_contribution;
    for (int i = 0; i < nb_bsdf_candidates; i++)
    {
        float bsdf_sample_pdf = 0.0f;
        float target_function = 0.0f;
        float candidate_weight = 0.0f;
        float cosine_light_source = 0.0f;
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
        ReservoirSample new_sample;
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

                // abs() here to allow backfacing lights
                cosine_light_source = hippt::abs(hippt::dot(bsdf_ray_hit_info.shading_normal, -sampled_direction));
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

                //float geometry_term = 1.0f / (bsdf_ray_hit_info.t * bsdf_ray_hit_info.t) * cosine_at_evaluated_point * cosine_light_source;
                target_function = bsdf_color.length() * bsdf_ray_payload.material.emission.length() * cosine_at_evaluated_point;

                float light_area = triangle_area(render_data, bsdf_ray_hit_info.primitive_index);
                float light_pdf = bsdf_ray_hit_info.t * bsdf_ray_hit_info.t / cosine_light_source;
                light_pdf /= light_area;
                light_pdf /= render_data.buffers.emissive_triangles_count;

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

                new_sample.emission = bsdf_ray_payload.material.emission;
                new_sample.point_on_light_source = bsdf_ray_hit_info.inter_point;
            }
        }

        // TODO optimize here and if we keep the sample of the BSDF, we don't have to re-test for visibility at the end of the function
        // because a BSDF sample can only be chosen if it's unoccluded
        if (reservoir.update(new_sample, candidate_weight, random_number_generator))
        {
            bsdf_sample_picked = true;
            bsdf_sample_cosine_at_evaluated_point = cosine_at_evaluated_point;
            bsdf_sample_contribution = bsdf_color;
        }
    }

    ColorRGB32F final_color;

    if (reservoir.weight_sum == 0.0f)
        // No valid sample means no light contribution
        return ColorRGB32F(0.0f);

    // Getting the sample
    ReservoirSample sample = reservoir.get_sample();

    bool in_shadow;
    float distance_to_light;
    float3 shadow_ray_direction = sample.point_on_light_source - evaluated_point;
    float3 shadow_ray_direction_normalized = shadow_ray_direction / (distance_to_light = hippt::length(shadow_ray_direction));
    if (bsdf_sample_picked)
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

        if (bsdf_sample_picked)
        {
            bsdf_color = bsdf_sample_contribution;

            // If we picked a BSDF sample, we're using the already computed cosine term
            // because it's annoying to recompute it (we have to know if the BSDF is a refarction
            // sample or not)
            cosine_at_evaluated_point = bsdf_sample_cosine_at_evaluated_point;
        }
        else
        {
            bsdf_color = bsdf_dispatcher_eval(render_data.buffers.materials_buffer, ray_payload.material, trash_volume_state, view_direction, closest_hit_info.shading_normal, shadow_ray_direction_normalized, bsdf_pdf);

            cosine_at_evaluated_point = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal * inside_surface_multiplier, shadow_ray_direction_normalized));
        }

        if (cosine_at_evaluated_point > 0.0f)
        {
            // Visibility of the target function is already included because we can only be here it the light wasn't occluded
            float target_function = bsdf_color.length() * sample.emission.length() * cosine_at_evaluated_point;
            float UCW = 1.0f / target_function * reservoir.weight_sum;

            final_color = bsdf_color * UCW * sample.emission * cosine_at_evaluated_point;
        }
    }

    return final_color;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_one_light(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        // No emmisive geometry in the scene to sample
        return ColorRGB32F(0.0f);

    if (ray_payload.material.emission.r != 0.0f || ray_payload.material.emission.g != 0.0f || ray_payload.material.emission.b != 0.0f)
        // We're not sampling direct lighting if we're already on an
        // emissive surface
        return ColorRGB32F(0.0f);

#if DirectLightSamplingStrategy == LSS_NO_DIRECT_LIGHT_SAMPLING
    return ColorRGB32F(0.0f);
#elif DirectLightSamplingStrategy == LSS_UNIFORM_ONE_LIGHT
    return sample_one_light_no_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_BSDF
    return sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_MIS_LIGHT_BSDF
    return sample_one_light_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightSamplingStrategy == LSS_RIS_BSDF_AND_LIGHT
    return sample_bsdf_and_lights_RIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#endif
}

#endif
