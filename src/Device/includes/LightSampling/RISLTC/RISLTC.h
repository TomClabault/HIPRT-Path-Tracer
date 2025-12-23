/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INLCUDES_LIGHT_SAMPLING_RISLTC_RISLTC_H
#define DEVICE_INLCUDES_LIGHT_SAMPLING_RISLTC_RISLTC_H

#include "Device/includes/BSDFs/BSDFContext.h"
#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/HitInfo.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/RISLTC/RISLTCReservoir.h"
#include "Device/includes/LightSampling/TriangleSampling.h"
#include "Device/includes/LightSampling/TriangleEmissiveSampling.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE ColorRGB32F evaluate_RISLTC_reservoir_sample(HIPRTRenderData& render_data, RayPayload& ray_payload,
    const HitInfo& closest_hit_info, const float3& view_direction,
    const RISLTCReservoir& reservoir, Xorshift32Generator& random_number_generator)
{
    ColorRGB32F final_color;

    if (reservoir.UCW <= 0.0f)
        // No valid sample means no light contribution
        return ColorRGB32F(0.0f);

    RISLTCSample sample = reservoir.sample;

    LightSamplePointInformation light_sample_info = sample_point_on_light_and_fill_light_sample_information(render_data,
         closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
         ray_payload.material,
		sample.light_index, random_number_generator);
    if (light_sample_info.area_measure_pdf == 0.0f)
        return ColorRGB32F();

    bool in_shadow;
    float distance_to_light;
    float3 evaluated_point = closest_hit_info.inter_point;
    float3 shadow_ray_direction = light_sample_info.point_on_light - evaluated_point;
    float3 shadow_ray_direction_normalized = shadow_ray_direction / (distance_to_light = hippt::length(shadow_ray_direction));

    NEEPlusPlusContext nee_plus_plus_context;
    if (sample.is_bsdf_sample)
        // A BSDF sample that has been picked by RIS cannot be occluded otherwise
        // it would have a weight of 0 and would never be picked by RIS
        in_shadow = false;
    else
    {
        hiprtRay shadow_ray;
        shadow_ray.origin = evaluated_point;
        shadow_ray.direction = shadow_ray_direction_normalized;

        nee_plus_plus_context.point_on_light = light_sample_info.point_on_light;
        nee_plus_plus_context.shaded_point = shadow_ray.origin;
        in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index, nee_plus_plus_context, random_number_generator, ray_payload.bounce);
    }

    if (!in_shadow)
    {
        float bsdf_pdf;
        float cosine_at_evaluated_point;
        ColorRGB32F bsdf_color;

        if (sample.is_bsdf_sample)
        {
            // If we picked a BSDF sample, we're using the already computed cosine term and color
            // because it's annoying to recompute it (we have to know if the BSDF is a refraction
            // sample or not)
            bsdf_color = sample.bsdf_sample_contribution;
        }
        else
        {
            BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
            BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray_direction_normalized, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
            bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
        }

		float point_pdf_solid_angle = area_to_solid_angle_pdf(light_sample_info.area_measure_pdf, distance_to_light, compute_cosine_term_at_light_source(light_sample_info.light_source_normal, -shadow_ray_direction_normalized));
        if (point_pdf_solid_angle == 0.0f)
            return ColorRGB32F();

        final_color = bsdf_color * reservoir.UCW / point_pdf_solid_angle * sample.emission * hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray_direction_normalized));
        if (!sample.is_bsdf_sample)
            final_color /= nee_plus_plus_context.unoccluded_probability;

        sanity_check<true>(render_data, final_color, -1, -1);
    }

    return final_color;
}

HIPRT_DEVICE RISLTCReservoir sample_bsdf_and_lights_RISLTC_reservoir(const HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo& closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    // If we're rendering at low resolution, only doing 1 candidate of each
    // for better interactive framerates
    int nb_light_candidates = render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.risltc_settings.number_of_light_candidates;

#if DirectLightSamplingBaseStrategy == LSS_BASE_LIGHT_TREE_ATS && LightTreeATSDoSplitting == KERNEL_OPTION_TRUE
    // BSDF MIS isn't allowed with the light tree & splitting, we don't have the PDF for the light tree
    // splitting implementation so it's biased
    int nb_bsdf_candidates = 0;
#else
    int nb_bsdf_candidates = render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.risltc_settings.number_of_bsdf_candidates;
#endif
    if (!ray_payload.material.can_do_light_sampling())
        nb_light_candidates = 0;

    // Sampling candidates with weighted reservoir sampling
    RISLTCReservoir reservoir;
    // Dividing by DirectLightSampleCount<DirectLightSamplingBaseStrategy>() here because 
    for (int light_candidate = 0; light_candidate < nb_light_candidates; light_candidate++)
    {
        LightSampleArray<DirectLightSampleCount<DirectLightSamplingBaseStrategy>()> light_samples = sample_one_light(render_data,
            closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
            closest_hit_info.primitive_index, ray_payload,
            random_number_generator);

        for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingBaseStrategy>(); i++)
        {
            LightSampleInformation& light_sample_info = light_samples[i];

            float3 vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[light_sample_info.emissive_triangle_global_index * 3 + 0]];
            float3 vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[light_sample_info.emissive_triangle_global_index * 3 + 1]];
            float3 vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[light_sample_info.emissive_triangle_global_index * 3 + 2]];

            float ltc_coat = 0.0f;
            float ltc_specular = 0.0f;
            float ltc_metallic = 0.0f;
            float ltc_diffuse = 0.0f;

#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
            ltc_diffuse = evaluate_ltc(render_data,
                vertex_A, vertex_B, vertex_C,
                closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
                ray_payload.material,
                LTCLobe::DIFFUSE_LOBE);
#else
            if (ray_payload.material.coat > 0.0f)
                ltc_coat = evaluate_ltc(render_data,
    				vertex_A, vertex_B, vertex_C,
				    closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
    				ray_payload.material,
				    LTCLobe::COAT_LOBE);

            if (ray_payload.material.specular > 0.0f && ray_payload.material.roughness < render_data.bsdfs_data.ltcs_data.specular_ltc_maximum_roughness)
                ltc_specular = evaluate_ltc(render_data,
                    vertex_A, vertex_B, vertex_C,
				    closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
                    ray_payload.material,
				    LTCLobe::SPECULAR_LOBE);

            if (ray_payload.material.metallic > 0.0f && ray_payload.material.roughness < render_data.bsdfs_data.ltcs_data.specular_ltc_maximum_roughness)
                ltc_metallic = evaluate_ltc(render_data,
                    vertex_A, vertex_B, vertex_C,
				    closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
				    ray_payload.material,
				    LTCLobe::METALLIC_LOBE);

            ltc_diffuse = evaluate_ltc(render_data,
				vertex_A, vertex_B, vertex_C,
                closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
				ray_payload.material,
				LTCLobe::DIFFUSE_LOBE);
#endif

			float target_function = 
                ltc_coat * ray_payload.material.coat + 
                ltc_specular * ray_payload.material.specular + 
                ltc_metallic * ray_payload.material.metallic + 
                ltc_diffuse * ray_payload.material.base_color.luminance();

            float mis_weight = 1.0f / nb_light_candidates;
            float candidate_weight = mis_weight * target_function / light_sample_info.pdf;

            RISLTCSample light_RIS_sample;
            light_RIS_sample.light_index = light_sample_info.emissive_triangle_global_index;
            light_RIS_sample.is_bsdf_sample = false;
            light_RIS_sample.target_function = target_function;
			light_RIS_sample.emission = triangle_load_emission(render_data, light_sample_info.emissive_triangle_global_index);

            reservoir.add_one_candidate(light_RIS_sample, candidate_weight, random_number_generator);
            reservoir.sanity_check();
        }
    }

    //// Whether or not a BSDF sample has been retained by the reservoir
    //for (int i = 0; i < nb_bsdf_candidates; i++)
    //{
    //    float bsdf_sample_pdf = 0.0f;
    //    float target_function = 0.0f;
    //    float candidate_weight = 0.0f;
    //    float3 sampled_bsdf_direction;

    //    BSDFIncidentLightInfo incident_light_info;
    //    BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
    //    ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

    //    RISSample bsdf_RIS_sample;
    //    if (bsdf_sample_pdf > 0.0f)
    //    {
    //        hiprtRay bsdf_ray;
    //        bsdf_ray.origin = closest_hit_info.inter_point;
    //        bsdf_ray.direction = sampled_bsdf_direction;

    //        BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
    //        bool hit_found = evaluate_bsdf_light_sample_ray(render_data, bsdf_ray, 1.0e35f, shadow_light_ray_hit_info, closest_hit_info.primitive_index, ray_payload.bounce, random_number_generator);

    //        if (hit_found && !shadow_light_ray_hit_info.hit_emission.is_black() && compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction) > 0.0f)
    //        {
    //            // If we intersected an emissive material, compute the weight. 
    //            // Otherwise, the weight is 0 because of the emision being 0 so we just don't compute it

    //            // Using abs here because we want the dot product to be positive.
    //            // You may be thinking that if we're doing this, then we're not going to discard BSDF
    //            // sampled direction that are below the surface (whereas we should discard them).
    //            // That would be correct but bsdf_dispatcher_sample return a PDF == 0.0f if a bad
    //            // direction was sampled and if the PDF is 0.0f, we never get to this line of code
    //            // you're reading. If we are here, this is because we sampled a direction that is
    //            // correct for the BSDF. Even if the direction is correct, the dot product may be
    //            // negative in the case of refractions / total internal reflections and so in this case,
    //            // we'll need to abs() the dot product for it to be positive
    //            float cosine_at_evaluated_point = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_bsdf_direction));

    //            // Our target function does not include the geometry term because we're integrating
    //            // in solid angle. The geometry term in the target function ( / in the integrand) is only
    //            // for surface area direct lighting integration
    //            ColorRGB32F light_contribution = bsdf_color * shadow_light_ray_hit_info.hit_emission * cosine_at_evaluated_point;
    //            target_function = light_contribution.luminance();

    //            float light_pdf = pdf_of_emissive_triangle_hit_solid_angle(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
    //                ray_payload.material,
    //                shadow_light_ray_hit_info, sampled_bsdf_direction);
    //            float mis_weight = balance_heuristic(bsdf_sample_pdf, nb_light_candidates, light_pdf, nb_light_candidates);
    //            candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

    //            bsdf_RIS_sample.emission = shadow_light_ray_hit_info.hit_emission;
    //            bsdf_RIS_sample.point_on_light_source = bsdf_ray.origin + bsdf_ray.direction * shadow_light_ray_hit_info.hit_distance;
    //            bsdf_RIS_sample.is_bsdf_sample = true;
    //            bsdf_RIS_sample.bsdf_sample_contribution = bsdf_color;
    //            bsdf_RIS_sample.bsdf_sample_cosine_term = cosine_at_evaluated_point;
    //            bsdf_RIS_sample.target_function = target_function;
    //        }
    //    }

    //    reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
    //    reservoir.sanity_check();
    //}

    reservoir.end();
    return reservoir;
}

HIPRT_DEVICE ColorRGB32F sample_lights_RISLTC(HIPRTRenderData& render_data, RayPayload& ray_payload, const HitInfo& closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator)
{
    if (render_data.buffers.emissive_triangles_count == 0)
        return ColorRGB32F(0.0f);

    RISLTCReservoir reservoir = sample_bsdf_and_lights_RISLTC_reservoir(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);

    return evaluate_RISLTC_reservoir_sample(render_data, ray_payload,
        closest_hit_info, view_direction,
        reservoir, random_number_generator);
}

#endif
