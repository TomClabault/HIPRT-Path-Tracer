/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_FINAL_SHADING_H
#define DEVICE_RESTIR_DI_FINAL_SHADING_H

#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/TriangleSampling.h"
#include "Device/includes/TriangleLoadUtils.h"

#include "Device/includes/HitInfo.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/RenderData.h"

// TODO make some simplification assuming that ReSTIR DI is never inside a surface (the camera being inside a surface may be an annoying case to handle)
HIPRT_DEVICE ColorRGB32F evaluate_ReSTIR_DI_reservoir(const HIPRTRenderData& render_data,
													  RayPayload& ray_payload,
													  const HitInfo& closest_hit_info,
													  const float3_t& view_direction,
													  const ReSTIRDIReservoir& reservoir,
													  Xorshift32Generator& random_number_generator)
{
	ColorRGB32F final_color;

	if (reservoir.UCW <= 0.0f)
		// No valid sample means no light contribution
		return ColorRGB32F(0.0f);

	ReSTIRDIReservoirSample sample = reservoir.sample;

	float distance_to_light;

	float3_t shadow_ray_direction;
	if (sample.is_envmap_sample())
	{
		shadow_ray_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, sample.point_on_light_source);
		distance_to_light	 = 1.0e35f;
	}
	else
	{
		shadow_ray_direction = sample.point_on_light_source - closest_hit_info.inter_point;
		shadow_ray_direction = shadow_ray_direction / (distance_to_light = hippt::length(shadow_ray_direction));
	}

	bool in_shadow = false;
	if (sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED)
		in_shadow = false;
	else if (render_data.render_settings.restir_di_settings.do_final_shading_visibility)
	{
		hiprtRay shadow_ray;
		shadow_ray.origin	 = closest_hit_info.inter_point;
		shadow_ray.direction = shadow_ray_direction;

		in_shadow = evaluate_shadow_ray_occluded(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index, random_number_generator);
	}

	if (!in_shadow)
	{
		float bsdf_pdf;
		float cosine_at_evaluated_point;

		BSDFIncidentLightInfo incident_light_info = sample.flags_to_BSDF_incident_light_info();
		BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray_direction, incident_light_info,
								 ray_payload.volume_state, false, ray_payload.material, 0.0f, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
		ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

		cosine_at_evaluated_point = hippt::dot(closest_hit_info.shading_normal, shadow_ray_direction);
		if (sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_SAMPLED_FROM_GLASS_REFRACT_LOBE)
			// We're not allowing samples that are below the surface
			// UNLESS it's a BSDF refraction sample in which case it's valid
			// so we're restoring the cosine term to be > 0.0f so that it passes
			// the if() condition below
			cosine_at_evaluated_point = hippt::abs(cosine_at_evaluated_point);

		if (cosine_at_evaluated_point > 0.0f)
		{
			ColorRGB32F sample_emission;

			if (sample.is_envmap_sample())
			{
				float envmap_pdf;
				sample_emission = envmap_eval(render_data, shadow_ray_direction, envmap_pdf);
			}
			else
			{
				sample_emission = get_triangle_emission_at_point(render_data, sample.emissive_triangle_global_index, sample.point_on_light_source);
			}

			float area_measure_to_solid_angle_conversion;
			if (sample.is_envmap_sample())
				area_measure_to_solid_angle_conversion = 1.0f;
			else
			{
				float3_t emissive_triangle_normal = hippt::normalize(triangle_load_normal_not_normalized(render_data, sample.emissive_triangle_global_index));
				area_measure_to_solid_angle_conversion = compute_cosine_term_at_light_source(emissive_triangle_normal, -shadow_ray_direction);
				area_measure_to_solid_angle_conversion /= hippt::square(distance_to_light);
			}

			final_color = bsdf_color * reservoir.UCW * sample_emission * cosine_at_evaluated_point * area_measure_to_solid_angle_conversion;
		}
	}

	return final_color;
}

HIPRT_DEVICE void validate_reservoir(const HIPRTRenderData& render_data, ReSTIRDIReservoir& reservoir)
{
	if (reservoir.sample.is_envmap_sample() && render_data.world_settings.ambient_light_type != AmbientLightType::ENVMAP)
		// Killing the reservoir if it was an envmap sample but the envmap is not used anymore
		reservoir.UCW = 0.0f;
}

HIPRT_DEVICE ColorRGB32F sample_light_ReSTIR_DI(const HIPRTRenderData& render_data,
												RayPayload& ray_payload,
												const HitInfo closest_hit_info,
												const float3_t& view_direction,
												Xorshift32Generator& random_number_generator,
												int2_t pixel_coords)
{
	int pixel_index = pixel_coords.x + pixel_coords.y * render_data.render_settings.render_resolution.x;

	// Because the spatial reuse pass runs last, the output buffer of the spatial
	// pass contains the reservoir whose sample we're going to shade
	ReSTIRDIReservoir& reservoir = render_data.render_settings.restir_di_settings.restir_output_reservoirs[pixel_index];

	// Validates the reservoir i.e. kills the reservoir if it isn't valid
	// anymore i.e. if it refers to a light that doesn't exist anymore
	validate_reservoir(render_data, reservoir);

	return evaluate_ReSTIR_DI_reservoir(render_data, ray_payload, closest_hit_info, view_direction, reservoir, random_number_generator);
}

#endif // #ifndef DEVICE_RESTIR_DI_FINAL_SHADING_H
