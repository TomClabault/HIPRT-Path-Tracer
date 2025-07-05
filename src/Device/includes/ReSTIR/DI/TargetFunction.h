/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_TARGET_FUNCTION_H
#define DEVICE_RESTIR_DI_TARGET_FUNCTION_H

#include "Device/includes/ReSTIR/Utils.h"
#include "HostDeviceCommon/RenderData.h"


HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReSTIR_DI_get_light_sample_direction(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, 
	float3 surface_shading_point, float& out_distance_to_light)
{
	float3 sample_direction;
	if (sample.is_envmap_sample())
	{
		sample_direction = matrix_X_vec(render_data.world_settings.envmap_to_world_matrix, sample.point_on_light_source);
		out_distance_to_light = 1.0e35f;
	}
	else
	{
		sample_direction = sample.point_on_light_source - surface_shading_point;
		sample_direction = sample_direction / (out_distance_to_light = hippt::length(sample_direction));
	}

	return sample_direction;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F ReSTIR_DI_get_light_sample_emission(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, float3 sample_direction)
{
	ColorRGB32F sample_emission;
	if (sample.is_envmap_sample())
	{
		float envmap_pdf;
		sample_emission = envmap_eval(render_data, sample_direction, envmap_pdf);
	}
	else
	{
		int material_index = render_data.buffers.material_indices[sample.emissive_triangle_index];
		sample_emission = render_data.buffers.materials_buffer.get_emission(material_index);
	}

	return sample_emission;
}

template <bool withVisibility>
HIPRT_HOST_DEVICE HIPRT_INLINE float ReSTIR_DI_evaluate_target_function(const HIPRTRenderData& render_data, const ReSTIRDISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator)
{
	if (sample.emissive_triangle_index == -1 && !sample.is_envmap_sample())
		// No sample
		return 0.0f;

	float bsdf_pdf;
	float distance_to_light;
	float3 sample_direction = ReSTIR_DI_get_light_sample_direction(render_data, sample, surface.shading_point, distance_to_light);

	float cosine_term = hippt::dot(surface.shading_normal, sample_direction);
	if (cosine_term <= 0.0f)
		// If the cosine term is 0.0f, the rest is going to be multiplied by that zero-cosine-term
		// and everything is going to be 0.0f anyway so we can return already
		return 0.0f;

	BSDFIncidentLightInfo incident_light_info = sample.flags_to_BSDF_incident_light_info();
	BSDFContext bsdf_context(surface.view_direction, surface.shading_normal, surface.geometric_normal, sample_direction, incident_light_info, surface.ray_volume_state, false, surface.material, /* bounce. Always 0 for ReSTIR DI */ 0, 0.0f);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
	ColorRGB32F sample_emission = ReSTIR_DI_get_light_sample_emission(render_data, sample, sample_direction);

	float geometry_term = 1.0f;
	if (!sample.is_envmap_sample())
	{
		float3 emissive_triangle_normal = hippt::normalize(get_triangle_normal_not_normalized(render_data, sample.emissive_triangle_index));
		geometry_term = compute_cosine_term_at_light_source(emissive_triangle_normal, -sample_direction);
		geometry_term /= hippt::square(distance_to_light);
	}

	float target_function = (bsdf_color * sample_emission * cosine_term * geometry_term).luminance();
	if (target_function == 0.0f)
		// Quick exit because computing the visiblity that follows isn't going
		// to change anything to the fact that we have 0.0f target function here
		return 0.0f;

	if constexpr (withVisibility)
	{
		hiprtRay shadow_ray;
		shadow_ray.origin = surface.shading_point;
		shadow_ray.direction = sample_direction;

		bool visible = !evaluate_shadow_ray_occluded(render_data, shadow_ray, distance_to_light, surface.primitive_index, /* bounce. Always 0 for ReSTIR DI*/ 0, random_number_generator);

		target_function *= visible;
	}

	return target_function;
}

#endif
