/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_TARGET_FUNCTION_H
#define DEVICE_RESTIR_GI_TARGET_FUNCTION_H

#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/Surface.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "HostDeviceCommon/RenderData.h"

template <bool withVisiblity, bool resamplingNeighbor = true>
HIPRT_HOST_DEVICE float ReSTIR_GI_evaluate_target_function(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator, bool DEBUG = true, int DEBUG_X = -1, int DEBUG_Y = -1)
{
	float distance_to_sample_point;
	float3 incident_light_direction;
	if (sample.is_envmap_path())
	{
		// For envmap path, the direction is stored in the 'sample_point' value
		incident_light_direction = sample.sample_point;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point
		incident_light_direction = sample.sample_point - surface.shading_point;
		distance_to_sample_point = hippt::length(incident_light_direction);
		if (distance_to_sample_point <= 1.0e-6f)
			// To avoid numerical instabilities
			return 0.0f;

		incident_light_direction /= distance_to_sample_point;
	}

	if (hippt::dot(incident_light_direction, surface.shading_normal) <= 0.001f && sample.incident_light_info_at_visible_point != BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE)
		return 0.0f;
	else if constexpr (resamplingNeighbor)
	{
		if (render_data.render_settings.DEBUG_DONT_REUSE_SPECULAR)
			if (sample.incident_light_info_at_sample_point == BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE)
				return 0.0f;

		// If resampling a neighbor, the target function is going to evaluate to 0.0f if the sample point of the neighbor
		// is specular: that is because when resampling a neighbor, i.e. reconnecting to the sample point of the neighbor,
		// we're changing the view direction of the BSDF at the sample point.
		//
		// And changing the view direction of a specular BSDF without changing the incident light direction (which we are not
		// modifying) isn't going to adhere to the law of perfect reflection and so the contribution of the BSDF at the neighbor's
		// sample point will be 0.0f.
		//
		// So that's why we're returning 0.0f here
		if (render_data.render_settings.restir_gi_settings.use_neighbor_sample_point_roughness_heuristic &&
			!MaterialUtils::can_do_light_sampling(sample.sample_point_material, render_data.render_settings.restir_gi_settings.neighbor_sample_point_roughness_threshold))
			return 0.0f;
	}

	if constexpr (withVisiblity)
	{
		hiprtRay visibility_ray;
		visibility_ray.origin = surface.shading_point;
		visibility_ray.direction = incident_light_direction;

		Xorshift32Generator random_number_generator_alpha_test(sample.visible_to_sample_point_alpha_test_random_seed);
		bool sample_point_occluded = evaluate_shadow_ray(render_data, visibility_ray, distance_to_sample_point, surface.last_hit_primitive_index, 0, random_number_generator_alpha_test);
		if (sample_point_occluded)
			return 0.0f;
	}

	float bsdf_pdf;
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, surface.material, surface.ray_volume_state, false, surface.view_direction, surface.shading_normal, surface.geometric_normal, incident_light_direction, bsdf_pdf, random_number_generator, 0, sample.incident_light_info_at_visible_point);
	if (bsdf_pdf > 0.0f)
		bsdf_color *= hippt::abs(hippt::dot(surface.shading_normal, incident_light_direction));

#if ReSTIRGIDoubleBSDFInTargetFunction == KERNEL_OPTION_TRUE
	if (!sample.is_envmap_path())
	{
		float sample_point_bsdf_pdf;
		ColorRGB32F sample_point_bsdf_color = bsdf_dispatcher_eval(render_data, sample.sample_point_material.unpack(), const_cast<RayVolumeState&>(sample.sample_point_volume_state), false, -incident_light_direction, sample.sample_point_shading_normal, sample.sample_point_geometric_normal, sample.incident_light_direction_at_sample_point, sample_point_bsdf_pdf, random_number_generator, 1, sample.incident_light_info_at_sample_point);
		ColorRGB32F incoming_radiance_to_visible_point_reconstructed;

		if (sample_point_bsdf_pdf > 0.0f)
			incoming_radiance_to_visible_point_reconstructed = sample_point_bsdf_color / sample_point_bsdf_pdf * hippt::abs(hippt::dot(sample.sample_point_shading_normal, sample.incident_light_direction_at_sample_point)) * sample.incoming_radiance_to_sample_point;

		return (bsdf_color * incoming_radiance_to_visible_point_reconstructed).luminance();
	}
	else
		return (bsdf_color * sample.incoming_radiance_to_visible_point).luminance();
#else
	return (bsdf_color * sample.incoming_radiance_to_visible_point).luminance();
#endif
}

#endif
