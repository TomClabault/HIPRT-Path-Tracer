/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_TARGET_FUNCTION_H
#define DEVICE_RESTIR_GI_TARGET_FUNCTION_H

#include "Device/includes/LightSampling/Lights.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/Surface.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "HostDeviceCommon/RenderData.h"

template <bool withVisiblity, bool resamplingNeighbor = true>
HIPRT_HOST_DEVICE float ReSTIR_GI_evaluate_target_function(const HIPRTRenderData& render_data, const ReSTIRGISample& sample, ReSTIRSurface& surface, Xorshift32Generator& random_number_generator)
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

	float cosine_term = hippt::dot(incident_light_direction, surface.shading_normal);
	if (cosine_term <= 0.0f && sample.incident_light_info_at_visible_point != BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE)
		return 0.0f;
	else if constexpr (resamplingNeighbor)
	{
		// If resampling a neighbor, the target function is going to evaluate to 0.0f if the sample point of the neighbor
		// is specular: that is because when resampling a neighbor, i.e. reconnecting to the sample point of the neighbor,
		// we're changing the view direction of the BSDF at the sample point.
		//
		// And changing the view direction of a specular BSDF without changing the incident light direction (which we are not
		// modifying) isn't going to adhere to the law of perfect reflection and so the contribution of the BSDF at the neighbor's
		// sample point will be 0.0f.
		//
		// So that's why we're returning 0.0f here
		if (render_data.render_settings.restir_gi_settings.use_neighbor_sample_point_roughness_heuristic && !sample.sample_point_rough_enough)
			return 0.0f;
	}

	if constexpr (withVisiblity)
	{
		hiprtRay visibility_ray;
		visibility_ray.origin = surface.shading_point;
		visibility_ray.direction = incident_light_direction;

		Xorshift32Generator random_number_generator_alpha_test(sample.visible_to_sample_point_alpha_test_random_seed);
		bool sample_point_occluded = evaluate_shadow_ray_occluded(render_data, visibility_ray, distance_to_sample_point, surface.primitive_index, 0, random_number_generator_alpha_test);
		if (sample_point_occluded)
			return 0.0f;
	}

	float bsdf_pdf;
	BSDFContext bsdf_context(surface.view_direction, surface.shading_normal, surface.geometric_normal, incident_light_direction, const_cast<BSDFIncidentLightInfo&>(sample.incident_light_info_at_visible_point), surface.ray_volume_state, false, surface.material, 0, 0.0f, MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION);
	ColorRGB32F visible_point_bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
	if (bsdf_pdf > 0.0f)
		visible_point_bsdf_color *= hippt::abs(cosine_term);

	return (visible_point_bsdf_color * sample.incoming_radiance_to_visible_point.unpack()).luminance();
}

#endif
