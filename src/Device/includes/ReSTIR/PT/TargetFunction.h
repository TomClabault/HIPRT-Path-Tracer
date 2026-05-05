/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_TARGET_FUNCTION_H
#define DEVICE_RESTIR_PT_TARGET_FUNCTION_H

#include "Device/includes/LightSampling/NEEEstimators.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/Surface.h"
#include "HostDeviceCommon/RenderData.h"

template <bool withVisiblity, bool resamplingNeighbor = true>
HIPRT_HOST_DEVICE float ReSTIR_PT_evaluate_target_function(const HIPRTRenderData& render_data,
														   const ReSTIRPTReservoirSample& sample,
														   ReSTIRSurface& surface,
														   Xorshift32Generator& random_number_generator)
{
	float distance_to_sample_point;
	float3_t incident_light_direction;
	if (sample.is_envmap_path())
	{
		// For envmap path, the direction is stored in the 'rc_vertex' value
		incident_light_direction = sample.rc_vertex;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point
		incident_light_direction = sample.rc_vertex - surface.shading_point;
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
		if (render_data.render_settings.restir_pt_settings.use_neighbor_sample_point_roughness_heuristic && !sample.sample_point_rough_enough)
			return 0.0f;
	}

	if constexpr (withVisiblity)
	{
		hiprtRay visibility_ray;
		visibility_ray.origin	 = surface.shading_point;
		visibility_ray.direction = incident_light_direction;

		Xorshift32Generator random_number_generator_alpha_test(sample.visible_to_sample_point_alpha_test_random_seed);
		bool sample_point_occluded =
			evaluate_shadow_ray_occluded(render_data, visibility_ray, distance_to_sample_point, surface.primitive_index, 0, random_number_generator_alpha_test);
		if (sample_point_occluded)
			return 0.0f;
	}

	float bsdf_pdf;
	BSDFContext bsdf_context(surface.view_direction, surface.shading_normal, surface.geometric_normal, incident_light_direction,
							 const_cast<BSDFIncidentLightInfo&>(sample.incident_light_info_at_visible_point), surface.ray_volume_state, false, surface.material,
							 0.0f, MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION);
	ColorRGB32F visible_point_throughput = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator) * hippt::abs(cosine_term);

	ColorRGB32F sample_point_throughput = ColorRGB32F(1.0f);

	if (!sample.di_sample)
	{
		float3_t view_direction					 = hippt::normalize(surface.shading_point - sample.rc_vertex);
		float3_t to_light_direction_sample_point = sample.rc_vertex_incident_light_direction;
		float3_t shading_normal_sample_point	 = sample.rc_vertex_shading_normal.unpack();
		float3_t geometric_normal_sample_point	 = sample.rc_vertex_geometric_normal.unpack();

		// TODO Reproducing roughness accumulation
		// ray_payload.accumulate_roughness(resampling_reservoir.sample.incident_light_info_at_visible_point);
		// TODO the ray volume state should be advanced/updated/pushed into here to reproduce the state that it's in at the sample point
		BSDFContext secondary_hit_eval_context(view_direction, shading_normal_sample_point, geometric_normal_sample_point, to_light_direction_sample_point,
											   const_cast<BSDFIncidentLightInfo&>(sample.incident_light_info_at_sample_point),
											   // TODO proper update volume state for the sample point
											   surface.ray_volume_state, false, const_cast<DeviceUnpackedEffectiveMaterial&>(sample.rc_vertex_material),
											   0.0f);

		// TODO can we use a simple target function visible point only for perf?
		float trash_pdf;
		ColorRGB32F sample_point_bsdf_color = bsdf_dispatcher_eval(render_data, secondary_hit_eval_context, trash_pdf, random_number_generator);
		sample_point_throughput				= sample_point_bsdf_color * hippt::abs(hippt::dot(to_light_direction_sample_point, shading_normal_sample_point));
	}

	// Note that this target function is not 100% accuracte, we would have to recompute the BSDF at the sample point with the new view direction to be fully
	// accurate but that would be more expensive so we're not doing that, not perfect but much cheaper
	return (visible_point_throughput * sample_point_throughput * sample.path_radiance).luminance();
}

#endif
