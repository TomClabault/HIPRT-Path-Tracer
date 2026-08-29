/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_TARGET_FUNCTION_H
#define DEVICE_RESTIR_PT_TARGET_FUNCTION_H

#include "Device/includes/LightSampling/NEEEstimators.h"
#include "Device/includes/Material.h"
#include "Device/includes/ReSTIR/Jacobian.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/Utils.h"
#include "Device/includes/ReSTIR/Surface.h"
#include "HostDeviceCommon/RenderData.h"

template <bool withVisiblity>
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

	if (!sample.is_envmap_path() && sample.di_sample &&
		compute_cosine_term_at_light_source(sample.rc_vertex_geometric_normal.unpack(), -incident_light_direction) <= 0.0f)
		// Backfacing light
		return 0.0f;

	float cosine_term = hippt::dot(incident_light_direction, surface.shading_normal);
	if (cosine_term <= 0.0f && !bsdf_incident_light_info_transmission_lobe(sample.incident_light_info_at_visible_point))
		return 0.0f;

	if constexpr (withVisiblity)
	{
		hiprtRay visibility_ray;
		visibility_ray.origin	 = surface.shading_point;
		visibility_ray.direction = incident_light_direction;

		bool sample_point_occluded =
			evaluate_shadow_ray_occluded(render_data, visibility_ray, distance_to_sample_point, surface.primitive_index, random_number_generator);
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

		RayVolumeState ray_volume_state_copy = surface.ray_volume_state;
		// TODO reproduce roughness accumumlation
		// ray_payload.accumulate_roughness(resampling_reservoir.sample.incident_light_info_at_visible_point);
		ReSTIR_PT_update_volume_state_for_sample_point(render_data, ray_volume_state_copy, surface.material, sample.incident_light_info_at_visible_point,
													   surface.primitive_index);

		int rc_vertex_material_index = render_data.buffers.material_indices[sample.rc_vertex_primitive_index];
		DeviceUnpackedEffectiveMaterial rc_vertex_material =
			get_intersection_material(render_data, rc_vertex_material_index, make_float2(sample.rc_vertex_texcoords_u, sample.rc_vertex_texcoords_v));
		BSDFContext secondary_hit_eval_context(view_direction, shading_normal_sample_point, geometric_normal_sample_point, to_light_direction_sample_point,
											   const_cast<BSDFIncidentLightInfo&>(sample.incident_light_info_at_sample_point), ray_volume_state_copy, false,
											   rc_vertex_material, 0.0f);

		// TODO can we use a simple target function visible point only for perf? We can have a template parameter to do that only during spatial reuse
		float trash_pdf;
		ColorRGB32F sample_point_bsdf_color = bsdf_dispatcher_eval(render_data, secondary_hit_eval_context, trash_pdf, random_number_generator);
		sample_point_throughput				= sample_point_bsdf_color * hippt::abs(hippt::dot(to_light_direction_sample_point, shading_normal_sample_point));
	}

	return (visible_point_throughput * sample_point_throughput * sample.rc_vertex_incident_radiance).luminance();
}

#endif // #ifndef DEVICE_RESTIR_PT_TARGET_FUNCTION_H
