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
#include "Device/includes/ReSTIR/PT/SpatialMISTargetFunctionBSDFApproximation.h"
#include "Device/includes/ReSTIR/PT/Utils.h"
#include "Device/includes/ReSTIR/Surface.h"
#include "HostDeviceCommon/RenderData.h"

// Forced inline: out of line, the sample, the material and the volume state passed by reference live in scratch memory in the caller
HIPRT_HOST_DEVICE HIPRT_FORCEINLINE float ReSTIR_PT_evaluate_target_function_bsdf(const HIPRTRenderData& render_data,
																				  const ReSTIRPTReservoirSample& sample,
																				  const float3_t& surface_shading_point,
																				  const float3_t& surface_view_direction,
																				  const float3_t& surface_shading_normal,
																				  const float3_t& surface_geometric_normal,
																				  int surface_primitive_index,
																				  RayVolumeState& surface_ray_volume_state,
																				  DeviceUnpackedPrincipledFullMaterial& surface_material,
																				  const float3_t& incident_light_direction,
																				  float cosine_term,
																				  Xorshift32Generator& random_number_generator)
{
	float bsdf_pdf;
	BSDFContext bsdf_context(surface_view_direction, surface_shading_normal, surface_geometric_normal, incident_light_direction,
							 const_cast<BSDFIncidentLightInfo&>(sample.incident_light_info_at_visible_point), surface_ray_volume_state, false, surface_material,
							 0.0f, MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION);
	ColorRGB32F visible_point_throughput = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator) * hippt::abs(cosine_term);

	ColorRGB32F sample_point_throughput = ColorRGB32F(1.0f);

	if (!sample.di_sample)
	{
		float3_t view_direction					 = hippt::normalize(surface_shading_point - sample.rc_vertex);
		float3_t to_light_direction_sample_point = sample.rc_vertex_incident_light_direction;
		float3_t shading_normal_sample_point	 = sample.rc_vertex_shading_normal.unpack();
		float3_t geometric_normal_sample_point	 = sample.rc_vertex_geometric_normal.unpack();

		int rc_vertex_material_index = render_data.buffers.material_indices[sample.rc_vertex_primitive_index];

		RayVolumeState ray_volume_state_copy = surface_ray_volume_state;
		// TODO reproduce roughness accumumlation
		// ray_payload.accumulate_roughness(resampling_reservoir.sample.incident_light_info_at_visible_point);
		ReSTIR_PT_update_volume_state_for_sample_point(render_data, ray_volume_state_copy, surface_material, sample.incident_light_info_at_visible_point,
													   rc_vertex_material_index);

		DeviceUnpackedPrincipledFullMaterial rc_vertex_material =
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

/**
 * 'onlyVisibility' (only with 'withVisiblity'): returns 1.0f if the sample gets past the zero tests and the visibility test below, 0.0f otherwise, without
 * evaluating the BSDFs. See ReSTIR_PT_sample_visible_at_surface()
 */
template <bool withVisiblity, bool onlyVisibility = false>
HIPRT_HOST_DEVICE float ReSTIR_PT_evaluate_target_function(const HIPRTRenderData& render_data,
														   const ReSTIRPTReservoirSample& sample,
														   ReSTIRPTSurface& surface,
														   Xorshift32Generator& random_number_generator)
{
	static_assert(withVisiblity || !onlyVisibility, "The visibility-only evaluation of the target function needs the visibility test");

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

		Xorshift32Generator random_number_generator_alpha_test(sample.visible_to_sample_point_alpha_test_random_seed);

		bool sample_point_occluded =
			evaluate_shadow_ray_occluded(render_data, visibility_ray, distance_to_sample_point, surface.primitive_index, random_number_generator_alpha_test);
		if (sample_point_occluded)
			return 0.0f;
	}

	if constexpr (onlyVisibility)
		return 1.0f;

	return ReSTIR_PT_evaluate_target_function_bsdf(render_data, sample, surface.shading_point, surface.view_direction, surface.shading_normal,
												   surface.geometric_normal, surface.primitive_index, surface.ray_volume_state, surface.material,
												   incident_light_direction, cosine_term, random_number_generator);
}

/**
 * Returns true exactly when ReSTIR_PT_evaluate_target_function<true>() gets past its zero tests (degenerate distance, backfacing light of a DI sample,
 * cosine <= 0 outside of a transmission lobe) and its visibility test for this sample at this surface: same tests and same shadow ray (alpha test seeded
 * by the sample, deterministic) but no BSDF evaluation and no material fetch
 */
HIPRT_HOST_DEVICE bool ReSTIR_PT_sample_visible_at_surface(const HIPRTRenderData& render_data,
														   const ReSTIRPTReservoirSample& sample,
														   ReSTIRPTSurface& surface,
														   Xorshift32Generator& random_number_generator)
{
	return ReSTIR_PT_evaluate_target_function<true, true>(render_data, sample, surface, random_number_generator) > 0.0f;
}

/**
 * 'misTargetFunctionBSDFApproximation': returns the MIS target function BSDF approximation of the union domain reuse instead of the target function
 * (SpatialMISTargetFunctionBSDFApproximation.h). The zero tests below are the same, the approximation is computed before the visibility test (a zero
 * approximation skips the ray): analytic value on the cheap path, full target function evaluation otherwise (see
 * ReSTIR_PT_MIS_target_function_bsdf_approximation_decision())
 *
 * This is a template parameter so that only one of the two evaluations is compiled at a call site: the approximation (with its full target function
 * fallback before the visibility test) or the full target function after the visibility test
 *
 * Forced inline: out of line, the sample and the shading point passed by reference live in scratch memory in the caller and every field read here
 * is a scratch load
 */
template <bool withVisiblity, bool misTargetFunctionBSDFApproximation = false>
HIPRT_HOST_DEVICE HIPRT_FORCEINLINE float ReSTIR_PT_evaluate_target_function_at_pixel(const HIPRTRenderData& render_data,
																					  const ReSTIRPTReservoirSample& sample,
																					  int surface_pixel_index,
																					  const float3_t& surface_shading_point,
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
		incident_light_direction = sample.rc_vertex - surface_shading_point;

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

	// The padding byte of the packed shading normal holds the flags of the pixel for the MIS target function BSDF approximation
	Octahedral24BitNormalPadded32b surface_packed_shading_normal = render_data.g_buffer.shading_normals[surface_pixel_index];
	float3_t surface_shading_normal								 = surface_packed_shading_normal.unpack();

	float cosine_term = hippt::dot(incident_light_direction, surface_shading_normal);
	if (cosine_term <= 0.0f && !bsdf_incident_light_info_transmission_lobe(sample.incident_light_info_at_visible_point))
		return 0.0f;

	int surface_primitive_index = render_data.g_buffer.first_hit_prim_index[surface_pixel_index];

	float approximation_value = 0.0f;

	if constexpr (misTargetFunctionBSDFApproximation)
	{
		int decision =
			ReSTIR_PT_MIS_target_function_bsdf_approximation_decision(render_data, sample, surface_packed_shading_normal.get_padding(), surface_shading_point,
																	  surface_shading_normal, incident_light_direction, cosine_term);
		if (decision == RESTIR_PT_MIS_TARGET_FUNCTION_BSDF_APPROXIMATION_CHEAP_PATH)
			approximation_value = ReSTIR_PT_MIS_target_function_bsdf_approximation_cheap_path_value(
				render_data, sample, surface_pixel_index, surface_shading_point, surface_shading_normal, incident_light_direction, cosine_term);
		else if (decision == RESTIR_PT_MIS_TARGET_FUNCTION_BSDF_APPROXIMATION_FALLBACK)
		{
			// The full target function before the visibility test, as at the end of this function
			DeviceUnpackedPrincipledFullMaterial fallback_surface_material = render_data.g_buffer.materials[surface_pixel_index].unpack();
			RayVolumeState fallback_surface_ray_volume_state;
			fallback_surface_ray_volume_state.reconstruct_first_hit(fallback_surface_material, render_data.buffers.material_indices, surface_primitive_index,
																	random_number_generator);

			float3_t fallback_surface_view_direction   = render_data.g_buffer.get_view_direction(render_data.current_camera.position, surface_pixel_index);
			float3_t fallback_surface_geometric_normal = render_data.g_buffer.geometric_normals[surface_pixel_index].unpack();

			float full_target_function =
				ReSTIR_PT_evaluate_target_function_bsdf(render_data, sample, surface_shading_point, fallback_surface_view_direction, surface_shading_normal,
														fallback_surface_geometric_normal, surface_primitive_index, fallback_surface_ray_volume_state,
														fallback_surface_material, incident_light_direction, cosine_term, random_number_generator);

			approximation_value = ReSTIR_PT_MIS_target_function_bsdf_approximation_fallback_value(sample, full_target_function, cosine_term);
		}

		if (approximation_value <= 0.0f)
			// Zero whatever the visibility
			return 0.0f;
	}

	if constexpr (withVisiblity)
	{
		hiprtRay visibility_ray;
		visibility_ray.origin	 = surface_shading_point;
		visibility_ray.direction = incident_light_direction;

		Xorshift32Generator random_number_generator_alpha_test(sample.visible_to_sample_point_alpha_test_random_seed);

		bool sample_point_occluded =
			evaluate_shadow_ray_occluded(render_data, visibility_ray, distance_to_sample_point, surface_primitive_index, random_number_generator_alpha_test);
		if (sample_point_occluded)
			return 0.0f;
	}

	if constexpr (misTargetFunctionBSDFApproximation)
		return approximation_value;
	else
	{
		// Keep the large material and volume state out of the geometric rejection and visibility phases.
		DeviceUnpackedPrincipledFullMaterial surface_material = render_data.g_buffer.materials[surface_pixel_index].unpack();
		RayVolumeState surface_ray_volume_state;
		surface_ray_volume_state.reconstruct_first_hit(surface_material, render_data.buffers.material_indices, surface_primitive_index,
													   random_number_generator);

		float3_t surface_view_direction	  = render_data.g_buffer.get_view_direction(render_data.current_camera.position, surface_pixel_index);
		float3_t surface_geometric_normal = render_data.g_buffer.geometric_normals[surface_pixel_index].unpack();

		return ReSTIR_PT_evaluate_target_function_bsdf(render_data, sample, surface_shading_point, surface_view_direction, surface_shading_normal,
													   surface_geometric_normal, surface_primitive_index, surface_ray_volume_state, surface_material,
													   incident_light_direction, cosine_term, random_number_generator);
	}
}

#endif // #ifndef DEVICE_RESTIR_PT_TARGET_FUNCTION_H
