/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PT_INITIAL_CANDIDATES_UTILS_H
#define DEVICE_INCLUDES_RESTIR_PT_INITIAL_CANDIDATES_UTILS_H

#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/PG/SampleDistribution.h"

#define ReSTIR_PT_invalid_throughput ColorRGB32F(-1.0f, -1.0f, -1.0f)

HIPRT_HOST_DEVICE ColorRGB32F ReSTIR_PT_update_ray_throughputs(HIPRTRenderData& render_data,
															   RayPayload& ray_payload,
															   HitInfo& closest_hit_info,
															   ColorRGB32F bsdf_color,
															   const float3_t& bounce_direction,
															   float bsdf_pdf,
															   Xorshift32Generator& random_number_generator,
															   NEEDeferredMISContext& nee_deferred_MIS_context)
{
	ColorRGB32F bsdf_throughput		  = bsdf_color * hippt::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal));
	ColorRGB32F dispersion_throughput = get_dispersion_ray_color(ray_payload.volume_state.sampled_wavelength, ray_payload.material.dispersion_scale);
	ColorRGB32F unweighted_throughput = bsdf_throughput * dispersion_throughput;
	ColorRGB32F weighted_throughput	  = unweighted_throughput / bsdf_pdf;

	nee_deferred_MIS_context.fill_last_bsdf_information(unweighted_throughput, bsdf_pdf);

	// With ReSTIR GI, we want the outgoing radiance from the second hit to the camera hit
	// This means that we're basically not taking the first hit into account and so we're not
	// updating the throughput (or the ray_color either, see the main loop) on the bounce 0

	float rr_throughput_scaling = 1.0f;
	// Doing the russian roulette
	if (!do_russian_roulette(render_data.render_settings, ray_payload.bounce, ray_payload.throughput, rr_throughput_scaling, weighted_throughput,
							 random_number_generator))
	{
		// Killed by russian roulette
		ray_payload.throughput = ColorRGB32F(0.0f);

		return ReSTIR_PT_invalid_throughput;
	}
	else
		// Not killed by russian roulette so we're scaling the throughputs
		unweighted_throughput *= rr_throughput_scaling;

	// Clamp every component to a minimum of 1.0e-5f to avoid numerical instabilities that can
	// happen: with some material, the throughput can get so low that it becomes denormalized and
	// this can cause issues in some parts of the renderer (most notably the NaN detection)
	unweighted_throughput.max(ColorRGB32F(1.0e-5f, 1.0e-5f, 1.0e-5f));

	ray_payload.throughput *= dispersion_throughput;
	ray_payload.throughput *= weighted_throughput;
	// Clamp every component to a minimum of 1.0e-5f to avoid numerical instabilities that can
	// happen: with some material, the throughput can get so low that it becomes denormalized and
	// this can cause issues in some parts of the renderer (most notably the NaN detection)
	ray_payload.throughput.max(ColorRGB32F(1.0e-5f, 1.0e-5f, 1.0e-5f));

	return unweighted_throughput;
}

/**
 * Returns true if the bounce was sampled successfully,
 * false otherwise (is the BSDF sample failed, if russian roulette killed the sample, ...)
 */
HIPRT_HOST_DEVICE ColorRGB32F ReSTIR_PT_compute_next_indirect_bounce(HIPRTRenderData& render_data,
																	 RayPayload& ray_payload,
																	 HitInfo& closest_hit_info,
																	 float3_t view_direction,
																	 hiprtRay& out_ray,
																	 Xorshift32Generator& random_number_generator,
																	 BSDFIncidentLightInfo& incident_light_info,
																	 float& out_bsdf_pdf,
																	 NEEDeferredMISContext& nee_deferred_MIS_context)
{
	nee_deferred_MIS_context.fill_last_hit_information(closest_hit_info, view_direction, ray_payload.volume_state, ray_payload.material,
													   ray_payload.throughput);

	ColorRGB32F bsdf_color;
	float3_t bounce_direction;
	float bsdf_pdf;

#if ReSTIRPGEnable == KERNEL_OPTION_FALSE
	path_tracing_sample_bsdf_next_indirect_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf,
												  random_number_generator, incident_light_info);
#else
	restir_pg_sample_bounce(render_data, ray_payload, closest_hit_info, view_direction, bsdf_color, bounce_direction, bsdf_pdf, random_number_generator,
							incident_light_info);
#endif

	out_bsdf_pdf = bsdf_pdf;

	// Terminate ray if bad sampling
	if (bsdf_pdf <= 0.0f)
		return ReSTIR_PT_invalid_throughput;

	ColorRGB32F this_bounce_unweighted_throughput;
	if ((this_bounce_unweighted_throughput = ReSTIR_PT_update_ray_throughputs(render_data, ray_payload, closest_hit_info, bsdf_color, bounce_direction,
																			  bsdf_pdf, random_number_generator, nee_deferred_MIS_context)) ==
		ReSTIR_PT_invalid_throughput)
		return ReSTIR_PT_invalid_throughput;

	out_ray.origin	  = closest_hit_info.inter_point;
	out_ray.direction = bounce_direction;

	// Returning this bounce's unweighted throughput
	return this_bounce_unweighted_throughput;
}

#endif
