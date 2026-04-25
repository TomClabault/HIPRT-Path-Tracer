/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PG_SAMPLPE_DISTRIBUTION_H
#define DEVICE_INCLUDES_RESTIR_PG_SAMPLPE_DISTRIBUTION_H

#include "Device/includes/PathGuiding/PathSampling.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE void restir_pg_sample_bounce(HIPRTRenderData& render_data,
										  RayPayload& ray_payload,
										  const HitInfo& closest_hit_info,
										  const float3_t view_direction,
										  ColorRGB32F& out_bsdf_color,
										  float3_t& out_bounce_direction,
										  float& out_sample_pdf,
										  Xorshift32Generator& random_number_generator,
										  BSDFIncidentLightInfo out_sampled_light_info)
{
	ReSTIRPGDistribution distribution = render_data.render_settings.restir_pg_settings.get_distribution_from_position_data(
		closest_hit_info.inter_point, closest_hit_info.geometric_normal, render_data.current_camera);

	// For one sample MIS between BSDF and the ReSTIR PG distribution
	float bsdf_probability = render_data.render_settings.restir_pg_settings.bsdf_sampling_probability;
	if (!distribution.is_valid())
		// No distribution, full BSDF sampling then
		bsdf_probability = 1.0f;

	bool path_guiding_sampled;
	float random_value = random_number_generator();
	if (random_value < bsdf_probability)
	{
		float trash_pdf;
		path_tracing_sample_bsdf_next_indirect_bounce<true>(render_data, ray_payload, closest_hit_info, view_direction, out_bsdf_color, out_bounce_direction,
															trash_pdf, random_number_generator, out_sampled_light_info);

		path_guiding_sampled = false;
	}
	else
	{
		out_bounce_direction = distribution.sample(random_number_generator);
		path_guiding_update_volume_stack_after_sampling(out_bounce_direction, closest_hit_info, ray_payload);

		out_sampled_light_info = BSDFIncidentLightInfo::NO_INFO;

		path_guiding_sampled = true;
	}

	// TODO THIS IS BROKEN AND THE RAY VOLUME STACK WILL NOT BE PROPERLY UPDATED IN CASE THE PATH GUIDING SAMPLES A REFRACTION. If a glass reflection is
	// sampled, we need to pop the stack, just as the principled_glass_sample function would do. Use this opportunity to maybe remove the pop/push stack logic
	// from Principled.h and do it externally or something
	// This sampled light info is unused because we're evaluating the BSDF
	BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, out_bounce_direction, out_sampled_light_info,
							 ray_payload.volume_state, true, ray_payload.material, ray_payload.accumulated_roughness);

	float bsdf_pdf;
	out_bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
	bsdf_pdf *= bsdf_probability;

	float distribution_pdf = distribution.pdf(out_bounce_direction);
	distribution_pdf *= (1.0f - bsdf_probability);

	float chosen_pdf			= random_value < bsdf_probability ? bsdf_pdf : distribution_pdf;
	float other_pdf				= random_value < bsdf_probability ? distribution_pdf : bsdf_pdf;
	float one_sample_mis_weight = power_heuristic(chosen_pdf, other_pdf);

	out_sample_pdf = chosen_pdf / one_sample_mis_weight;

	if (path_guiding_sampled)
		path_guiding_compute_sampled_lobe(render_data, bsdf_context, ray_payload, out_bounce_direction, out_sampled_light_info, random_number_generator);
}

#endif
