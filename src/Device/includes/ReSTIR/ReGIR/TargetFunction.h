/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_TARGET_FUNCTION_H
#define DEVICE_INCLUDES_REGIR_TARGET_FUNCTION_H

#include "Device/includes/BSDFs/BSDFContext.h"
#include "Device/includes/Dispatcher.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE float ReGIR_grid_fill_evaluate_target_function(HIPRTRenderData& render_data, float3 cell_center, const LightSampleInformation& regir_sample)
{
	return regir_sample.emission.luminance() / hippt::length2(cell_center - regir_sample.point_on_light);
}

HIPRT_HOST_DEVICE float ReGIR_shading_evaluate_target_function(const HIPRTRenderData& render_data,
	const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
	RayPayload& ray_payload,
	const ReGIRReservoir& reservoir,
	const ColorRGB32F& light_emission,
	Xorshift32Generator& rng)
{

	float3 to_light_direction = reservoir.sample.point_on_light - shading_point;
	float distance_to_light = hippt::length(to_light_direction);
	to_light_direction /= distance_to_light; // Normalization

	float bsdf_pdf;
	BSDFIncidentLightInfo ili = BSDFIncidentLightInfo::NO_INFO;
	BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, to_light_direction, ili, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, rng);

	float cosine_term = hippt::abs(hippt::dot(shading_normal, to_light_direction));
	float geometry_term = hippt::abs(hippt::dot(reservoir.sample.light_source_normal, to_light_direction)) / hippt::square(distance_to_light);

	return (bsdf_color * light_emission * cosine_term * geometry_term).luminance();
}

#endif