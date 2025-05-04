/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_REGIR_TARGET_FUNCTION_H
#define DEVICE_INCLUDES_REGIR_TARGET_FUNCTION_H

#include "Device/includes/BSDFs/BSDFContext.h"
#include "Device/includes/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/PDFConversion.h"
#include "Device/includes/ReSTIR/ReGIR/VisibilityTest.h"

#include "HostDeviceCommon/RenderData.h"

template <bool includeVisibility, bool withCosineTerm, bool withCosineTermLightSource>
HIPRT_DEVICE float ReGIR_non_shading_evaluate_target_function(const HIPRTRenderData& render_data, int hash_grid_cell_index, 
	ColorRGB32F sample_emission, float3 sample_normal, float3 sample_position, Xorshift32Generator& rng)
{
    int representative_primitive_index = ReGIR_get_cell_representative_primitive(render_data, hash_grid_cell_index);
    float3 representative_point = ReGIR_get_cell_representative_point(render_data, hash_grid_cell_index);
	float3 representative_normal = ReGIR_get_cell_representative_shading_normal(render_data, hash_grid_cell_index);

	float3 to_light_direction = sample_position - representative_point;
	float distance_to_light = hippt::length(to_light_direction);
	to_light_direction /= distance_to_light;

	float target_function = sample_emission.luminance() / hippt::square(distance_to_light);
	if (representative_primitive_index != -1 && withCosineTerm)
		// We do have a representative normal, taking the cosine term into account
		target_function *= hippt::max(0.0f, hippt::dot(representative_normal, to_light_direction));

	if constexpr (withCosineTermLightSource)
		target_function *= compute_cosine_term_at_light_source(sample_normal, -to_light_direction);

	if constexpr (includeVisibility)
	{
		if (target_function > 0.0f)
			// No need to visibility test if the target function is already 0
			target_function *= ReGIR_grid_cell_visibility_test(render_data, representative_point, representative_primitive_index, sample_position, rng);
	}

	return target_function;
}

template <bool withVisibility>
HIPRT_DEVICE float ReGIR_shading_evaluate_target_function(const HIPRTRenderData& render_data,
	const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
	int last_hit_primitive_index, RayPayload& ray_payload,
	const float3& point_on_light, const float3& light_source_normal,
	const ColorRGB32F& light_emission,
	Xorshift32Generator& rng)
{
	float3 to_light_direction = point_on_light - shading_point;
	float distance_to_light = hippt::length(to_light_direction);
	to_light_direction /= distance_to_light; // Normalization

#if ReGIR_ShadingResamplingIncludeBSDF == KERNEL_OPTION_TRUE
	float bsdf_pdf;
	BSDFIncidentLightInfo ili = BSDFIncidentLightInfo::NO_INFO;
	BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, to_light_direction, ili, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, rng);
#else
	ColorRGB32F bsdf_color = ColorRGB32F(1.0f);
#endif

	float cosine_term = hippt::max(0.0f, hippt::dot(shading_normal, to_light_direction));
	float geometry_term = compute_cosine_term_at_light_source(light_source_normal, -to_light_direction) / hippt::square(distance_to_light);

	float target_function = (bsdf_color * light_emission * cosine_term * geometry_term).luminance();

	if constexpr (withVisibility)
	{
		if (target_function > 0.0f)
		{
			hiprtRay shadow_ray;
			shadow_ray.origin = shading_point;
			shadow_ray.direction = to_light_direction;

			if (evaluate_shadow_ray(render_data, shadow_ray, distance_to_light, last_hit_primitive_index, ray_payload.bounce, rng))
				target_function = 0.0f;
		}
	}
	
	return target_function;
}

template <bool withVisibility>
HIPRT_DEVICE float ReGIR_shading_evaluate_target_function(const HIPRTRenderData& render_data,
	const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
	int last_hit_primitive_index, RayPayload& ray_payload,
	const ReGIRReservoir& reservoir,
	Xorshift32Generator& rng)
{
	return ReGIR_shading_evaluate_target_function<withVisibility>(render_data, 
		shading_point, view_direction, shading_normal, geometric_normal,
		last_hit_primitive_index, ray_payload,
		reservoir.sample.point_on_light, reservoir.sample.light_source_normal.unpack(),
		reservoir.sample.emission.unpack(), rng);
}

HIPRT_DEVICE bool ReGIR_shading_can_sample_be_produced_by_internal(const HIPRTRenderData& render_data, 
	ColorRGB32F sample_emission, float3 sample_normal, float3 point_on_light,
	int hash_grid_cell_index, Xorshift32Generator& rng)
{
	return ReGIR_non_shading_evaluate_target_function<ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility, ReGIR_GridFillTargetFunctionCosineTerm, ReGIR_GridFillTargetFunctionCosineTermLightSource>(
		render_data, hash_grid_cell_index, 
		sample_emission, sample_normal, point_on_light, 
		rng) > 0.0f;
}

HIPRT_DEVICE bool ReGIR_shading_can_sample_be_produced_by(const HIPRTRenderData& render_data, const LightSampleInformation& light_sample, int hash_grid_cell_index,
	Xorshift32Generator& rng)
{
	return ReGIR_shading_can_sample_be_produced_by_internal(render_data, 
		light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, 
		hash_grid_cell_index, rng);
}

HIPRT_DEVICE bool ReGIR_shading_can_sample_be_produced_by(const HIPRTRenderData& render_data, const ReGIRSample& light_sample, int hash_grid_cell_index,
	Xorshift32Generator& rng)
{
	return ReGIR_shading_can_sample_be_produced_by_internal(render_data, 
		light_sample.emission.unpack(), light_sample.light_source_normal.unpack(), light_sample.point_on_light,
		hash_grid_cell_index, rng);
}

#endif