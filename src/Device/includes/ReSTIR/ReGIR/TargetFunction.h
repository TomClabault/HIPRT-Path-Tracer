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

template <bool includeVisibility, bool withCosineTerm, bool withCosineTermLightSource, bool includeBSDF, bool withNeePlusPlusVisibilityEstimation>
HIPRT_DEVICE float ReGIR_grid_fill_evaluate_target_function(const HIPRTRenderData& render_data, unsigned int hash_grid_cell_index, 
	ColorRGB32F sample_emission, float3 sample_normal, float3 sample_position, Xorshift32Generator& rng)
{
    int cell_primitive_index = ReGIR_get_cell_primitive_index(render_data, hash_grid_cell_index);
    float3 cell_point = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index);
	float3 cell_normal = ReGIR_get_cell_world_shading_normal(render_data, hash_grid_cell_index);
	float cell_roughness = ReGIR_get_cell_roughness(render_data, hash_grid_cell_index);
	float cell_metallic = ReGIR_get_cell_metallic(render_data, hash_grid_cell_index);
	float cell_specular = ReGIR_get_cell_specular(render_data, hash_grid_cell_index);

	float3 to_light_direction = sample_position - cell_point;
	float distance_to_light = hippt::length(to_light_direction);
	to_light_direction /= distance_to_light;

	float target_function = sample_emission.luminance() / hippt::square(distance_to_light);
	if (cell_primitive_index != -1 && withCosineTerm)
		// We do have a representative normal, taking the cosine term into account
		target_function *= hippt::max(0.0f, hippt::dot(cell_normal, to_light_direction));

	if constexpr (withCosineTermLightSource)
		target_function *= compute_cosine_term_at_light_source(sample_normal, -to_light_direction);

	if (target_function <= 0.0f)
		return 0.0f;

	if constexpr (includeBSDF)
	{
		float out_pdf;
		RayVolumeState empty_volume_state;
		BSDFIncidentLightInfo out_incident_light_info;
		DeviceUnpackedEffectiveMaterial approximate_material;
		approximate_material.roughness = cell_roughness;
		approximate_material.metallic = cell_metallic;
		approximate_material.specular = cell_specular;

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
		BSDFContext bsdf_context = BSDFContext(hippt::normalize(render_data.current_camera.position - cell_point), cell_normal, cell_normal, to_light_direction, out_incident_light_info, empty_volume_state, false, approximate_material, 0, 0, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
#else
		BSDFContext bsdf_context = BSDFContext(hippt::normalize(render_data.current_camera.position - cell_point), cell_normal, cell_normal, to_light_direction, out_incident_light_info, empty_volume_state, false, approximate_material, 0, 0, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
#endif
		ColorRGB32F bsdf_radiance = bsdf_dispatcher_eval(render_data, bsdf_context, out_pdf, rng);
		target_function *= bsdf_radiance.luminance();
	}

	if constexpr (includeVisibility)
	{
		if (target_function > 0.0f)
			// No need to visibility test if the target function is already 0
			target_function *= ReGIR_grid_cell_visibility_test(render_data, cell_point, cell_primitive_index, sample_position, rng);
	}
	else if constexpr (withNeePlusPlusVisibilityEstimation && DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE)
	{
		NEEPlusPlusContext context;
		context.envmap = false;
		context.point_on_light = sample_position;
		context.shaded_point = cell_point;

		target_function *= render_data.nee_plus_plus.estimate_visibility_probability(context, render_data.current_camera);
	}

	return target_function;
}

HIPRT_DEVICE float ReGIR_grid_fill_evaluate_non_canonical_target_function(const HIPRTRenderData& render_data, unsigned int hash_grid_cell_index,
	ColorRGB32F sample_emission, float3 sample_normal, float3 sample_position, Xorshift32Generator& rng)
{
	return ReGIR_grid_fill_evaluate_target_function<
		ReGIR_GridFillTargetFunctionVisibility, ReGIR_GridFillTargetFunctionCosineTerm, ReGIR_GridFillTargetFunctionCosineTermLightSource, ReGIR_GridFillTargetFunctionBSDF, ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation>(
		render_data, hash_grid_cell_index, sample_emission, sample_normal, sample_position, rng);
}

HIPRT_DEVICE float ReGIR_grid_fill_evaluate_canonical_target_function(const HIPRTRenderData& render_data, unsigned int hash_grid_cell_index,
	ColorRGB32F sample_emission, float3 sample_normal, float3 sample_position, Xorshift32Generator& rng)
{
	return ReGIR_grid_fill_evaluate_target_function<false, false, false, false, false>(
		render_data, hash_grid_cell_index, sample_emission, sample_normal, sample_position, rng);
}

template <bool withVisibility, bool withNeePlusPlusVisibilityEstimation, bool withGeometryTerm = true>
HIPRT_DEVICE float ReGIR_shading_evaluate_target_function(const HIPRTRenderData& render_data,
	const float3& shading_point, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal,
	int last_hit_primitive_index, RayPayload& ray_payload,
	const float3& point_on_light, const float3& light_source_normal,
	const ColorRGB32F& light_emission,
	Xorshift32Generator& rng,
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO)
{
	float3 to_light_direction = point_on_light - shading_point;
	float distance_to_light = hippt::length(to_light_direction);
	to_light_direction /= distance_to_light; // Normalization

#if ReGIR_ShadingResamplingIncludeBSDF == KERNEL_OPTION_TRUE
	float bsdf_pdf;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
	BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, to_light_direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
#else
	BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, to_light_direction, incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.bounce, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
#endif
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, rng);
#else
	ColorRGB32F bsdf_color = ColorRGB32F(1.0f);
#endif

	float cosine_term = hippt::max(0.0f, hippt::dot(shading_normal, to_light_direction));
	float geometry_term = compute_cosine_term_at_light_source(light_source_normal, -to_light_direction) / hippt::square(distance_to_light);
	if constexpr (!withGeometryTerm)
		geometry_term = 1.0f;

	float target_function = (bsdf_color * light_emission * cosine_term * geometry_term).luminance();
	if (target_function <= 0.0f)
		return 0.0f;

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
	else if constexpr (withNeePlusPlusVisibilityEstimation && DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE)
	{
		NEEPlusPlusContext context;
		context.envmap = false;
		context.point_on_light = point_on_light;
		context.shaded_point = shading_point;

		float visibility_proba = render_data.nee_plus_plus.estimate_visibility_probability(context, render_data.current_camera);
		if (visibility_proba > 0.005f)
			visibility_proba = 1.0f;
		visibility_proba = hippt::max(0.1f, visibility_proba);
			
		target_function *= visibility_proba;
	}
	
	return target_function;
}

HIPRT_DEVICE bool ReGIR_shading_can_sample_be_produced_by_internal(const HIPRTRenderData& render_data, 
	ColorRGB32F sample_emission, float3 sample_normal, float3 point_on_light,
	int hash_grid_cell_index, Xorshift32Generator& rng)
{
	return ReGIR_grid_fill_evaluate_target_function<
		ReGIR_DoVisibilityReuse || ReGIR_GridFillTargetFunctionVisibility, 
		ReGIR_GridFillTargetFunctionCosineTerm, 
		ReGIR_GridFillTargetFunctionCosineTermLightSource,
		ReGIR_GridFillTargetFunctionBSDF,
		ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation>(
		render_data, hash_grid_cell_index, 
		sample_emission, sample_normal, point_on_light, 
		rng) > 0.0f;
}

HIPRT_DEVICE bool ReGIR_shading_can_sample_be_produced_by(const HIPRTRenderData& render_data, const LightSampleInformation& light_sample, unsigned int hash_grid_cell_index,
	Xorshift32Generator& rng)
{
	return ReGIR_shading_can_sample_be_produced_by_internal(render_data, 
		light_sample.emission, light_sample.light_source_normal, light_sample.point_on_light, 
		hash_grid_cell_index, rng);
}

#endif