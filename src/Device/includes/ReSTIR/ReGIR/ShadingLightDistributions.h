/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_SHADING_H
#define DEVICE_KERNELS_REGIR_SHADING_H

#include "Device/includes/LightSampling/PDFTriangles.h"
#include "Device/includes/ReSTIR/ReGIR/LightDistributionsGridFill.h"

template <int samplingStrategy>
HIPRT_DEVICE LightSamplePointArray<DirectLightSampleCount<samplingStrategy>()> sample_one_point_on_light(const HIPRTRenderData& render_data,
																										 const float3_t& shading_point,
																										 const float3_t& view_direction,
																										 const float3_t& shading_normal,
																										 const float3_t& geometric_normal,
																										 int last_hit_primitive_index,
																										 RayPayload& ray_payload,
																										 Xorshift32Generator& random_number_generator);

HIPRT_DEVICE static ReGIRReservoir ReGIR_shading_sample_light_distributions(const HIPRTRenderData& render_data,
																			float3_t view_direction,
																			float3_t shading_point,
																			float3_t shading_normal,
																			float3_t geometric_normal,
																			RayPayload& ray_payload,
																			int last_hit_primitive_index,
																			unsigned int hash_grid_cell_index,
																			bool primary_hit,

																			ColorRGB32F& selected_sample_radiance,

																			Xorshift32Generator& rng)
{
	ReGIRReservoir reservoir;

	const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

	// Sampling some samples with per-cell light distributions
	for (int light_sample_index = 0; light_sample_index < regir_settings.shading_settings.number_of_neighbors; light_sample_index++)
	{
		LightSamplePointInformation light_point_sample;

		light_point_sample = sample_one_emissive_triangle_with_cell_light_distribution(render_data, shading_point, view_direction, shading_normal,
																					   ray_payload.material, hash_grid_cell_index, primary_hit, rng);
		// TODO do we need this?
		// if (light_point_sample.emissive_triangle_global_index == REGIR_NEEDS_LIGHT_SAMPLE_FALLBACK)
		//    // Falling back on the base strategy
		//    // TODO MULTI LIGHT SAMPLE
		//    light_point_sample = LightSamplePointInformation();
		//    //light_point_sample = sample_one_point_on_light<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(render_data, shading_point,
		//    view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload, rng);

		if (light_point_sample.emissive_triangle_global_index == -1)
			continue;

		ColorRGB32F sample_radiance;
		float target_function =
			ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(
				render_data, shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
				light_point_sample.point_on_light, light_point_sample.light_source_normal, light_point_sample.emission, rng, sample_radiance);

		if (target_function == 0.0f)
			continue;

		float bsdf_pdf_area_measure = 0.0f;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
		BSDFIncidentLightInfo incident_light_no_info = BSDFIncidentLightInfo::NO_INFO;
		BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, hippt::normalize(light_point_sample.point_on_light - shading_point),
								 incident_light_no_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
								 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
		// BSDF PDF here is approximate because it should contain visibility but the increase in variance is fine
		bsdf_pdf_area_measure = solid_angle_to_area_pdf(
			bsdf_dispatcher_pdf(render_data, bsdf_context), hippt::length(light_point_sample.point_on_light - shading_point),
			compute_cosine_term_at_light_source(light_point_sample.light_source_normal, hippt::normalize(shading_point - light_point_sample.point_on_light)));
#endif
		float canonical_strategy_PDF = pdf_of_emissive_triangle_hit_area_measure<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(
			render_data, shading_point, view_direction, shading_normal, ray_payload.material, light_point_sample.point_on_light,
			light_point_sample.light_source_normal, light_point_sample.emissive_triangle_global_index);
		float mis_weight = balance_heuristic(light_point_sample.area_measure_pdf, regir_settings.shading_settings.number_of_neighbors, canonical_strategy_PDF,
											 ReGIR_GridFillCellDistributionsCanonicalSampleCount * DirectLightIntegrationFactor<DirectLightSamplingStrategy>(),
											 bsdf_pdf_area_measure, ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE);

		if (reservoir.stream_sample(mis_weight, target_function, light_point_sample.area_measure_pdf, light_point_sample, rng))
			selected_sample_radiance = sample_radiance;
		sanity_check<true>(render_data, reservoir.weight_sum, -1, -1);
	}

	// Sampling some samples with a canonical 'cover-all-triangles" strategy (power sampling for example)
	// for unbiasedness
	for (int light_sample_index = 0; light_sample_index < ReGIR_GridFillCellDistributionsCanonicalSampleCount; light_sample_index++)
	{
		ReGIRGridFillSurface surface;
		surface.cell_roughness		 = ray_payload.material.roughness;
		surface.cell_metallic		 = ray_payload.material.metallic;
		surface.cell_specular		 = ray_payload.material.specular;
		surface.cell_normal			 = shading_normal;
		surface.cell_point			 = shading_point;
		surface.cell_primitive_index = last_hit_primitive_index;

		LightSamplePointArray<DirectLightSampleCount<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>()> light_samples =
			grid_fill_cell_light_distributions_canonical_sample(render_data, surface, view_direction, rng);
		for (int i = 0; i < DirectLightSampleCount<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(); i++)
		{
			LightSamplePointInformation& light_point_sample = light_samples[i];

			if (light_point_sample.emissive_triangle_global_index == -1)
				// Can happen if the triangle sampled is degenerate and thus rejected
				continue;

			ColorRGB32F sample_radiance;
			float target_function = ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility,
																		   ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(
				render_data, shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
				light_point_sample.point_on_light, light_point_sample.light_source_normal, light_point_sample.emission, rng, sample_radiance);

			if (target_function == 0.0f)
				continue;

			float bsdf_pdf_area_measure = 0.0f;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
			BSDFIncidentLightInfo incident_light_no_info = BSDFIncidentLightInfo::NO_INFO;
			BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, hippt::normalize(light_point_sample.point_on_light - shading_point),
									 incident_light_no_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
									 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
			// BSDF PDF here is approximate because it should contain visibility but the increase in variance is fine
			bsdf_pdf_area_measure =
				solid_angle_to_area_pdf(bsdf_dispatcher_pdf(render_data, bsdf_context), hippt::length(light_point_sample.point_on_light - shading_point),
										compute_cosine_term_at_light_source(light_point_sample.light_source_normal,
																			hippt::normalize(shading_point - light_point_sample.point_on_light)));
#endif
			unsigned int sampled_mesh_index =
				render_data.buffers.emissive_meshes_data.global_triangle_index_to_emissive_mesh_index[light_point_sample.emissive_triangle_global_index];
			float cell_light_distributions_pdf =
				get_cell_distribution_PDF_of_light_sample(render_data, hash_grid_cell_index, primary_hit, light_point_sample, sampled_mesh_index);
			// 3-way balance heuristic for simplicity (pairwise MIS would probably be more performant)
			float mis_weight = balance_heuristic(
				light_point_sample.area_measure_pdf,
				ReGIR_GridFillCellDistributionsCanonicalSampleCount * DirectLightIntegrationFactor<DirectLightSamplingStrategy>(), cell_light_distributions_pdf,
				regir_settings.shading_settings.number_of_neighbors, bsdf_pdf_area_measure, ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE);

			if (reservoir.stream_sample(mis_weight, target_function, light_point_sample.area_measure_pdf, light_point_sample, rng))
				selected_sample_radiance = sample_radiance;
			sanity_check<true>(render_data, reservoir.weight_sum, -1, -1);
		}
	}

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
	float bsdf_sample_pdf;
	float3_t sampled_bsdf_direction;

	BSDFIncidentLightInfo incident_light_info;
	BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state,
							 false, ray_payload.material, ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
	ColorRGB32F bsdf_contribution = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, rng);

	bool intersection_found = false;
	BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
	if (bsdf_sample_pdf > 0.0f)
	{
		hiprtRay bsdf_ray;
		bsdf_ray.origin	   = shading_point;
		bsdf_ray.direction = sampled_bsdf_direction;

#if ReGIR_ShadingResamplingDoBSDFMISSimplifiedRay == KERNEL_OPTION_TRUE
		intersection_found =
			evaluate_bsdf_light_sample_ray_simplified(render_data, bsdf_ray, 1.0e35f, shadow_light_ray_hit_info, last_hit_primitive_index, rng);
#else
		intersection_found =
			evaluate_bsdf_light_sample_ray(render_data, bsdf_ray, 1.0e35f, shadow_light_ray_hit_info, last_hit_primitive_index, ray_payload.bounce, rng);
#endif

		float bsdf_sample_pdf_area_measure =
			solid_angle_to_area_pdf(bsdf_sample_pdf, shadow_light_ray_hit_info.hit_distance,
									compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -bsdf_ray.direction));

		if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black() && bsdf_sample_pdf_area_measure > 0.0f)
		{
			ColorRGB32F sample_radiance;
			float target_function = ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility,
																		   ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(
				render_data, shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload,
				shading_point + bsdf_ray.direction * shadow_light_ray_hit_info.hit_distance, shadow_light_ray_hit_info.hit_geometric_normal,
				shadow_light_ray_hit_info.hit_emission, rng, sample_radiance);

			float mis_weight = 0.0f;
			int mesh_index	 = render_data.buffers.emissive_meshes_data.global_triangle_index_to_emissive_mesh_index[shadow_light_ray_hit_info.hit_prim_index];
			if (mesh_index != -1)
			{
				// The mesh index of the emissive triangle hit can be -1 if the emissive mesh is only
				// emissive thanks to using an emissive texture

				float PDF_light_distributions = get_cell_distribution_PDF_of_light_sample(
					render_data, hash_grid_cell_index, primary_hit,
					hippt::length(triangle_load_normal_not_normalized(render_data, shadow_light_ray_hit_info.hit_prim_index)) * 0.5f,
					shadow_light_ray_hit_info.hit_emission, mesh_index);
				float canonical_technique_pdf = pdf_of_emissive_triangle_hit_area_measure<ReGIR_GridFillCellDistributionsCanonicalSamplingTechnique>(
					render_data, shading_point, view_direction, shading_normal, ray_payload.material,
					shading_point + bsdf_ray.direction * shadow_light_ray_hit_info.hit_distance, shadow_light_ray_hit_info);
				mis_weight = balance_heuristic(
					bsdf_sample_pdf_area_measure, 1, PDF_light_distributions, regir_settings.shading_settings.number_of_neighbors, canonical_technique_pdf,
					ReGIR_GridFillCellDistributionsCanonicalSampleCount * DirectLightIntegrationFactor<DirectLightSamplingStrategy>());
			}
			else
				// If we couldn't find the emissive mesh index of the emissive triangle that we just hit,
				// this has to be because this is a mesh that is using an emissive texture
				//
				// Emissive texture sampling isn't available at the time of writing this code so light
				// sampling can't sample so the BSDF sampling gets all the weight
				mis_weight = 1.0f;

			if (reservoir.stream_sample_raw(mis_weight, target_function, bsdf_sample_pdf_area_measure, shadow_light_ray_hit_info.hit_prim_index,
											shading_point + bsdf_ray.direction * shadow_light_ray_hit_info.hit_distance, rng))
				selected_sample_radiance = sample_radiance;
			sanity_check<true>(render_data, reservoir.weight_sum, -1, -1);
		}
	}
#endif

	return reservoir;
}

#endif
