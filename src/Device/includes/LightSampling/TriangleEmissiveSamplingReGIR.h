/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_EMISSIVE_SAMPLING_REGIR_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_EMISSIVE_SAMPLING_REGIR_H

#include "Device/includes/ReSTIR/ReGIR/ShadingLightDistributions.h"
#include "Device/includes/ReSTIR/ReGIR/ShadingPairwiseMIS.h"

HIPRT_DEVICE LightSamplePointInformation sample_one_emissive_triangle_regir_with_selected_sample_radiance(const HIPRTRenderData& render_data,
																										  const float3_t& shading_point,
																										  const float3_t& view_direction,
																										  const float3_t& shading_normal,
																										  const float3_t& geometric_normal,
																										  int last_hit_primitive_index,
																										  RayPayload& ray_payload,
																										  bool& out_need_fallback_sampling,
																										  ColorRGB32F& out_selected_sample_radiance,
																										  Xorshift32Generator& random_number_generator)
{
	const ReGIRSettings& regir_settings = render_data.render_settings.regir_settings;

#if ReGIR_ShadingResamplingSampleOnlyLightDistributions == KERNEL_OPTION_TRUE && ReGIR_GridFillUsePerCellLightDistributions == KERNEL_OPTION_TRUE
	unsigned int canonical_grid_cell_index = regir_settings.find_valid_jittered_neighbor_cell_index<true>(
		shading_point, geometric_normal, render_data.current_camera, ray_payload.material.roughness, regir_settings.compute_is_primary_hit(ray_payload),
		ReGIR_ShadingResamplingJitterCanonicalCandidates &&
			regir_settings.shading_settings.get_do_cell_jittering(regir_settings.compute_is_primary_hit(ray_payload)),
		regir_settings.shading_settings.jittering_radius_canonical_candidates, random_number_generator);

	if (canonical_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		out_need_fallback_sampling = true;

		return LightSamplePointInformation();
	}
	else
		out_need_fallback_sampling = false;

	ReGIRReservoir reservoir = ReGIR_shading_sample_light_distributions(
		render_data, view_direction, shading_point, shading_normal, geometric_normal, ray_payload, last_hit_primitive_index, canonical_grid_cell_index,
		regir_settings.compute_is_primary_hit(ray_payload), out_selected_sample_radiance, random_number_generator);
	// No normalization because we're already using proper MIS weights
	reservoir.finalize_resampling(1.0f, 1.0f);

	if (reservoir.sample.emissive_triangle_global_index == -1)
		return LightSamplePointInformation();

	float3_t normal = triangle_load_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_global_index);
	float area		= hippt::length(normal) * 0.5f;
	normal			= hippt::normalize(normal);

	LightSamplePointInformation out_sample2;
	out_sample2.area_measure_pdf			   = 1.0f / reservoir.UCW;
	out_sample2.emission					   = triangle_load_emission(render_data, reservoir.sample.emissive_triangle_global_index);
	out_sample2.emissive_triangle_global_index = reservoir.sample.emissive_triangle_global_index;
	out_sample2.light_source_normal			   = triangle_load_normal_not_normalized(render_data, reservoir.sample.emissive_triangle_global_index);
	out_sample2.light_area					   = hippt::length(out_sample2.light_source_normal) * 0.5f;
	out_sample2.light_source_normal /= out_sample2.light_area * 2.0f;
	out_sample2.point_on_light = reservoir.sample.point_on_light;

	return out_sample2;
#else // #if ReGIR_ShadingResamplingSampleOnlyLightDistributions == KERNEL_OPTION_TRUE && ReGIR_GridFillUsePerCellLightDistributions == KERNEL_OPTION_TRUE
	// Starting with this at true and if we find a single good neighbor,
	// this will be set to false
	out_need_fallback_sampling = true;

	float3_t selected_point_on_light				   = make_float3(0.0f, 0.0f, 0.0f);
	float3_t selected_light_source_normal			   = make_float3(0.0f, 0.0f, 0.0f);
	float selected_light_source_area				   = 0.0f;
	BSDFIncidentLightInfo selected_incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	ColorRGB32F selected_emission;

	ReGIRReservoir out_reservoir;

	// Some random seed to generate to positions of the neighbors (when jittering)
	// XORing here because not XORing was causing RNG correlations issues...
	// not sure how that works but more randomness here seems to be getting rid of those correlations issues
	unsigned neighbor_rng_seed			= random_number_generator.xorshift32() ^ random_number_generator.xorshift32();
	unsigned non_cano_neighbor_rng_seed = neighbor_rng_seed ^ random_number_generator.xorshift32();
	Xorshift32Generator non_canonical_neighbor_rng(non_cano_neighbor_rng_seed);
	Xorshift32Generator neighbor_rng(neighbor_rng_seed);

	unsigned int valid_non_canonical_neighbors = 0;
	for (int neighbor = 0; neighbor < regir_settings.shading_settings.number_of_neighbors; neighbor++)
	{
		unsigned int neighbor_grid_cell_index = regir_settings.find_valid_jittered_neighbor_cell_index<false>(
			shading_point, geometric_normal, render_data.current_camera, ray_payload.material.roughness, regir_settings.compute_is_primary_hit(ray_payload),
			regir_settings.shading_settings.get_do_cell_jittering(regir_settings.compute_is_primary_hit(ray_payload)),
			regir_settings.shading_settings.jittering_radius, non_canonical_neighbor_rng);
		if (neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			// Not a valid neighbor
			continue;
		else
			valid_non_canonical_neighbors++;
	}
	// Resetting the seed after the counting of the neighbors
	non_canonical_neighbor_rng.m_state.seed = non_cano_neighbor_rng_seed;

	ReGIRPairwiseMIS pairwise;

	unsigned int canonical_grid_cell_index = regir_settings.find_valid_jittered_neighbor_cell_index<true>(
		shading_point, geometric_normal, render_data.current_camera, ray_payload.material.roughness, regir_settings.compute_is_primary_hit(ray_payload),
		ReGIR_ShadingResamplingJitterCanonicalCandidates &&
			regir_settings.shading_settings.get_do_cell_jittering(regir_settings.compute_is_primary_hit(ray_payload)),
		regir_settings.shading_settings.jittering_radius_canonical_candidates, neighbor_rng);

	float UCW_1 = 0.0f, UCW_2 = 0.0f;
	int triangle_index_canonical_technique_1 = -1, triangle_index_canonical_technique_2 = -1, triangle_index_canonical_technique_3 = -1;
	float3_t point_on_light_1 = make_float3(0.0f, 0.0f, 0.0f), point_on_light_2 = make_float3(0.0f, 0.0f, 0.0f),
			 point_on_light_3	   = make_float3(0.0f, 0.0f, 0.0f);
	float3_t light_source_normal_1 = make_float3(0.0f, 0.0f, 0.0f), light_source_normal_2 = make_float3(0.0f, 0.0f, 0.0f),
			 light_source_normal_3 = make_float3(0.0f, 0.0f, 0.0f);
	ColorRGB32F emission_1, emission_2, emission_3;

	BSDFIncidentLightInfo canonical_technique_3_sample_ili = BSDFIncidentLightInfo::NO_INFO;

	ReGIRGridFillSurface center_cell_surface;

	float canonical_technique_1_canonical_reservoir_1_pdf = 0.0f;
	float canonical_technique_1_canonical_reservoir_2_pdf = 0.0f;
	float canonical_technique_1_canonical_reservoir_3_pdf = 0.0f;
	float canonical_technique_2_canonical_reservoir_1_pdf = 0.0f;
	float canonical_technique_2_canonical_reservoir_2_pdf = 0.0f;
	float canonical_technique_2_canonical_reservoir_3_pdf = 0.0f;
	float canonical_technique_3_canonical_reservoir_1_pdf = 0.0f;
	float canonical_technique_3_canonical_reservoir_2_pdf = 0.0f;
	float canonical_technique_3_canonical_reservoir_3_pdf = 0.0f;
	float mis_weight_normalization						  = pairwise.compute_MIS_weight_normalization(render_data, valid_non_canonical_neighbors);

	float non_canonical_RIS_integral_center_grid_cell;
	float canonical_RIS_integral_center_grid_cell;

	// Fetching the center cell should never fail because the center cell always exists but it may actually fail in case of collisions
	// that cannot be resolved
	if (canonical_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		// We found at least one good sample so we're not going to need a fallback on another light sampling strategy than ReGIR
		out_need_fallback_sampling = false;

		// Producing the canonical techniques samples
		{
			ReGIRReservoir canonical_technique_1_reservoir = regir_settings.get_random_reservoir_in_grid_cell_for_shading<false>(
				canonical_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
			ReGIRReservoir canonical_technique_2_reservoir;

#if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE && ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_FALSE
			canonical_technique_2_reservoir = regir_settings.get_random_reservoir_in_grid_cell_for_shading<true>(
				canonical_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
#endif // #if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE && ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_FALSE

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
			float bsdf_sample_pdf;
			float3_t sampled_bsdf_direction;

			BSDFContext bsdf_context(view_direction, shading_normal, geometric_normal, make_float3(0.0f, 0.0f, 0.0f), canonical_technique_3_sample_ili,
									 ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
									 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
			bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

			bool intersection_found = false;
			BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
			if (bsdf_sample_pdf > 0.0f)
			{
				hiprtRay new_ray;
				new_ray.origin	  = shading_point;
				new_ray.direction = sampled_bsdf_direction;

#if ReGIR_ShadingResamplingDoBSDFMISSimplifiedRay == KERNEL_OPTION_TRUE
				intersection_found = evaluate_bsdf_light_sample_ray_simplified(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info,
																			   last_hit_primitive_index, random_number_generator);
#else // #if ReGIR_ShadingResamplingDoBSDFMISSimplifiedRay == KERNEL_OPTION_TRUE
				intersection_found = evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, last_hit_primitive_index,
																	ray_payload.bounce, random_number_generator);
#endif // #if ReGIR_ShadingResamplingDoBSDFMISSimplifiedRay == KERNEL_OPTION_TRUE

				// Checking that we did hit something and if we hit something,
				// it needs to be emissive
				if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black())
				{
					triangle_index_canonical_technique_3 = shadow_light_ray_hit_info.hit_prim_index;
					point_on_light_3					 = shading_point + shadow_light_ray_hit_info.hit_distance * sampled_bsdf_direction;
					light_source_normal_3				 = shadow_light_ray_hit_info.hit_geometric_normal;
					emission_3							 = shadow_light_ray_hit_info.hit_emission;

					// We want ReGIR to produce PDFs that are in area measure so we're converting from solid angle to area measure here
					canonical_technique_3_canonical_reservoir_3_pdf =
						solid_angle_to_area_pdf(bsdf_sample_pdf, shadow_light_ray_hit_info.hit_distance,
												compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction));
				}
			}
#endif // #if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE

			// Extracting the data of the canonical reservoirs 1 and 2
			if (canonical_technique_1_reservoir.UCW > 0.0f)
			{
				UCW_1								 = canonical_technique_1_reservoir.UCW;
				triangle_index_canonical_technique_1 = canonical_technique_1_reservoir.sample.emissive_triangle_global_index;

				point_on_light_1 = canonical_technique_1_reservoir.sample.point_on_light;
				light_source_normal_1 =
					hippt::normalize(triangle_load_normal_not_normalized(render_data, canonical_technique_1_reservoir.sample.emissive_triangle_global_index));
				emission_1 = triangle_load_emission(render_data, canonical_technique_1_reservoir.sample.emissive_triangle_global_index);
			}

#if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
			LightSamplePointInformation canonical_sample = sample_one_emissive_triangle_light_tree_ats(
				render_data, shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload, random_number_generator);
			if (canonical_sample.emissive_triangle_global_index != -1)
			{
				UCW_2								 = 1.0f / canonical_sample.area_measure_pdf;
				triangle_index_canonical_technique_2 = canonical_sample.emissive_triangle_global_index;

				light_source_normal_2 = canonical_sample.light_source_normal;
				point_on_light_2	  = canonical_sample.point_on_light;
				emission_2			  = canonical_sample.emission;
			}
#else // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
			if (canonical_technique_2_reservoir.UCW > 0.0f)
			{
				UCW_2								 = canonical_technique_2_reservoir.UCW;
				triangle_index_canonical_technique_2 = canonical_technique_2_reservoir.sample.emissive_triangle_global_index;

				point_on_light_2 = canonical_technique_2_reservoir.sample.point_on_light;
				light_source_normal_2 =
					hippt::normalize(triangle_load_normal_not_normalized(render_data, canonical_technique_2_reservoir.sample.emissive_triangle_global_index));
				emission_2 = triangle_load_emission(render_data, canonical_technique_2_reservoir.sample.emissive_triangle_global_index);
			}
#endif // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
		}

		// Computing all the PDFs of the canonical techniques that we're going to need for pairwise MIS
		{
			if (!emission_1.is_black())
			{
				// TODO we already have the canonical / non-canonical PDF normalization (fetched below) so we can use them because otherwise, that function
				// fetches them again
				canonical_technique_1_canonical_reservoir_1_pdf =
					ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, point_on_light_1, light_source_normal_1, emission_1, canonical_grid_cell_index,
																regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
#if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE
#if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
				canonical_technique_2_canonical_reservoir_1_pdf =
					pdf_of_emissive_triangle_light_tree_ats(render_data, shading_point, shading_normal, triangle_index_canonical_technique_1) /
					triangle_load_area(render_data, triangle_index_canonical_technique_1);
#else // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
				canonical_technique_2_canonical_reservoir_1_pdf =
					ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, point_on_light_1, light_source_normal_1, emission_1, canonical_grid_cell_index,
															   regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
#endif // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
#endif // #if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
				canonical_technique_3_canonical_reservoir_1_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(
					render_data, point_on_light_1, light_source_normal_1, emission_1, view_direction, shading_point, shading_normal, geometric_normal,
					BSDFIncidentLightInfo::NO_INFO, ray_payload, last_hit_primitive_index);
#endif // #if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
			}

			if (!emission_2.is_black())
			{
				canonical_technique_1_canonical_reservoir_2_pdf =
					ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, point_on_light_2, light_source_normal_2, emission_2, canonical_grid_cell_index,
																regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
#if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE
#if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
				canonical_technique_2_canonical_reservoir_2_pdf =
					pdf_of_emissive_triangle_light_tree_ats(render_data, shading_point, shading_normal, triangle_index_canonical_technique_2) /
					triangle_load_area(render_data, triangle_index_canonical_technique_2);
#else // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
				canonical_technique_2_canonical_reservoir_2_pdf =
					ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, point_on_light_2, light_source_normal_2, emission_2, canonical_grid_cell_index,
															   regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
#endif // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
#endif // #if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
				canonical_technique_3_canonical_reservoir_2_pdf = ReGIR_get_reservoir_sample_BSDF_PDF(
					render_data, point_on_light_2, light_source_normal_2, emission_2, view_direction, shading_point, shading_normal, geometric_normal,
					BSDFIncidentLightInfo::NO_INFO, ray_payload, last_hit_primitive_index);
#endif // #if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
			}

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
			if (!emission_3.is_black())
			{
				canonical_technique_1_canonical_reservoir_3_pdf =
					ReGIR_get_reservoir_sample_ReGIR_PDF<false>(render_data, point_on_light_3, light_source_normal_3, emission_3, canonical_grid_cell_index,
																regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
#if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE
#if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
				canonical_technique_2_canonical_reservoir_3_pdf =
					pdf_of_emissive_triangle_light_tree_ats(render_data, shading_point, shading_normal, triangle_index_canonical_technique_3) /
					triangle_load_area(render_data, triangle_index_canonical_technique_3);
#else // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
				canonical_technique_2_canonical_reservoir_3_pdf =
					ReGIR_get_reservoir_sample_ReGIR_PDF<true>(render_data, point_on_light_3, light_source_normal_3, emission_3, canonical_grid_cell_index,
															   regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);
#endif // #if ReGIR_ShadingResamplingCanonicalCandidatesLightTreeATS == KERNEL_OPTION_TRUE
#endif // #if ReGIR_ShadingResamplingIncludeCanonicalCandidates == KERNEL_OPTION_TRUE
				// This one has already been computed when sampling the BSDF sample
				// canonical_technique_3_canonical_reservoir_3_pdf....
			}
#endif // #if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
		}

		{
			non_canonical_RIS_integral_center_grid_cell =
				regir_settings.get_non_canonical_pre_integration_factor(canonical_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload));
			if (non_canonical_RIS_integral_center_grid_cell == 0.0f)
				non_canonical_RIS_integral_center_grid_cell = 1.0f;

			canonical_RIS_integral_center_grid_cell =
				regir_settings.get_canonical_pre_integration_factor(canonical_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload));
			if (canonical_RIS_integral_center_grid_cell == 0.0f)
				canonical_RIS_integral_center_grid_cell = 1.0f;
		}

		center_cell_surface = ReGIR_get_cell_surface(render_data, canonical_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload));
	}
	else
	{
		// The center grid cell is invalid (must be because of hash grid collisions that couldn't be resolved)
		out_need_fallback_sampling = true;

		return LightSamplePointInformation();
	}

	for (int neighbor = 0; neighbor < regir_settings.shading_settings.number_of_neighbors; neighbor++)
	{
		unsigned int neighbor_grid_cell_index = regir_settings.find_valid_jittered_neighbor_cell_index<false>(
			shading_point, geometric_normal, render_data.current_camera, ray_payload.material.roughness, regir_settings.compute_is_primary_hit(ray_payload),
			regir_settings.shading_settings.get_do_cell_jittering(regir_settings.compute_is_primary_hit(ray_payload)),
			regir_settings.shading_settings.jittering_radius, non_canonical_neighbor_rng);
		if (neighbor_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			// Couldn't find a valid neighbor
			continue;
		else
			out_need_fallback_sampling = false;

		ReGIRGridFillSurface neighbor_surface =
			ReGIR_get_cell_surface(render_data, neighbor_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload));

		float neighbor_RIS_integral =
			regir_settings.get_non_canonical_pre_integration_factor(neighbor_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload));
		if (neighbor_RIS_integral == 0.0f)
			neighbor_RIS_integral = 1.0f;

		// Will be set to true if the jittering causes the current shading point to be jittered out of the scene
		ReGIRReservoir non_canonical_reservoir = regir_settings.get_random_reservoir_in_grid_cell_for_shading<false>(
			neighbor_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload), neighbor_rng);

		if (non_canonical_reservoir.UCW <= 0.0f || non_canonical_reservoir.sample.emissive_triangle_global_index == -1)
		{
			// No valid sample in that reservoir

			pairwise.sum_non_canonical_sample_to_canonical_weights(
				render_data, point_on_light_1, light_source_normal_1, emission_1, point_on_light_2, light_source_normal_2, emission_2, point_on_light_3,
				light_source_normal_3, emission_3,

				canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_1_canonical_reservoir_2_pdf,
				canonical_technique_1_canonical_reservoir_3_pdf, canonical_technique_2_canonical_reservoir_1_pdf,
				canonical_technique_2_canonical_reservoir_2_pdf, canonical_technique_2_canonical_reservoir_3_pdf,
				canonical_technique_3_canonical_reservoir_1_pdf, canonical_technique_3_canonical_reservoir_2_pdf,
				canonical_technique_3_canonical_reservoir_3_pdf, mis_weight_normalization,

				neighbor_surface, neighbor_RIS_integral, regir_settings.compute_is_primary_hit(ray_payload), random_number_generator);

			continue;
		}

		float3_t light_source_normal = triangle_load_normal_not_normalized(render_data, non_canonical_reservoir.sample.emissive_triangle_global_index);
		float light_source_area		 = hippt::length(light_source_normal) * 0.5f;
		light_source_normal /= hippt::length(light_source_normal);
		float3_t point_on_light = non_canonical_reservoir.sample.point_on_light;
		ColorRGB32F emission	= triangle_load_emission(render_data, non_canonical_reservoir.sample.emissive_triangle_global_index);

		ColorRGB32F sample_radiance;
		float shading_target_function =
			ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(
				render_data, shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload, point_on_light,
				light_source_normal, emission, random_number_generator, sample_radiance);

		float non_canonical_sample_PDF_unnormalized =
			ReGIR_grid_fill_evaluate_non_canonical_target_function(render_data, neighbor_surface, regir_settings.compute_is_primary_hit(ray_payload), emission,
																   light_source_normal, point_on_light, random_number_generator);
		float current_sample_PDF = non_canonical_sample_PDF_unnormalized / neighbor_RIS_integral;
		float mis_weight		 = pairwise.compute_MIS_weight_for_non_canonical_sample(
			render_data, point_on_light, light_source_normal, emission, non_canonical_reservoir.sample.emissive_triangle_global_index, shading_target_function,

			point_on_light_1, light_source_normal_1, emission_1, point_on_light_2, light_source_normal_2, emission_2, point_on_light_3, light_source_normal_3,
			emission_3,

			center_cell_surface, regir_settings.compute_is_primary_hit(ray_payload),

			canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_1_canonical_reservoir_2_pdf, canonical_technique_1_canonical_reservoir_3_pdf,
			canonical_technique_2_canonical_reservoir_1_pdf, canonical_technique_2_canonical_reservoir_2_pdf, canonical_technique_2_canonical_reservoir_3_pdf,
			canonical_technique_3_canonical_reservoir_1_pdf, canonical_technique_3_canonical_reservoir_2_pdf, canonical_technique_3_canonical_reservoir_3_pdf,

			mis_weight_normalization,

			non_canonical_RIS_integral_center_grid_cell, canonical_RIS_integral_center_grid_cell, current_sample_PDF, neighbor_surface, neighbor_RIS_integral,

			view_direction, shading_point, shading_normal, geometric_normal, ray_payload, last_hit_primitive_index, random_number_generator);

		if (out_reservoir.stream_reservoir(mis_weight, shading_target_function, non_canonical_reservoir, random_number_generator))
		{
			selected_point_on_light		 = point_on_light;
			selected_light_source_normal = light_source_normal;
			selected_light_source_area	 = light_source_area;
			selected_emission			 = emission;

			out_selected_sample_radiance = sample_radiance;
		}
	}

	/**
	 * CANONICAL TECHNIQUE: NON-CANONICAL CANDIDATE
	 * For good variance reduction in the donominator of pairwise MIS
	 */
	if (canonical_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		if (!emission_1.is_black())
		{
			// Adding visibility in the canonical sample target function's if we have visibility in the grid fill target function
			// or if we wwant visibility in the target function during shading resampling
			// or if we're shading all candidates because then we want the target function to produce
			// the radiance towards the shading point directly which means that we need the visibility in the target function
			ColorRGB32F sample_radiance;
			float target_function =
					  ReGIR_shading_evaluate_target_function < ReGIR_GridFillTargetFunctionVisibility || ReGIR_ShadingResamplingTargetFunctionVisibility,
				  ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility > (render_data, shading_point, view_direction, shading_normal, geometric_normal,
																				last_hit_primitive_index, ray_payload, point_on_light_1, light_source_normal_1,
																				emission_1, random_number_generator, sample_radiance);

			float RIS_integral =
				regir_settings.get_non_canonical_pre_integration_factor(canonical_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload));
			if (RIS_integral == 0.0f)
				RIS_integral = 1.0f;
			float non_canonical_sample_PDF_unnormalized = ReGIR_grid_fill_evaluate_non_canonical_target_function(
				render_data, canonical_grid_cell_index, regir_settings.compute_is_primary_hit(ray_payload), emission_1, light_source_normal_1, point_on_light_1,
				random_number_generator);
			float non_canonical_sample_PDF = non_canonical_sample_PDF_unnormalized / RIS_integral;

			float mis_weight =
				pairwise.get_canonical_MIS_weight_1(canonical_technique_1_canonical_reservoir_1_pdf, canonical_technique_2_canonical_reservoir_1_pdf,
													canonical_technique_3_canonical_reservoir_1_pdf, mis_weight_normalization);

			ReGIRReservoir canonical_technique_1_reservoir;
			canonical_technique_1_reservoir.sample.emissive_triangle_global_index = triangle_index_canonical_technique_1;
			canonical_technique_1_reservoir.UCW									  = UCW_1;
			if (out_reservoir.stream_reservoir(mis_weight, target_function, canonical_technique_1_reservoir, random_number_generator))
			{
				selected_point_on_light		 = point_on_light_1;
				selected_light_source_normal = light_source_normal_1;
				selected_light_source_area	 = hippt::length(triangle_load_normal_not_normalized(render_data, triangle_index_canonical_technique_1)) * 0.5f;
				selected_emission			 = emission_1;

				out_selected_sample_radiance = sample_radiance;
			}
		}
	}

	/**
	 * TRUE CANONICAL CANDIDATE (with the very simple target function)
	 */
	// Incorporating a canonical candidate if doing visibility reuse because visibility reuse
	// may cause the grid cell to produce no valid reservoir at all so we need canonical samples to
	// cover those cases for unbiased results
	//
	// Fetching the center cell should never fail because the center cell always exists but it may actually fail in case of collisions
	// that cannot be resolved
	if (canonical_grid_cell_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		if (!emission_2.is_black())
		{
			// Adding visibility in the canonical sample target function's if we have visibility in the grid fill target function
			// or if we wwant visibility in the target function during shading resampling
			// or if we're shading all candidates because then we want the target function to produce
			// the radiance towards the shading point directly which means that we need the visibility in the target function
			ColorRGB32F sample_radiance;
			float target_function =
					  ReGIR_shading_evaluate_target_function < ReGIR_GridFillTargetFunctionVisibility || ReGIR_ShadingResamplingTargetFunctionVisibility,
				  ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility > (render_data, shading_point, view_direction, shading_normal, geometric_normal,
																				last_hit_primitive_index, ray_payload, point_on_light_2, light_source_normal_2,
																				emission_2, random_number_generator, sample_radiance);

			float mis_weight =
				pairwise.get_canonical_MIS_weight_2(canonical_technique_1_canonical_reservoir_2_pdf, canonical_technique_2_canonical_reservoir_2_pdf,
													canonical_technique_3_canonical_reservoir_2_pdf, mis_weight_normalization);

			ReGIRReservoir canonical_technique_2_reservoir;
			canonical_technique_2_reservoir.sample.emissive_triangle_global_index = triangle_index_canonical_technique_2;
			canonical_technique_2_reservoir.UCW									  = UCW_2;
			if (out_reservoir.stream_reservoir(mis_weight, target_function, canonical_technique_2_reservoir, random_number_generator))
			{
				selected_point_on_light		 = point_on_light_2;
				selected_light_source_normal = light_source_normal_2;
				selected_light_source_area	 = hippt::length(triangle_load_normal_not_normalized(render_data, triangle_index_canonical_technique_2)) * 0.5f;
				selected_emission			 = emission_2;

				out_selected_sample_radiance = sample_radiance;
			}
		}
	}

#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE
	if (canonical_technique_3_canonical_reservoir_3_pdf > 0.0f)
	{
		float mis_weight = pairwise.get_canonical_MIS_weight_3(canonical_technique_1_canonical_reservoir_3_pdf, canonical_technique_2_canonical_reservoir_3_pdf,
															   canonical_technique_3_canonical_reservoir_3_pdf, mis_weight_normalization);

		ColorRGB32F sample_radiance;
		float target_function =
			ReGIR_shading_evaluate_target_function<ReGIR_ShadingResamplingTargetFunctionVisibility, ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility>(
				render_data, shading_point, view_direction, shading_normal, geometric_normal, last_hit_primitive_index, ray_payload, point_on_light_3,
				light_source_normal_3, emission_3, random_number_generator, sample_radiance, canonical_technique_3_sample_ili);

		// Giving 0u as the random seed here because we don't have it for BSDF samples.
		//
		// Not an issue here because out_reservoir.sample.random_seed is never used anyways
		if (out_reservoir.stream_sample_raw(mis_weight, target_function, canonical_technique_3_canonical_reservoir_3_pdf, triangle_index_canonical_technique_3,
											point_on_light_3, random_number_generator))
		{
			selected_point_on_light		 = point_on_light_3;
			selected_light_source_normal = light_source_normal_3;
			selected_light_source_area	 = hippt::length(triangle_load_normal_not_normalized(render_data, triangle_index_canonical_technique_3)) * 0.5f;
			selected_emission			 = emission_3;
			selected_incident_light_info = canonical_technique_3_sample_ili;

			out_selected_sample_radiance = sample_radiance;
		}
	}
#endif // #if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE

	if (out_reservoir.weight_sum == 0.0f || out_need_fallback_sampling)
		return LightSamplePointInformation();

	out_reservoir.finalize_resampling(1.0f, 1.0f);

	LightSamplePointInformation out_sample;

	// The UCW is the inverse of the PDF but we expect the PDF to be in 'area_measure_pdf', not the inverse PDF (UCW), so we invert it
	out_sample.area_measure_pdf				  = 1.0f / out_reservoir.UCW;
	out_sample.emissive_triangle_global_index = out_reservoir.sample.emissive_triangle_global_index;
	out_sample.emission						  = selected_emission;
	out_sample.light_area					  = selected_light_source_area;
	out_sample.light_source_normal			  = selected_light_source_normal;
	out_sample.point_on_light				  = selected_point_on_light;
#if DirectLightSamplingStrategy == LSS_BASE_REGIR
	// Compile guard because 'out_sample.incident_light_info' is only defined if ReGIR is enabled
	out_sample.incident_light_info = selected_incident_light_info;
#endif // #if DirectLightSamplingStrategy == LSS_BASE_REGIR

	return out_sample;
#endif // #if ReGIR_ShadingResamplingSampleOnlyLightDistributions == KERNEL_OPTION_TRUE && ReGIR_GridFillUsePerCellLightDistributions == KERNEL_OPTION_TRUE
}

HIPRT_DEVICE LightSamplePointInformation sample_one_point_on_light_regir(const HIPRTRenderData& render_data,
																		 const float3_t& shading_point,
																		 const float3_t& view_direction,
																		 const float3_t& shading_normal,
																		 const float3_t& geometric_normal,
																		 int last_hit_primitive_index,
																		 RayPayload& ray_payload,
																		 bool& out_need_fallback_sampling,
																		 Xorshift32Generator& random_number_generator)
{
	ColorRGB32F trash_selected_sample_color;
	return sample_one_emissive_triangle_regir_with_selected_sample_radiance(render_data, shading_point, view_direction, shading_normal, geometric_normal,
																			last_hit_primitive_index, ray_payload, out_need_fallback_sampling,
																			trash_selected_sample_color, random_number_generator);
}

#endif // #ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_EMISSIVE_SAMPLING_REGIR_H
