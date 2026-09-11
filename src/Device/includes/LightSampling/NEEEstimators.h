/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_NEE_ESTIMATORS_H
#define DEVICE_NEE_ESTIMATORS_H

#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/HitInfo.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/LightSampling/NEEDeferredMISContext.h"
#include "Device/includes/LightSampling/NISML/NISML.h"
#include "Device/includes/LightSampling/RIS/RIS.h"
#include "Device/includes/LightSampling/RISLTC/RISLTC.h"
#include "Device/includes/LightSampling/TriangleEmissiveSampling.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/DI/FinalShading.h"
#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/ReSTIR/ReGIR/FinalShading.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE ColorRGB32F sample_one_light_no_MIS(HIPRTRenderData& render_data,
												 RayPayload& ray_payload,
												 const HitInfo closest_hit_info,
												 const float3_t& view_direction,
												 Xorshift32Generator& random_number_generator)
{
	if (!ray_payload.material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	ColorRGB32F light_source_radiance;

	LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
		sample_one_point_on_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
								  closest_hit_info.primitive_index, ray_payload, random_number_generator);

	for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
	{
		LightSamplePointInformation& light_sample = light_samples[i];

		if (light_sample.area_measure_pdf <= 0.0f)
			// Can happen for very small triangles or the light
			// sampling technique couldn't sample a triangle
			continue;

		float3_t shadow_ray_origin				 = closest_hit_info.inter_point;
		float3_t shadow_ray_direction			 = light_sample.point_on_light - shadow_ray_origin;
		float distance_to_light					 = hippt::length(shadow_ray_direction);
		float3_t shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

		hiprtRay shadow_ray;
		shadow_ray.origin	 = shadow_ray_origin;
		shadow_ray.direction = shadow_ray_direction_normalized;

		// abs() here to allow backfacing light sources
		float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);

		if (dot_light_source > 0.0f)
		{
			NEEPlusPlusContext nee_plus_plus_context;
			nee_plus_plus_context.point_on_light = light_sample.point_on_light;
			nee_plus_plus_context.shaded_point	 = shadow_ray_origin;
			bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
															   nee_plus_plus_context, random_number_generator);

			if (!in_shadow)
			{
				float bsdf_pdf;

				BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingStrategy == LSS_BASE_REGIR
				BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
										 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
										 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
#else
				BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
										 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
										 MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
#endif // #if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingStrategy == LSS_BASE_REGIR
				ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

				if (bsdf_pdf != 0.0f)
				{
					// Conversion to solid angle from surface area measure
					float light_sample_solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
					if (light_sample_solid_angle_pdf > 0.0f)
					{
						float cosine_term			= hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction));
						const ColorRGB32F numerator = light_sample.emission * cosine_term * bsdf_color;
						const ColorRGB32F estimator = numerator / light_sample_solid_angle_pdf;
						light_source_radiance += estimator;

						// Just a CPU-only sanity check
						sanity_check</* CPUOnly */ true>(render_data, light_source_radiance, 0, 0);
					}
				}
			}
		}
	}

	return light_source_radiance / DirectLightIntegrationFactor<DirectLightSamplingStrategy>();
}

HIPRT_DEVICE ColorRGB32F sample_one_light_bsdf(const HIPRTRenderData& render_data,
											   RayPayload& ray_payload,
											   const HitInfo closest_hit_info,
											   const float3_t& view_direction,
											   Xorshift32Generator& random_number_generator)
{
	float bsdf_sample_pdf;
	float3_t sampled_bsdf_direction;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;

	BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f),
							 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
							 MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
	ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

	ColorRGB32F bsdf_radiance = ColorRGB32F(0.0f);
	if (bsdf_sample_pdf > 0.0f)
	{
		hiprtRay new_ray;
		new_ray.origin	  = closest_hit_info.inter_point;
		new_ray.direction = sampled_bsdf_direction;

		BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
		bool intersection_found =
			evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, closest_hit_info.primitive_index, random_number_generator);

		// Checking that we did hit something and if we hit something,
		// it needs to be emissive
		if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black() &&
			compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction) > 0.0f)
		{
			float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_bsdf_direction));
			bsdf_radiance	  = bsdf_color * cosine_term * shadow_light_ray_hit_info.hit_emission / bsdf_sample_pdf;

			// Just a CPU-only sanity check
			sanity_check</* CPUOnly */ true>(render_data, bsdf_radiance, 0, 0);
		}
	}

	return bsdf_radiance;
}

HIPRT_DEVICE ColorRGB32F sample_one_light_MIS_deferred_BSDF(HIPRTRenderData& render_data,
															RayPayload& ray_payload,
															const HitInfo closest_hit_info,
															const float3_t& view_direction,
															Xorshift32Generator& random_number_generator)
{
	ColorRGB32F light_source_radiance_mis;

	if (ray_payload.material.can_do_light_sampling())
	{
		LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
			sample_one_point_on_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
									  closest_hit_info.geometric_normal, closest_hit_info.primitive_index, ray_payload, random_number_generator);

		for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
		{
			LightSamplePointInformation& light_sample = light_samples[i];

			// Can happen for very small triangles that the PDF of the sampled triangle couldn't be computed
			if (light_sample.area_measure_pdf > 0.0f)
			{
				float3_t shadow_ray_direction			 = light_sample.point_on_light - closest_hit_info.inter_point;
				float distance_to_light					 = hippt::length(shadow_ray_direction);
				float3_t shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

				hiprtRay shadow_ray;
				shadow_ray.origin	 = closest_hit_info.inter_point;
				shadow_ray.direction = shadow_ray_direction_normalized;

				NEEPlusPlusContext nee_plus_plus_context;
				nee_plus_plus_context.point_on_light = light_sample.point_on_light;
				nee_plus_plus_context.shaded_point	 = shadow_ray.origin;
				bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																   nee_plus_plus_context, random_number_generator);

				if (!in_shadow)
				{
					float bsdf_pdf;
					BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
					BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
											 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
											 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
					ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

					if (bsdf_pdf > 0.0f)
					{
						float cos_theta_at_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);

						// Preventing division by 0 in the conversion to solid angle here
						if (cos_theta_at_light_source > 1.0e-5f)
						{
							float light_sample_solid_angle_pdf =
								area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, cos_theta_at_light_source);
							float mis_weight =
								balance_heuristic(light_sample_solid_angle_pdf, DirectLightIntegrationFactor<DirectLightSamplingStrategy>(), bsdf_pdf, 1);

							float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction));
							light_source_radiance_mis += bsdf_color * cosine_term * light_sample.emission * mis_weight / light_sample_solid_angle_pdf;

							// Just a CPU-only sanity check
							sanity_check</* CPUOnly */ true>(render_data, light_source_radiance_mis, 0, 0);
						}
					}
				}
			}
		}
	}

	// Returning only the light source radiance but that is weighted by the MIS weight. This is "waiting" for a bounce ray to hit a light and perform MIS
	// weighting there
	return light_source_radiance_mis;
}

HIPRT_DEVICE ColorRGB32F sample_one_light_MIS_multi_sample(HIPRTRenderData& render_data,
														   RayPayload& ray_payload,
														   const HitInfo closest_hit_info,
														   const float3_t& view_direction,
														   Xorshift32Generator& random_number_generator)
{
	ColorRGB32F light_source_radiance_mis;

	if (ray_payload.material.can_do_light_sampling())
	{
		LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
			sample_one_point_on_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
									  closest_hit_info.geometric_normal, closest_hit_info.primitive_index, ray_payload, random_number_generator);

		for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
		{
			LightSamplePointInformation& light_sample = light_samples[i];

			// Can happen for very small triangles that the PDF of the sampled triangle couldn't be computed
			if (light_sample.area_measure_pdf > 0.0f)
			{
				float3_t shadow_ray_direction			 = light_sample.point_on_light - closest_hit_info.inter_point;
				float distance_to_light					 = hippt::length(shadow_ray_direction);
				float3_t shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

				hiprtRay shadow_ray;
				shadow_ray.origin	 = closest_hit_info.inter_point;
				shadow_ray.direction = shadow_ray_direction_normalized;

				NEEPlusPlusContext nee_plus_plus_context;
				nee_plus_plus_context.point_on_light = light_sample.point_on_light;
				nee_plus_plus_context.shaded_point	 = shadow_ray.origin;
				bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																   nee_plus_plus_context, random_number_generator);

				if (!in_shadow)
				{
					float bsdf_pdf;
					BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
					BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
											 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
											 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
					ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

					if (bsdf_pdf > 0.0f)
					{
						float cos_theta_at_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);

						// Preventing division by 0 in the conversion to solid angle here
						if (cos_theta_at_light_source > 1.0e-5f)
						{
							float light_sample_solid_angle_pdf =
								area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, cos_theta_at_light_source);
							float mis_weight =
								balance_heuristic(light_sample_solid_angle_pdf, DirectLightIntegrationFactor<DirectLightSamplingStrategy>(), bsdf_pdf, 1);

							float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction));
							light_source_radiance_mis += bsdf_color * cosine_term * light_sample.emission * mis_weight / light_sample_solid_angle_pdf;

							// Just a CPU-only sanity check
							sanity_check</* CPUOnly */ true>(render_data, light_source_radiance_mis, 0, 0);
						}
					}
				}
			}
		}
	}

	float bsdf_sample_pdf;
	float3_t sampled_bsdf_direction;
	float3_t bsdf_shadow_ray_origin			  = closest_hit_info.inter_point;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	ColorRGB32F bsdf_radiance_mis;

	unsigned int previous_seed = random_number_generator.m_state.seed;

	random_number_generator.m_state.seed = previous_seed;
	BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f),
							 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
							 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
	ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, sampled_bsdf_direction, bsdf_sample_pdf, random_number_generator);

	if (bsdf_sample_pdf > 0.0f)
	{
		hiprtRay new_ray;
		new_ray.origin	  = bsdf_shadow_ray_origin;
		new_ray.direction = sampled_bsdf_direction;

		BSDFLightSampleRayHitInfo shadow_light_ray_hit_info;
		bool intersection_found =
			evaluate_bsdf_light_sample_ray(render_data, new_ray, 1.0e35f, shadow_light_ray_hit_info, closest_hit_info.primitive_index, random_number_generator);

		// Checking that we did hit something and if we hit something,
		// it needs to be emissive
		//
		// We're also checking if the light is backfacing maybe with compute_cosine_term()
		if (intersection_found && !shadow_light_ray_hit_info.hit_emission.is_black() &&
			compute_cosine_term_at_light_source(shadow_light_ray_hit_info.hit_geometric_normal, -sampled_bsdf_direction) > 0.0f)
		{
			float light_pdf_solid_angle =
				pdf_of_emissive_triangle_hit_solid_angle(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
														 ray_payload.material, shadow_light_ray_hit_info, sampled_bsdf_direction);
			float mis_weight = balance_heuristic(bsdf_sample_pdf, 1, light_pdf_solid_angle, DirectLightIntegrationFactor<DirectLightSamplingStrategy>());

			// Using abs here because we want the dot product to be positive.
			// You may be thinking that if we're doing this, then we're not going to discard BSDF
			// sampled direction that are below the surface (whereas we should discard them).
			// That would be correct but bsdf_dispatcher_sample return a PDF == 0.0f if a bad
			// direction was sampled and if the PDF is 0.0f, we never get to this line of code
			// you're reading. If we are here, this is because we sampled a direction that is
			// correct for the BSDF. Even if the direction is correct, the dot product may be
			// negative in the case of refractions / total internal reflections and so in this case,
			// we'll need to negative the dot product for it to be positive
			float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_bsdf_direction));
			bsdf_radiance_mis = bsdf_color * cosine_term * shadow_light_ray_hit_info.hit_emission * mis_weight / bsdf_sample_pdf;

			// Just a CPU-only sanity check
			sanity_check</* CPUOnly */ true>(render_data, bsdf_radiance_mis, 0, 0);
		}
	}

	return light_source_radiance_mis + bsdf_radiance_mis;
}

// A selected learned cluster can have no SG-valid descendant for the current shading context. Keep that failed choice observable to the
// learning-to-cluster update as a zero-reward observation.
HIPRT_DEVICE void append_failed_light_clustering_training_sample(HIPRTRenderData& render_data,
																 const float3_t& position,
																 const IlluminationAwareKDTreeSGShadingContext& shading_context,
																 unsigned int mesh_id,
																 const IlluminationAwareKDTreeLearningToClusterCutTriangleSample& triangle_sample)
{
	if (triangle_sample.lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX ||
		triangle_sample.cluster_node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || !(triangle_sample.cluster_probability > 0.0f))
		return;

	IlluminationAwareKDTreeLearningToClusterTrainingSample training_sample{};
	training_sample.position					= position;
	training_sample.mesh_id						= mesh_id;
	training_sample.shading_context				= shading_context;
	training_sample.selected_cluster_node_index = triangle_sample.cluster_node_index;
	training_sample.cluster_probability			= triangle_sample.cluster_probability;
	training_sample.sampled_lightcut_index		= triangle_sample.lightcut_index;
	training_sample.selected_lightcut_slot		= triangle_sample.lightcut_slot;
	training_sample.sampled_lightcut_size		= triangle_sample.lightcut_size_at_sampling;
	training_sample.valid_for_lightcut			= true;

	render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(training_sample);
}

template <bool use_MIS>
HIPRT_DEVICE ColorRGB32F sample_one_light_SG_tree_learning_to_cluster(HIPRTRenderData& render_data,
																	  RayPayload& ray_payload,
																	  const HitInfo closest_hit_info,
																	  const float3_t& view_direction,
																	  Xorshift32Generator& random_number_generator,
																	  int2_t pixel_coords)
{
	if (!ray_payload.material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	IlluminationAwareKDTreeSGShadingContext shading_context =
		build_light_clustering_shading_context(closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, ray_payload.material);
	unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
	if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && closest_hit_info.primitive_index >= 0)
		mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[closest_hit_info.primitive_index];

	IlluminationAwareKDTreeLearningToClusterCutTriangleSample triangle_sample =
		sample_one_emissive_triangle_learning_to_cluster(render_data, shading_context, mesh_id, random_number_generator);

	if (!triangle_sample.valid())
	{
		append_failed_light_clustering_training_sample(render_data, closest_hit_info.inter_point, shading_context, mesh_id, triangle_sample);

		return ColorRGB32F(0.0f);
	}

	IlluminationAwareKDTreeLearningToClusterTrainingSample learning_to_cluster_training_sample{};
	learning_to_cluster_training_sample.position					   = closest_hit_info.inter_point;
	learning_to_cluster_training_sample.mesh_id						   = mesh_id;
	learning_to_cluster_training_sample.shading_context				   = shading_context;
	learning_to_cluster_training_sample.selected_cluster_node_index	   = triangle_sample.cluster_node_index;
	learning_to_cluster_training_sample.emissive_triangle_global_index = triangle_sample.emissive_triangle_global_index;
	learning_to_cluster_training_sample.cluster_probability			   = triangle_sample.cluster_probability;
	learning_to_cluster_training_sample.sampled_lightcut_index		   = triangle_sample.lightcut_index;
	learning_to_cluster_training_sample.selected_lightcut_slot		   = triangle_sample.lightcut_slot;
	learning_to_cluster_training_sample.sampled_lightcut_size		   = triangle_sample.lightcut_size_at_sampling;
	learning_to_cluster_training_sample.valid_for_lightcut			   = true;

	LightSamplePointInformation light_sample =
		sample_point_on_light_and_fill_light_sample_information(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
																ray_payload.material, triangle_sample.emissive_triangle_global_index, random_number_generator);
	light_sample.area_measure_pdf *= triangle_sample.triangle_probability();

	if (!(light_sample.area_measure_pdf > 0.0f) || !isfinite(light_sample.area_measure_pdf))
	{
		// Keep a failed point proposal observable to learning-to-cluster as a zero-reward observation without resampling from another proposal.
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return ColorRGB32F(0.0f);
	}

	IlluminationAwareKDTreeDirectIlluminationTrainingSample spatial_training_sample{};
	spatial_training_sample.position				   = closest_hit_info.inter_point;
	spatial_training_sample.valid_for_spatial_training = true;
	spatial_training_sample.cached_guiding_node_index  = triangle_sample.guiding_node_index;

	ColorRGB32F light_source_radiance(0.0f);

	float3_t shadow_direction = light_sample.point_on_light - closest_hit_info.inter_point;
	float distance_to_light	  = hippt::length(shadow_direction);
	shadow_direction /= distance_to_light;

	spatial_training_sample.incoming_direction = shadow_direction;

	hiprtRay shadow_ray;
	shadow_ray.origin		 = closest_hit_info.inter_point;
	shadow_ray.direction	 = shadow_direction;
	bool shadow_ray_occluded = true;

	float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_direction);
	if (dot_light_source > 0.0f)
	{
		NEEPlusPlusContext nee_plus_plus_context;
		nee_plus_plus_context.point_on_light = light_sample.point_on_light;
		nee_plus_plus_context.shaded_point	 = closest_hit_info.inter_point;

		shadow_ray_occluded = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																nee_plus_plus_context, random_number_generator);
		if (!shadow_ray_occluded)
		{
			float solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
			if (solid_angle_pdf > 0.0f && isfinite(solid_angle_pdf))
			{
				spatial_training_sample.spatial_radiance_weight = (light_sample.emission / solid_angle_pdf).max_component();

				float bsdf_pdf													 = 0.0f;
				BSDFIncidentLightInfo incident_light_info						 = BSDFIncidentLightInfo::NO_INFO;
				MicrofacetRegularization::RegularizationMode regularization_mode = use_MIS
																					   ? MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS
																					   : MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC;
				BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_direction,
										 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
										 regularization_mode);
				ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

				if (bsdf_pdf != 0.0f)
				{
					float cosine_term	  = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));
					ColorRGB32F numerator = light_sample.emission * cosine_term * bsdf_color;
					ColorRGB32F estimator = numerator / solid_angle_pdf;
					float mis_weight	  = 1.0f;
					if constexpr (use_MIS)
						mis_weight = balance_heuristic(solid_angle_pdf, 1, bsdf_pdf, 1);

					ColorRGB32F mis_estimator = estimator * mis_weight;
					light_source_radiance += mis_estimator;

					// Cancel only cluster selection: Q learns the conditional contribution assigned to NEE by MIS.
					float cluster_reward									 = mis_estimator.luminance() * triangle_sample.cluster_probability;
					learning_to_cluster_training_sample.q_reward			 = cluster_reward;
					learning_to_cluster_training_sample.variance_observation = cluster_reward;

					// Just a CPU-only sanity check
					sanity_check</* CPUOnly */ true>(render_data, light_source_radiance, 0, 0);
				}
			}
		}
	}

	render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
	render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

	return light_source_radiance;
}

HIPRT_DEVICE ColorRGB32F sample_one_light_no_MIS_SG_tree_learning_to_cluster(HIPRTRenderData& render_data,
																			 RayPayload& ray_payload,
																			 const HitInfo closest_hit_info,
																			 const float3_t& view_direction,
																			 Xorshift32Generator& random_number_generator,
																			 int2_t pixel_coords)
{
	return sample_one_light_SG_tree_learning_to_cluster<false>(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator,
															   pixel_coords);
}

HIPRT_DEVICE ColorRGB32F sample_one_light_MIS_SG_tree_learning_to_cluster(HIPRTRenderData& render_data,
																		  RayPayload& ray_payload,
																		  const HitInfo closest_hit_info,
																		  const float3_t& view_direction,
																		  Xorshift32Generator& random_number_generator,
																		  int2_t pixel_coords)
{
	return sample_one_light_SG_tree_learning_to_cluster<true>(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator,
															  pixel_coords);
}

HIPRT_DEVICE ColorRGB32F sample_one_light_bsdf_MIS_SG_tree_learning_to_cluster(HIPRTRenderData& render_data,
																			   RayPayload& ray_payload,
																			   const HitInfo& closest_hit_info,
																			   const float3_t& view_direction,
																			   Xorshift32Generator& random_number_generator)
{
	float bsdf_pdf;
	float3_t bsdf_direction;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, make_float3(0.0f, 0.0f, 0.0f),
							 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
							 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
	ColorRGB32F bsdf_color = bsdf_dispatcher_sample(render_data, bsdf_context, bsdf_direction, bsdf_pdf, random_number_generator);
	if (bsdf_pdf <= 0.0f)
		return ColorRGB32F(0.0f);

	hiprtRay shadow_ray;
	shadow_ray.origin	 = closest_hit_info.inter_point;
	shadow_ray.direction = bsdf_direction;
	BSDFLightSampleRayHitInfo light_hit_info;
	bool intersection_found =
		evaluate_bsdf_light_sample_ray(render_data, shadow_ray, 1.0e35f, light_hit_info, closest_hit_info.primitive_index, random_number_generator);
	if (!intersection_found || light_hit_info.hit_emission.is_black() ||
		compute_cosine_term_at_light_source(light_hit_info.hit_geometric_normal, -bsdf_direction) <= 0.0f)
		return ColorRGB32F(0.0f);

	IlluminationAwareKDTreeSGShadingContext shading_context =
		build_light_clustering_shading_context(closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, ray_payload.material);
	unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
	if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && closest_hit_info.primitive_index >= 0)
		mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[closest_hit_info.primitive_index];

	float light_pdf = pdf_of_emissive_triangle_hit_solid_angle_learning_to_cluster(
		render_data, shading_context, mesh_id, ray_payload.material, light_hit_info.hit_prim_index,
		closest_hit_info.inter_point + light_hit_info.hit_distance * bsdf_direction, light_hit_info.hit_geometric_normal);
	float mis_weight  = balance_heuristic(bsdf_pdf, 1, light_pdf, 1);
	float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, bsdf_direction));

	return bsdf_color * cosine_term * light_hit_info.hit_emission * mis_weight / bsdf_pdf;
}

template <bool deferred_BSDF_MIS = true>
HIPRT_DEVICE ColorRGB32F sample_one_light_ReSTIR_DI(HIPRTRenderData& render_data,
													RayPayload& ray_payload,
													const HitInfo closest_hit_info,
													const float3_t& view_direction,
													int2_t pixel_coords,
													NEEDeferredMISContext& out_nee_mis_context,
													Xorshift32Generator& random_number_generator)
{
	// ReSTIR DI doesn't support explicitely looping to sample
	// multiple lights per shading point so that's why we don't
	// have a loop for it

	ColorRGB32F direct_light_contribution;
	if (ray_payload.bounce == 0)
		// Can only do ReSTIR DI on the first bounce
		direct_light_contribution = sample_light_ReSTIR_DI(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, pixel_coords);
	else
	{
		// ReSTIR DI isn't used for the secondary/tertiary/... bounces
		// so there we can take multiple light samples per path vertex
#if ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT
		direct_light_contribution = sample_one_light_no_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_BSDF
		direct_light_contribution = sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF
		direct_light_contribution = sample_one_light_MIS_deferred_BSDF(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif ReSTIR_DI_LaterBouncesSamplingStrategy ==                                                                                                                \
	RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT // #if ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT
		if constexpr (deferred_BSDF_MIS)
		{
			RISReservoir reservoir =
				sample_lights_RIS_for_deferred_NEE_BSDF_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);

			out_nee_mis_context.fill_ris_reservoir(reservoir);
		}
		else
			direct_light_contribution = sample_lights_RIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#endif										   // #if ReSTIR_DI_LaterBouncesSamplingStrategy == RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT
	}

	return direct_light_contribution;
}

HIPRT_DEVICE ColorRGB32F shade_one_light_no_MIS_neural_many_lights(HIPRTRenderData& render_data,
																   RayPayload& ray_payload,
																   const HitInfo closest_hit_info,
																   const float3_t& view_direction,
																   Xorshift32Generator& random_number_generator,
																   const NISMLLightSample& nisml_sample)
{
	bool collect_training_record = nisml_sample.emissive_triangle_global_index >= 0 && nisml_sample.cluster_index < NISML_MAX_CLUSTER_COUNT &&
								   nisml_sample.cluster_probability > 0.0f && nisml_sample.conditional_light_probability > 0.0f &&
								   nisml_sample.emissive_triangle_pdf > 0.0f;

	NISMLTrainingSample training_record;
	if (collect_training_record)
	{
		training_record.position		   = closest_hit_info.inter_point;
		training_record.outgoing_direction = view_direction;
		training_record.normal			   = closest_hit_info.shading_normal;
		get_sg_specular_importance_parameters(ray_payload.material, training_record.sg_specular_weight, training_record.alpha_x, training_record.alpha_y);
		training_record.cluster_index				  = static_cast<unsigned char>(nisml_sample.cluster_index);
		training_record.cluster_probability			  = nisml_sample.cluster_probability;
		training_record.conditional_light_probability = nisml_sample.conditional_light_probability;
	}

	LightSamplePointArray<1> light_samples;
	light_samples[0] =
		sample_point_on_light_and_fill_light_sample_information(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
																ray_payload.material, nisml_sample.emissive_triangle_global_index, random_number_generator);

	ColorRGB32F light_source_radiance;
	float point_on_light_area_pdf = light_samples[0].area_measure_pdf;

	light_samples[0].area_measure_pdf *= nisml_sample.emissive_triangle_pdf;

	bool valid_nisml_training_sample = nisml_sample.emissive_triangle_pdf > 0.0f;
	IlluminationAwareKDTreeDirectIlluminationTrainingSample training_sample;
	training_sample.position				   = closest_hit_info.inter_point;
	training_sample.incoming_direction		   = float3_t(0.0f, 0.0f, 0.0f);
	training_sample.valid_for_spatial_training = valid_nisml_training_sample;

	ColorRGB32F numerator(0.0f);
	float visibility = 0.0f;
	for (int i = 0; i < 1; i++)
	{
		LightSamplePointInformation& light_sample = light_samples[i];

		if (light_sample.area_measure_pdf <= 0.0f)
			// Can happen for very small triangles or the light
			// sampling technique couldn't sample a triangle
			continue;

		float3_t shadow_ray_origin				 = closest_hit_info.inter_point;
		float3_t shadow_ray_direction			 = light_sample.point_on_light - shadow_ray_origin;
		float distance_to_light					 = hippt::length(shadow_ray_direction);
		float3_t shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;
		training_sample.incoming_direction		 = shadow_ray_direction_normalized;

		hiprtRay shadow_ray;
		shadow_ray.origin	 = shadow_ray_origin;
		shadow_ray.direction = shadow_ray_direction_normalized;

		// abs() here to allow backfacing light sources
		float dot_light_source = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_ray.direction);

		if (dot_light_source > 0.0f)
		{
			NEEPlusPlusContext nee_plus_plus_context;
			nee_plus_plus_context.point_on_light = light_sample.point_on_light;
			nee_plus_plus_context.shaded_point	 = shadow_ray_origin;
			bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
															   nee_plus_plus_context, random_number_generator);

			if (!in_shadow)
			{
				visibility = 1.0f;

				// Conversion to solid angle from surface area measure
				float light_sample_solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
				if (collect_training_record)
					training_record.point_on_light_pdf_solid_angle = area_to_solid_angle_pdf(point_on_light_area_pdf, distance_to_light, dot_light_source);

				if (light_sample_solid_angle_pdf > 0.0f)
				{
					training_sample.spatial_radiance_weight = (light_sample.emission / light_sample_solid_angle_pdf).max_component();
					float bsdf_pdf;

					BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
#if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingStrategy == LSS_BASE_REGIR
					BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
											 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
											 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
#else
					BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_ray.direction,
											 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
											 MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
#endif // #if ReGIR_ShadingResamplingDoBSDFMIS == KERNEL_OPTION_TRUE && DirectLightSamplingStrategy == LSS_BASE_REGIR
					ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

					if (bsdf_pdf != 0.0f)
					{
						float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction));

						numerator = light_sample.emission * cosine_term * bsdf_color * visibility;

						ColorRGB32F estimator = numerator / light_sample_solid_angle_pdf;

						light_source_radiance += estimator;

						// Just a CPU-only sanity check
						sanity_check</* CPUOnly */ true>(render_data, light_source_radiance, 0, 0);
					}
				}
			}
		}
	}

	if (valid_nisml_training_sample)
		render_data.kd_tree_device.core.append_direct_illumination_training_sample(training_sample);

	if (collect_training_record)
	{
		training_record.contribution_luminance = numerator.luminance();

		Xorshift32Generator random_number_generator_copy = random_number_generator;
		render_data.nisml.append_training_record(training_record, random_number_generator_copy);
	}

	return light_source_radiance / DirectLightIntegrationFactor<DirectLightSamplingStrategy>();
}

HIPRT_DEVICE ColorRGB32F sample_one_light_no_MIS_neural_many_lights_from_residuals(HIPRTRenderData& render_data,
																				   RayPayload& ray_payload,
																				   const HitInfo closest_hit_info,
																				   const float3_t& view_direction,
																				   float sg_specular_weight,
																				   float alpha_x,
																				   float alpha_y,
																				   Xorshift32Generator& random_number_generator,
																				   const float* residuals,
																				   NISMLLightSample& out_nisml_sample)
{
	out_nisml_sample = NISMLLightSample();

	if (!ray_payload.material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	out_nisml_sample = sample_one_emissive_triangle_neural_many_lights_from_residuals(render_data, closest_hit_info.inter_point, view_direction,
																					  closest_hit_info.shading_normal, ray_payload.material, sg_specular_weight,
																					  alpha_x, alpha_y, random_number_generator, residuals);

	return shade_one_light_no_MIS_neural_many_lights(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, out_nisml_sample);
}

HIPRT_DEVICE ColorRGB32F sample_one_light_no_MIS_neural_many_lights(HIPRTRenderData& render_data,
																	RayPayload& ray_payload,
																	const HitInfo closest_hit_info,
																	const float3_t& view_direction,
																	Xorshift32Generator& random_number_generator)
{
	if (!ray_payload.material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	float representative_sg_specular_weight;
	float representative_alpha_x;
	float representative_alpha_y;
	get_sg_specular_importance_parameters(ray_payload.material, representative_sg_specular_weight, representative_alpha_x, representative_alpha_y);

	IlluminationAwareKDTreeDevice& kd_tree_device = render_data.kd_tree_device;
	unsigned int node_index						  = kd_tree_device.core.find_guiding_cell(closest_hit_info.inter_point);

	Xorshift32Generator representative_random_number_generator = random_number_generator;
	kd_tree_device.nisml.append_nisml_representative(node_index, kd_tree_device.core.node_capacity, closest_hit_info.inter_point, view_direction,
													 closest_hit_info.shading_normal, representative_sg_specular_weight, representative_alpha_x,
													 representative_alpha_y, representative_random_number_generator);

	NISMLLightSample nisml_sample = sample_one_emissive_triangle_neural_many_lights(
		render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, ray_payload.material, random_number_generator);

	return shade_one_light_no_MIS_neural_many_lights(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, nisml_sample);
}

HIPRT_DEVICE ColorRGB32F sample_one_light_LTC_shading(HIPRTRenderData& render_data,
													  RayPayload& ray_payload,
													  const HitInfo closest_hit_info,
													  const float3_t& view_direction,
													  Xorshift32Generator& random_number_generator)
{
	if (!ray_payload.material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	int valid_light_sample_count		= 0;
	ColorRGB32F total_outgoing_radiance = ColorRGB32F(0.0f);

	LightSampleArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
		sample_one_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
						 closest_hit_info.primitive_index, ray_payload, random_number_generator);

	for (int i = 0; i < DirectLightSampleCount<DirectLightSamplingStrategy>(); i++)
	{
		LightSampleInformation& light_sample = light_samples[i];

		if (light_sample.pdf <= 0.0f)
			// Can happen for very small triangles or the light
			// sampling technique couldn't sample a triangle
			return ColorRGB32F(0.0f);

		valid_light_sample_count++;

		float3_t vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[light_sample.emissive_triangle_global_index * 3 + 0]];
		float3_t vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[light_sample.emissive_triangle_global_index * 3 + 1]];
		float3_t vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[light_sample.emissive_triangle_global_index * 3 + 2]];

		float specular_lobe = evaluate_ltc(render_data, vertex_A, vertex_B, vertex_C, closest_hit_info.inter_point, view_direction,
										   closest_hit_info.shading_normal, ray_payload.material, LTCLobe::SPECULAR_LOBE);

		float diffuse_lobe = evaluate_ltc(render_data, vertex_A, vertex_B, vertex_C, closest_hit_info.inter_point, view_direction,
										  closest_hit_info.shading_normal, ray_payload.material, LTCLobe::DIFFUSE_LOBE);

		ColorRGB32F light_sample_emission = triangle_load_emission(render_data, light_sample.emissive_triangle_global_index);
		total_outgoing_radiance += (ColorRGB32F(specular_lobe) + diffuse_lobe * ray_payload.material.base_color) * light_sample_emission;
	}

	return total_outgoing_radiance / valid_light_sample_count;
}

template <bool deferred_BSDF_MIS = true>
HIPRT_DEVICE ColorRGB32F sample_multiple_emissive_geometry(HIPRTRenderData& render_data,
														   RayPayload& ray_payload,
														   const HitInfo closest_hit_info,
														   const float3_t& view_direction,
														   int2_t pixel_coords,
														   NEEDeferredMISContext& out_nee_mis_context,
														   Xorshift32Generator& random_number_generator)
{
	ColorRGB32F direct_light_contribution;

	// Any of these light sampling strategy support sampling multiple lights
	// per each shading point, effectively "amortizing" camera and bounce rays
#if DirectLightSamplingStrategy == LSS_BASE_REGIR && DirectLightNEEEstimator != LSS_BSDF
	// ReGIR has its own special path to optimize things a bit.
	//
	// Also, BSDF sampling only can be handled by the usual path because then
	// ReGIR isn't used
	direct_light_contribution = sample_one_light_ReGIR(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);

#else // Not ReGIR // #if DirectLightSamplingStrategy == LSS_BASE_REGIR && DirectLightNEEEstimator != LSS_BSDF

#if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_one_light_no_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_BSDF
	// This code here is legacy. We are now using the main path's bounce for BSDF sampling of lights
	// direct_light_contribution += sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_MIS_LIGHT_BSDF			 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	if constexpr (deferred_BSDF_MIS)
		direct_light_contribution = sample_one_light_MIS_deferred_BSDF(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
	else
		direct_light_contribution = sample_one_light_MIS_multi_sample(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT		 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	if constexpr (deferred_BSDF_MIS)
	{
		RISReservoir reservoir =
			sample_lights_RIS_for_deferred_NEE_BSDF_MIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);

		out_nee_mis_context.fill_ris_reservoir(reservoir);
	}
	else
		direct_light_contribution += sample_lights_RIS(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_RISLTC					 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_lights_RISLTC(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_LTC_SHADING			 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_one_light_LTC_shading(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS		 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_one_light_no_MIS_neural_many_lights(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER	 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution =
		sample_one_light_no_MIS_SG_tree_learning_to_cluster(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, pixel_coords);
#elif DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution =
		sample_one_light_MIS_SG_tree_learning_to_cluster(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator, pixel_coords);
	if constexpr (!deferred_BSDF_MIS)
		// ReSTIR GI's final shading has no path continuation to supply the complementary sample.
		direct_light_contribution +=
			sample_one_light_bsdf_MIS_SG_tree_learning_to_cluster(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#endif														 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT

#endif // #if ReGIR

	return direct_light_contribution;
}

/**
 * Importance sample lights in the scene with NEE
 *
 * Just a random note for myself and maybe future readers
 * that are wondering the same:
 *
 * In the case where we shot a ray (camera ray or indirect bounce ray, doesn't matter)
 * and we hit an emissive material, we should still estimate NEE at that point. i.e. we
 * should also do NEE when standing on emissive materials because emissive materials can
 * reflect light just fine (unless they are blackbodies).
 *
 * Consider a glowing light bulb for example: this is just metal so hot that it glows
 * but because this is metal, it also reflects light.
 *
 * I think the better morale to remember is that the material being emissive doesn't matter at
 * all. As long as the material itself reflects light, then we should do NEE.
 */
template <bool deferred_BSDF_MIS = true>
HIPRT_DEVICE ColorRGB32F sample_emissive_geometry(HIPRTRenderData& render_data,
												  RayPayload& ray_payload,
												  const HitInfo closest_hit_info,
												  const float3_t& view_direction,
												  int2_t pixel_coords,
												  NEEDeferredMISContext& out_nee_mis_context,
												  Xorshift32Generator& random_number_generator)
{
	if (render_data.buffers.emissive_triangles_count == 0 &&
		!(render_data.world_settings.ambient_light_type == AmbientLightType::ENVMAP && DirectLightNEEEstimator == LSS_RESTIR_DI))
		// No emissive geometry in the scene to sample
		// And we're not sampling the envmap with ReSTIR DI which means
		// that we're not sampling anything so return black
		return ColorRGB32F(0.0f);

	if (render_data.bsdfs_data.white_furnace_mode && render_data.bsdfs_data.white_furnace_mode_turn_off_emissives)
		return ColorRGB32F(0.0f);

	ColorRGB32F direct_light_contribution;
#if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING
	direct_light_contribution = ColorRGB32F(0.0f);
#else // A light sampling strategy is used

#if DirectLightNEEEstimator != LSS_RESTIR_DI
	// A light sampling strategy that is not ReSTIR DI
	// meaning that we can sample more than 1 light per
	// path vertex
	direct_light_contribution = sample_multiple_emissive_geometry<deferred_BSDF_MIS>(render_data, ray_payload, closest_hit_info, view_direction, pixel_coords,
																					 out_nee_mis_context, random_number_generator);
#elif DirectLightNEEEstimator == LSS_RESTIR_DI // #if DirectLightNEEEstimator != LSS_RESTIR_DI
	direct_light_contribution = sample_one_light_ReSTIR_DI<deferred_BSDF_MIS>(render_data, ray_payload, closest_hit_info, view_direction, pixel_coords,
																			  out_nee_mis_context, random_number_generator);
#endif										   // #if DirectLightNEEEstimator != LSS_RESTIR_DI
#endif										   // #if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING

	return direct_light_contribution;
}

HIPRT_DEVICE ColorRGB32F clamp_direct_lighting_estimation(ColorRGB32F direct_lighting_contribution, float direct_contribution_clamp, int bounce)
{
	return clamp_light_contribution(direct_lighting_contribution, direct_contribution_clamp, bounce > 0);
}

/**
 * The x & y parameters are only used if using ReSTIR DI (they are for fetching the ReSTIR DI reservoir).
 * They can be ignored if not using ReSTIR DI
 */
template <bool deferred_BSDF_MIS = true>
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting_from_emissive_contribution(HIPRTRenderData& render_data,
																			 RayPayload& ray_payload,
																			 ColorRGB32F ray_throughput,
																			 ColorRGB32F emissive_geometry_direct_contribution,
																			 HitInfo& closest_hit_info,
																			 float3_t view_direction,
																			 Xorshift32Generator& random_number_generator)
{
	ColorRGB32F total_direct_lighting;

	ColorRGB32F envmap_direct_contribution = sample_environment_map(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);

	// Clamping direct lighting
	emissive_geometry_direct_contribution =
		clamp_light_contribution(emissive_geometry_direct_contribution, render_data.render_settings.direct_contribution_clamp, ray_payload.bounce == 0);
	envmap_direct_contribution =
		clamp_light_contribution(envmap_direct_contribution, render_data.render_settings.envmap_contribution_clamp, ray_payload.bounce == 0);

#if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING // No direct light sampling
	// This if() rejects backfacing lights if backfacing lights are disabled
	if (compute_cosine_term_at_light_source(closest_hit_info.original_geometric_normal(), view_direction) > 0.0f)
	{
		ColorRGB32F hit_emission = ray_payload.material.get_emission();

		hit_emission = clamp_light_contribution(hit_emission, render_data.render_settings.indirect_contribution_clamp, ray_payload.bounce > 0);

		if (render_data.render_settings.enable_direct_lighting || ray_payload.bounce > 1)
			total_direct_lighting += hit_emission * ray_throughput;
	}
#else  // #if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING
	if (ray_payload.bounce == 0 && compute_cosine_term_at_light_source(closest_hit_info.original_geometric_normal(), view_direction) > 0.0f)
		// If we do have emissive geometry sampling, we only want to take
		// it into account on the first bounce, otherwise we would be
		// accounting for direct light sampling twice (bounce on emissive
		// geometry + direct light sampling). Otherwise, we don't check for bounce == 0
		total_direct_lighting += ray_payload.material.get_emission();

	// Clamped indirect lighting
	ColorRGB32F direct_lighting_contribution = (emissive_geometry_direct_contribution + envmap_direct_contribution) * ray_throughput;

	total_direct_lighting += direct_lighting_contribution;
#endif // #if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING

	return total_direct_lighting;
}

template <bool deferred_BSDF_MIS = true>
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting(HIPRTRenderData& render_data,
												  RayPayload& ray_payload,
												  ColorRGB32F ray_throughput,
												  HitInfo& closest_hit_info,
												  float3_t view_direction,
												  int x,
												  int y,
												  NEEDeferredMISContext& out_nee_mis_context,
												  Xorshift32Generator& random_number_generator)
{
	ColorRGB32F emissive_geometry_direct_contribution = sample_emissive_geometry<deferred_BSDF_MIS>(
		render_data, ray_payload, closest_hit_info, view_direction, make_int2(x, y), out_nee_mis_context, random_number_generator);

	return estimate_direct_lighting_from_emissive_contribution<deferred_BSDF_MIS>(
		render_data, ray_payload, ray_throughput, emissive_geometry_direct_contribution, closest_hit_info, view_direction, random_number_generator);
}

template <bool deferred_BSDF_MIS = true>
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting_from_emissive_contribution(HIPRTRenderData& render_data,
																			 RayPayload& ray_payload,
																			 ColorRGB32F emissive_geometry_direct_contribution,
																			 HitInfo& closest_hit_info,
																			 float3_t view_direction,
																			 Xorshift32Generator& random_number_generator)
{
	ColorRGB32F unclamped_direct_lighting = estimate_direct_lighting_from_emissive_contribution<deferred_BSDF_MIS>(
		render_data, ray_payload, ray_payload.throughput, emissive_geometry_direct_contribution, closest_hit_info, view_direction, random_number_generator);

	return clamp_direct_lighting_estimation(unclamped_direct_lighting, render_data.render_settings.indirect_contribution_clamp, ray_payload.bounce);
}

/**
 * The x & y parameters are only used if using ReSTIR DI (they are for fetching the ReSTIR DI reservoir).
 * They can be ignored if not using ReSTIR DI
 */
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting_no_clamping(HIPRTRenderData& render_data,
															  RayPayload& ray_payload,
															  ColorRGB32F ray_throughput,
															  HitInfo& closest_hit_info,
															  float3_t view_direction,
															  int x,
															  int y,
															  NEEDeferredMISContext& out_nee_mis_context,
															  Xorshift32Generator& random_number_generator)
{
	return estimate_direct_lighting(render_data, ray_payload, ray_throughput, closest_hit_info, view_direction, x, y, out_nee_mis_context,
									random_number_generator);
}

/**
 * The x & y parameters are only used if using ReSTIR DI (they are for fetching the ReSTIR DI reservoir).
 * They can be ignored if not using ReSTIR DI
 */
template <bool deferred_BSDF_MIS = true>
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting(HIPRTRenderData& render_data,
												  RayPayload& ray_payload,
												  HitInfo& closest_hit_info,
												  float3_t view_direction,
												  int x,
												  int y,
												  NEEDeferredMISContext& out_nee_mis_context,
												  Xorshift32Generator& random_number_generator)
{
	ColorRGB32F unclamped_direct_lighting = estimate_direct_lighting<deferred_BSDF_MIS>(render_data, ray_payload, ray_payload.throughput, closest_hit_info,
																						view_direction, x, y, out_nee_mis_context, random_number_generator);

	return clamp_direct_lighting_estimation(unclamped_direct_lighting, render_data.render_settings.indirect_contribution_clamp, ray_payload.bounce);
}

HIPRT_DEVICE RISReservoir deferred_NEE_MIS_add_one_RIS_BSDF_sample(HIPRTRenderData& render_data,
																   bool intersection_found,
																   HitInfo& main_path_ray_hit_info,
																   RayPayload& ray_payload,
																   NEEDeferredMISContext& nee_deferred_MIS_context,
																   RISReservoir& reservoir,
																   Xorshift32Generator& random_number_generator)
{
#if PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT

#if DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT
	int nb_light_candidates = render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.ris_settings.number_of_light_candidates;
	int nb_bsdf_candidates	= render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.ris_settings.number_of_bsdf_candidates;

	if (!ray_payload.material.is_emissive() || !intersection_found || nb_bsdf_candidates == 0)
	{
		reservoir.end();

		return reservoir;
	}

	float bsdf_sample_pdf = nee_deferred_MIS_context.last_bsdf_sample_pdf;

	float3_t to_light_direction = main_path_ray_hit_info.inter_point - nee_deferred_MIS_context.last_shading_point;
	float hit_distance			= hippt::length(to_light_direction);
	to_light_direction /= hit_distance;

	RISSample bsdf_RIS_sample;
	float candidate_weight = 0.0f;

	if (bsdf_sample_pdf > 0.0f)
	{
		if (compute_cosine_term_at_light_source(main_path_ray_hit_info.original_geometric_normal(), -to_light_direction) > 0.0f)
		{
			// Our target function does not include the geometry term because we're integrating
			// in solid angle. The geometry term in the target function ( / in the integrand) is only
			// for surface area direct lighting integration
			ColorRGB32F hit_emission	   = ray_payload.material.get_emission();
			ColorRGB32F light_contribution = nee_deferred_MIS_context.last_bsdf_x_cos_theta * hit_emission;
			float target_function		   = light_contribution.luminance();

			float light_pdf = pdf_of_emissive_triangle_hit_solid_angle(
				render_data, nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction,
				nee_deferred_MIS_context.last_shading_normal, nee_deferred_MIS_context.last_material, main_path_ray_hit_info.primitive_index,
				main_path_ray_hit_info.original_geometric_normal(), hit_distance, to_light_direction);

			float mis_weight = balance_heuristic(bsdf_sample_pdf, nb_bsdf_candidates, light_pdf,
												 nb_light_candidates * DirectLightIntegrationFactor<DirectLightSamplingStrategy>());
			candidate_weight = mis_weight * target_function / bsdf_sample_pdf;

			bsdf_RIS_sample.emission				 = hit_emission;
			bsdf_RIS_sample.point_on_light_source	 = main_path_ray_hit_info.inter_point;
			bsdf_RIS_sample.is_bsdf_sample			 = true;
			bsdf_RIS_sample.bsdf_sample_contribution = nee_deferred_MIS_context.last_bsdf_x_cos_theta;
			// The RIS integrator will compute bsdf_sample_contribution * bsdf_sample_cosine_term / ... when evaluating the reservoir but our BSDF sample
			// contribution here already contains the cosine term so we're setting cosine term = 1.0f in the reservoir's sample
			bsdf_RIS_sample.bsdf_sample_cosine_term = 1.0f;
			bsdf_RIS_sample.target_function			= target_function;
		}
	}

	reservoir.add_one_candidate(bsdf_RIS_sample, candidate_weight, random_number_generator);
	reservoir.sanity_check();

	reservoir.end();

	return reservoir;
#else  // #if DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT
	return RISReservoir();
#endif // #if DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT

#endif // PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT // #if PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT

	return RISReservoir();
}

/**
 * If the bounce ray of the main path hits an emissive light, computes the MIS weight for that emissive hit against the light sampler of the last hit and
 * returns the contribution of that emissive hit with that MIS weight.
 */
[[nodiscard]] HIPRT_DEVICE ColorRGB32F do_deferred_NEE_MIS(HIPRTRenderData& render_data,
														   bool intersection_found,
														   RayPayload& ray_payload,
														   HitInfo& light_hit_info,
														   NEEDeferredMISContext& nee_deferred_MIS_context,
														   Xorshift32Generator& random_number_generator)
{
#if PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT

#if !DirectLightNEEEstimatorHasBSDFSampling
	return ColorRGB32F(0.0f);
#else
	if (ray_payload.bounce == 0 && !render_data.render_settings.enable_direct_lighting)
		// Deferred NEE MIS for the primary hit but we're not doing direct lighting
		return ColorRGB32F(0.0f);
	else if (nee_deferred_MIS_context.last_bsdf_sample_pdf <= 0.0f)
		return ColorRGB32F(0.0f);

#if DirectLightNEEEstimator == LSS_BSDF
	if (!ray_payload.material.is_emissive() || !intersection_found)
		return ColorRGB32F(0.0f);
	else if (compute_cosine_term_at_light_source(light_hit_info.original_geometric_normal(),
												 hippt::normalize(nee_deferred_MIS_context.last_shading_point - light_hit_info.inter_point)) <= 0.0f)
		// If the light is backfacing and backfacing lights are disabled, then we don't want to add its contribution
		return ColorRGB32F(0.0f);

	float bsdf_sample_mis_weight = 1.0f;

	return nee_deferred_MIS_context.last_ray_throughput * ray_payload.material.get_emission() * nee_deferred_MIS_context.last_bsdf_x_cos_theta /
		   nee_deferred_MIS_context.last_bsdf_sample_pdf * bsdf_sample_mis_weight;
#elif DirectLightNEEEstimator == LSS_MIS_LIGHT_BSDF || DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS // #if DirectLightNEEEstimator == LSS_BSDF
	if (!ray_payload.material.is_emissive() || !intersection_found)
		return ColorRGB32F(0.0f);
	else if (compute_cosine_term_at_light_source(light_hit_info.original_geometric_normal(),
												 hippt::normalize(nee_deferred_MIS_context.last_shading_point - light_hit_info.inter_point)) <= 0.0f)
		// If the light is backfacing and backfacing lights are disabled, then we don't want to add its contribution
		return ColorRGB32F(0.0f);

	ColorRGB32F hit_emission = ray_payload.material.get_emission();

	float3_t ray_direction = light_hit_info.inter_point - nee_deferred_MIS_context.last_shading_point;
	float hit_distance	   = hippt::length(ray_direction);
	ray_direction /= hit_distance;

#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	float light_sampler_solid_angle_pdf = 0.0f;
	if (nee_deferred_MIS_context.last_material.can_do_light_sampling())
	{
		IlluminationAwareKDTreeSGShadingContext shading_context =
			build_light_clustering_shading_context(nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction,
												   nee_deferred_MIS_context.last_shading_normal, nee_deferred_MIS_context.last_material);
		unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
		if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && nee_deferred_MIS_context.last_primitive_index >= 0)
			mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[nee_deferred_MIS_context.last_primitive_index];

		// Learning updates run after the megakernel, so this resolves the same cut and CDF used at the previous vertex.
		light_sampler_solid_angle_pdf = pdf_of_emissive_triangle_hit_solid_angle_learning_to_cluster(
			render_data, shading_context, mesh_id, nee_deferred_MIS_context.last_material, light_hit_info.primitive_index, light_hit_info.inter_point,
			light_hit_info.original_geometric_normal());
	}

	float bsdf_sample_mis_weight = balance_heuristic(nee_deferred_MIS_context.last_bsdf_sample_pdf, 1, light_sampler_solid_angle_pdf, 1);
#else													// #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	float light_sampler_solid_angle_pdf = pdf_of_emissive_triangle_hit_solid_angle(
		render_data, nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction, nee_deferred_MIS_context.last_shading_normal,
		nee_deferred_MIS_context.last_material, light_hit_info.primitive_index, light_hit_info.original_geometric_normal(), hit_distance, ray_direction);

	float bsdf_sample_mis_weight = balance_heuristic(nee_deferred_MIS_context.last_bsdf_sample_pdf, 1, light_sampler_solid_angle_pdf,
													 DirectLightIntegrationFactor<DirectLightSamplingStrategy>());
#endif													// #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS

	return nee_deferred_MIS_context.last_ray_throughput * hit_emission * nee_deferred_MIS_context.last_bsdf_x_cos_theta /
		   nee_deferred_MIS_context.last_bsdf_sample_pdf * bsdf_sample_mis_weight;
#elif DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT // #if DirectLightNEEEstimator == LSS_BSDF
	RISReservoir final_reservoir =
		deferred_NEE_MIS_add_one_RIS_BSDF_sample(render_data, intersection_found, light_hit_info, ray_payload, nee_deferred_MIS_context,
												 nee_deferred_MIS_context.ris_reservoir, random_number_generator);
	if (final_reservoir.UCW == 0.0f)
		return ColorRGB32F(0.0f);

	HitInfo last_hit_info;
	last_hit_info.inter_point	   = nee_deferred_MIS_context.last_shading_point;
	last_hit_info.geometric_normal = nee_deferred_MIS_context.last_geometric_normal;
	last_hit_info.shading_normal   = nee_deferred_MIS_context.last_shading_normal;
	last_hit_info.primitive_index  = nee_deferred_MIS_context.last_primitive_index;

	RayPayload last_hit_payload	  = ray_payload;
	last_hit_payload.bounce		  = ray_payload.bounce;
	last_hit_payload.material	  = nee_deferred_MIS_context.last_material;
	last_hit_payload.volume_state = nee_deferred_MIS_context.last_volume_state;

	ColorRGB32F last_hit_NEE_estimate = evaluate_RIS_reservoir_sample(render_data, last_hit_payload, last_hit_info,
																	  nee_deferred_MIS_context.last_view_direction, final_reservoir, random_number_generator);

	nee_deferred_MIS_context.ris_reservoir = RISReservoir();
	return last_hit_NEE_estimate * nee_deferred_MIS_context.last_ray_throughput;
#endif													// #if DirectLightNEEEstimator == LSS_BSDF

#endif // PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT // #if !DirectLightNEEEstimatorHasBSDFSampling
#endif // DirectLightNEEEstimatorHasBSDFSampling // #if PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT

	return ColorRGB32F();
}

[[nodiscard]] HIPRT_DEVICE ColorRGB32F do_last_deferred_NEE_MIS(HIPRTRenderData& render_data,
																hiprtRay ray,
																RayPayload& ray_payload,
																HitInfo& closest_hit_info,
																Xorshift32Generator& random_number_generator,
																NEEDeferredMISContext& nee_deferred_MIS_context)
{
#if PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT

#if DirectLightNEEEstimatorHasBSDFSampling
	// We will have one more bounce than necessary when getting here and this can throw off the 'max bounce' of alpha testing so we need to substract one bounce
	// here
	ray_payload.bounce--;
	bool intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, random_number_generator);
	// And add it back before deferred NEE MIS so that the code inside deferred NEE MIS receives the bounce index that it expects
	ray_payload.bounce++;

	return do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, closest_hit_info, nee_deferred_MIS_context, random_number_generator);
#endif // #if DirectLightNEEEstimatorHasBSDFSampling

	return ColorRGB32F(0.0f);

#endif // PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT // #if PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT

	return ColorRGB32F();
}

#endif // #ifndef DEVICE_NEE_ESTIMATORS_H
