/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_NEE_ESTIMATORS_H
#define DEVICE_NEE_ESTIMATORS_H

#include "Device/includes/BSDFs/BSDFDispatcher.h"
#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/HitInfo.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDevice.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LearningToClusterEnvmapSelection.h"
#include "Device/includes/LightSampling/NEEDeferredMISContext.h"
#include "Device/includes/LightSampling/RIS/RIS.h"
#include "Device/includes/LightSampling/TriangleEmissiveSampling.h"
#include "Device/includes/PathTracing.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/SanityCheck.h"
#include "Device/includes/Wavefront/WavefrontShadow.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

template <typename BsdfMaterialType>
HIPRT_DEVICE ColorRGB32F evaluate_one_light_no_mis_without_visibility(HIPRTRenderData& render_data,
																	  RayPayloadT<BsdfMaterialType>& ray_payload,
																	  BsdfMaterialType& bsdf_material,
																	  const HitInfo& closest_hit_info,
																	  const float3_t& view_direction,
																	  const LightSamplePointInformation& light_sample,
																	  float3_t shadow_direction,
																	  float distance_to_light,
																	  float dot_light_source,
																	  Xorshift32Generator& random_number_generator)
{
	float bsdf_pdf;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	BSDFContextT<BsdfMaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_direction,
												incident_light_info, ray_payload.volume_state, false, bsdf_material, ray_payload.accumulated_roughness,
												MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

	if (bsdf_pdf == 0.0f)
		return ColorRGB32F(0.0f);

	float light_sample_solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
	if (!(light_sample_solid_angle_pdf > 0.0f))
		return ColorRGB32F(0.0f);

	float cosine_term	  = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));
	ColorRGB32F numerator = light_sample.emission * cosine_term * bsdf_color;

	return numerator / light_sample_solid_angle_pdf / DirectLightIntegrationFactor<DirectLightSamplingStrategy>();
}

template <typename BsdfMaterialType, typename ProposalMaterialType, NEEShadowMode shadow_mode = NEE_SHADOW_INLINE>
HIPRT_DEVICE ColorRGB32F sample_one_light_no_MIS(HIPRTRenderData& render_data,
												 RayPayloadT<BsdfMaterialType>& ray_payload,
												 BsdfMaterialType& bsdf_material,
												 const HitInfo closest_hit_info,
												 const float3_t& view_direction,
												 const ProposalMaterialType& proposal_state,
												 Xorshift32Generator& random_number_generator,
												 WavefrontShadowSink* shadow_sink = nullptr)
{
	if (!bsdf_material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	ColorRGB32F light_source_radiance;

	LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples =
		sample_one_point_on_light(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
								  closest_hit_info.primitive_index, ray_payload, proposal_state, random_number_generator);

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
			if constexpr (shadow_mode == NEE_SHADOW_WAVEFRONT)
			{
				ColorRGB32F contribution_if_visible =
					evaluate_one_light_no_mis_without_visibility(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, light_sample,
																 shadow_ray.direction, distance_to_light, dot_light_source, random_number_generator);
				if (!contribution_if_visible.is_black())
					shadow_sink->emit(shadow_ray.origin, shadow_ray.direction, distance_to_light, closest_hit_info.primitive_index, light_sample.point_on_light,
									  contribution_if_visible, WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS);
			}
			else
			{
				NEEPlusPlusContext nee_plus_plus_context;
				nee_plus_plus_context.point_on_light = light_sample.point_on_light;
				nee_plus_plus_context.shaded_point	 = shadow_ray_origin;

				bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																   nee_plus_plus_context, random_number_generator);
				if (!in_shadow)
					light_source_radiance +=
						evaluate_one_light_no_mis_without_visibility(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, light_sample,
																	 shadow_ray.direction, distance_to_light, dot_light_source, random_number_generator);
			}
		}
	}

	return light_source_radiance;
}

template <typename BsdfMaterialType>
HIPRT_DEVICE ColorRGB32F sample_one_light_bsdf(const HIPRTRenderData& render_data,
											   RayPayloadT<BsdfMaterialType>& ray_payload,
											   BsdfMaterialType& bsdf_material,
											   const HitInfo closest_hit_info,
											   const float3_t& view_direction,
											   Xorshift32Generator& random_number_generator)
{
	float bsdf_sample_pdf;
	float3_t sampled_bsdf_direction;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;

	BSDFContextT<BsdfMaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
												make_float3(0.0f, 0.0f, 0.0f), incident_light_info, ray_payload.volume_state, false, bsdf_material,
												ray_payload.accumulated_roughness, MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC);
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

template <typename BsdfMaterialType>
HIPRT_DEVICE ColorRGB32F evaluate_one_light_mis_without_visibility(HIPRTRenderData& render_data,
																   RayPayloadT<BsdfMaterialType>& ray_payload,
																   BsdfMaterialType& bsdf_material,
																   const HitInfo& closest_hit_info,
																   const float3_t& view_direction,
																   const LightSamplePointInformation& light_sample,
																   float3_t shadow_direction,
																   float distance_to_light,
																   Xorshift32Generator& random_number_generator)
{
	float bsdf_pdf;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	BSDFContextT<BsdfMaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_direction,
												incident_light_info, ray_payload.volume_state, false, bsdf_material, ray_payload.accumulated_roughness,
												MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

	if (!(bsdf_pdf > 0.0f))
		return ColorRGB32F(0.0f);

	float cosine_at_light = compute_cosine_term_at_light_source(light_sample.light_source_normal, -shadow_direction);
	if (!(cosine_at_light > 1.0e-5f))
		return ColorRGB32F(0.0f);

	float light_pdf	  = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, cosine_at_light);
	float mis_weight  = balance_heuristic(light_pdf, DirectLightIntegrationFactor<DirectLightSamplingStrategy>(), bsdf_pdf, 1);
	float cosine_term = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));

	return bsdf_color * cosine_term * light_sample.emission * mis_weight / light_pdf;
}

template <typename BsdfMaterialType, typename ProposalMaterialType, NEEShadowMode shadow_mode = NEE_SHADOW_INLINE>
HIPRT_DEVICE ColorRGB32F sample_one_light_MIS_deferred_BSDF(HIPRTRenderData& render_data,
															RayPayloadT<BsdfMaterialType>& ray_payload,
															BsdfMaterialType& bsdf_material,
															const HitInfo closest_hit_info,
															const float3_t& view_direction,
															const ProposalMaterialType& proposal_state,
															Xorshift32Generator& random_number_generator,
															WavefrontShadowSink* shadow_sink = nullptr)
{
	ColorRGB32F light_source_radiance_mis;

	if (bsdf_material.can_do_light_sampling())
	{
		LightSamplePointArray<DirectLightSampleCount<DirectLightSamplingStrategy>()> light_samples = sample_one_point_on_light(
			render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
			closest_hit_info.primitive_index, ray_payload, proposal_state, random_number_generator);

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

				if constexpr (shadow_mode == NEE_SHADOW_WAVEFRONT)
				{
					ColorRGB32F contribution_if_visible =
						evaluate_one_light_mis_without_visibility(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, light_sample,
																  shadow_ray.direction, distance_to_light, random_number_generator);
					if (!contribution_if_visible.is_black())
						shadow_sink->emit(shadow_ray.origin, shadow_ray.direction, distance_to_light, closest_hit_info.primitive_index,
										  light_sample.point_on_light, contribution_if_visible, WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS);
				}
				else
				{
					NEEPlusPlusContext nee_plus_plus_context;
					nee_plus_plus_context.point_on_light = light_sample.point_on_light;
					nee_plus_plus_context.shaded_point	 = shadow_ray.origin;

					bool in_shadow = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																	   nee_plus_plus_context, random_number_generator);
					if (!in_shadow)
						light_source_radiance_mis +=
							evaluate_one_light_mis_without_visibility(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, light_sample,
																	  shadow_ray.direction, distance_to_light, random_number_generator);
				}
			}
		}
	}

	// Returning only the light source radiance but that is weighted by the MIS weight. This is "waiting" for a bounce ray to hit a light and perform MIS
	// weighting there
	return light_source_radiance_mis;
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

// Forced inline: see the comment above principled_bsdf_eval()
template <bool use_MIS, typename BsdfMaterialType>
HIPRT_DEVICE HIPRT_INLINE ColorRGB32F evaluate_learning_to_cluster_without_visibility(HIPRTRenderData& render_data,
																					  RayPayloadT<BsdfMaterialType>& ray_payload,
																					  BsdfMaterialType& bsdf_material,
																					  const HitInfo& closest_hit_info,
																					  const float3_t& view_direction,
																					  const LightSamplePointInformation& light_sample,
																					  float3_t shadow_direction,
																					  float solid_angle_pdf,
																					  float cluster_probability,
																					  Xorshift32Generator& random_number_generator,
																					  float& spatial_radiance_weight,
																					  float& cluster_reward)
{
	spatial_radiance_weight = (light_sample.emission / solid_angle_pdf).max_component();

	float bsdf_pdf							  = 0.0f;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	MicrofacetRegularization::RegularizationMode regularization_mode =
		use_MIS ? MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS : MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC;
	BSDFContextT<BsdfMaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_direction,
												incident_light_info, ray_payload.volume_state, false, bsdf_material, ray_payload.accumulated_roughness,
												regularization_mode);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

	if (bsdf_pdf == 0.0f)
		return ColorRGB32F(0.0f);

	float cosine_term	  = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));
	ColorRGB32F estimator = light_sample.emission * cosine_term * bsdf_color / solid_angle_pdf;

	float mis_weight = 1.0f;
	if constexpr (use_MIS)
		mis_weight = balance_heuristic(solid_angle_pdf, 1, bsdf_pdf, 1);

	ColorRGB32F mis_estimator = estimator * mis_weight;
	cluster_reward			  = mis_estimator.luminance() * cluster_probability;

	return mis_estimator;
}

/**
 * Light sample of learning to cluster when the selected cluster of the lightcut is the envmap: samples a direction from the envmap distribution,
 * traces the shadow ray towards the envmap and appends the learning-to-cluster and spatial training samples of that light sample.
 *
 * 'learning_to_cluster_training_sample' is already filled with the cluster selection of 'cluster_sample'. 'cluster_sample.cluster_probability'
 * is the probability that the NEE of the shading point sampled the envmap, 1 when the envmap has its own NEE sample
 */
template <bool use_MIS, NEEShadowMode shadow_mode, typename BsdfMaterialType>
HIPRT_DEVICE ColorRGB32F sample_envmap_cluster_learning_to_cluster(HIPRTRenderData& render_data,
																   RayPayloadT<BsdfMaterialType>& ray_payload,
																   BsdfMaterialType& bsdf_material,
																   const HitInfo& closest_hit_info,
																   const float3_t& view_direction,
																   const IlluminationAwareKDTreeLearningToClusterCutTriangleSample& cluster_sample,
																   IlluminationAwareKDTreeLearningToClusterTrainingSample& learning_to_cluster_training_sample,
																   const LearningToClusterPixelObservationTarget& pixel_observation_target,
																   Xorshift32Generator& random_number_generator,
																   WavefrontShadowSink* shadow_sink)
{
	float envmap_pdf_solid_angle;
	float3_t sampled_direction;
	ColorRGB32F envmap_radiance = envmap_sample(render_data.world_settings, sampled_direction, envmap_pdf_solid_angle, random_number_generator);

	float solid_angle_pdf = envmap_pdf_solid_angle * cluster_sample.cluster_probability;
	float cosine_term	  = hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_direction));

	if (!(solid_angle_pdf > 0.0f) || !hippt::is_finite(solid_angle_pdf) || cosine_term <= 0.0f)
	{
		// Keep a failed direction proposal observable to learning-to-cluster as a zero-reward observation, as for the emissive clusters
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return ColorRGB32F(0.0f);
	}

	IlluminationAwareKDTreeDirectIlluminationTrainingSample spatial_training_sample{};
	spatial_training_sample.position				   = closest_hit_info.inter_point;
	spatial_training_sample.incoming_direction		   = sampled_direction;
	spatial_training_sample.valid_for_spatial_training = true;
	spatial_training_sample.cached_guiding_node_index  = cluster_sample.guiding_node_index;

	float bsdf_pdf							  = 0.0f;
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
	MicrofacetRegularization::RegularizationMode regularization_mode =
		use_MIS ? MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS : MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC;
	BSDFContextT<BsdfMaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_direction,
												incident_light_info, ray_payload.volume_state, false, bsdf_material, ray_payload.accumulated_roughness,
												regularization_mode);
	ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

	ColorRGB32F mis_estimator(0.0f);

	if (bsdf_pdf > 0.0f)
	{
		float mis_weight = 1.0f;
		if constexpr (use_MIS)
			mis_weight = balance_heuristic(solid_angle_pdf, 1, bsdf_pdf, 1);

		mis_estimator = envmap_radiance * cosine_term * bsdf_color / solid_angle_pdf * mis_weight;
	}

	// Cancel only cluster selection: Q learns the conditional contribution assigned to NEE by MIS.
	float cluster_reward_if_visible			 = mis_estimator.luminance() * cluster_sample.cluster_probability;
	float spatial_radiance_weight_if_visible = (envmap_radiance / solid_angle_pdf).max_component();
	float pixel_observation_if_visible		 = mis_estimator.luminance();

	hiprtRay shadow_ray;
	shadow_ray.origin	 = closest_hit_info.inter_point;
	shadow_ray.direction = sampled_direction;

	if constexpr (shadow_mode == NEE_SHADOW_WAVEFRONT)
	{
		bool update_training_samples			  = render_data.kd_tree_device.shadow_rays_update_training_samples();
		unsigned int training_update_flag		  = update_training_samples ? WAVEFRONT_SHADOW_UPDATE_LEARNING_TO_CLUSTER : 0u;
		unsigned int pixel_statistics_update_flag = pixel_observation_target.record ? WAVEFRONT_SHADOW_UPDATE_PIXEL_ENVMAP_STATISTICS : 0u;

		// The shadow sink multiplies the contribution by the path throughput
		unsigned int queue_slot =
			shadow_sink->emit(shadow_ray.origin, shadow_ray.direction, 1.0e35f, closest_hit_info.primitive_index, sampled_direction, mis_estimator,
							  WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS | WAVEFRONT_SHADOW_IS_ENVMAP | training_update_flag | pixel_statistics_update_flag);
		if (queue_slot != WAVEFRONT_INVALID_SHADOW_QUEUE_SLOT)
		{
			WavefrontLearningToClusterShadowSidecar& sidecar = render_data.wavefront_data.learning_to_cluster_shadow_sidecars[queue_slot];

			if (update_training_samples)
			{
				sidecar.spatial_training_sample_index = render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
				sidecar.learning_to_cluster_sample_index =
					render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);
				sidecar.spatial_radiance_weight_if_visible = spatial_radiance_weight_if_visible;
				sidecar.cluster_reward_if_visible		   = cluster_reward_if_visible;
			}

			if (pixel_observation_target.record)
				sidecar.pixel_envmap_statistic_observation_if_visible = pixel_observation_if_visible;
		}

		return ColorRGB32F(0.0f);
	}
	else
	{
		NEEPlusPlusContext nee_plus_plus_context;
		nee_plus_plus_context.shaded_point	 = closest_hit_info.inter_point;
		nee_plus_plus_context.point_on_light = sampled_direction;
		nee_plus_plus_context.envmap		 = true;

		bool shadow_ray_occluded = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, 1.0e35f, closest_hit_info.primitive_index, nee_plus_plus_context,
																	 random_number_generator);
		if (!shadow_ray_occluded)
		{
			spatial_training_sample.spatial_radiance_weight			 = spatial_radiance_weight_if_visible;
			learning_to_cluster_training_sample.q_reward			 = cluster_reward_if_visible;
			learning_to_cluster_training_sample.variance_observation = cluster_reward_if_visible;

			if (pixel_observation_target.record)
				add_learning_to_cluster_pixel_envmap_statistic_observation(render_data, pixel_observation_target.pixel_index, true,
																		   pixel_observation_if_visible);
		}

		render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return shadow_ray_occluded ? ColorRGB32F(0.0f) : mis_estimator;
	}
}

/**
 * Light sample of learning to cluster when the selected cluster of the lightcut is an emissive cluster: samples a point on the triangle of
 * 'triangle_sample', traces the shadow ray towards it and appends the learning-to-cluster and spatial training samples of that light sample.
 *
 * 'triangle_sample.cluster_probability' is the probability that the NEE of the shading point sampled that cluster, selection between the envmap
 * and the emissive lights included. 'learning_to_cluster_training_sample' is already filled with that selection
 */
template <bool use_MIS, NEEShadowMode shadow_mode, typename BsdfMaterialType, typename ProposalMaterialType>
HIPRT_DEVICE ColorRGB32F
sample_emissive_cluster_learning_to_cluster(HIPRTRenderData& render_data,
											RayPayloadT<BsdfMaterialType>& ray_payload,
											BsdfMaterialType& bsdf_material,
											const HitInfo& closest_hit_info,
											const float3_t& view_direction,
											const ProposalMaterialType& proposal_state,
											const IlluminationAwareKDTreeLearningToClusterCutTriangleSample& triangle_sample,
											IlluminationAwareKDTreeLearningToClusterTrainingSample& learning_to_cluster_training_sample,
											const LearningToClusterPixelObservationTarget& pixel_observation_target,
											Xorshift32Generator& random_number_generator,
											WavefrontShadowSink* shadow_sink)
{
	LightSamplePointInformation light_sample =
		sample_point_on_light_and_fill_light_sample_information(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
																proposal_state, triangle_sample.emissive_triangle_global_index, random_number_generator);
	light_sample.area_measure_pdf *= triangle_sample.triangle_probability();

	if (!(light_sample.area_measure_pdf > 0.0f) || !hippt::is_finite(light_sample.area_measure_pdf))
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
		if constexpr (shadow_mode == NEE_SHADOW_WAVEFRONT)
		{
			float solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
			if (solid_angle_pdf > 0.0f && hippt::is_finite(solid_angle_pdf))
			{
				float spatial_radiance_weight		= 0.0f;
				float cluster_reward				= 0.0f;
				ColorRGB32F contribution_if_visible = evaluate_learning_to_cluster_without_visibility<use_MIS>(
					render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, light_sample, shadow_direction, solid_angle_pdf,
					triangle_sample.cluster_probability, random_number_generator, spatial_radiance_weight, cluster_reward);

				bool update_training_samples			  = render_data.kd_tree_device.shadow_rays_update_training_samples();
				unsigned int training_update_flag		  = update_training_samples ? WAVEFRONT_SHADOW_UPDATE_LEARNING_TO_CLUSTER : 0u;
				unsigned int pixel_statistics_update_flag = pixel_observation_target.record ? WAVEFRONT_SHADOW_UPDATE_PIXEL_ENVMAP_STATISTICS : 0u;

				unsigned int queue_slot =
					shadow_sink->emit(shadow_ray.origin, shadow_ray.direction, distance_to_light, closest_hit_info.primitive_index, light_sample.point_on_light,
									  contribution_if_visible, WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS | training_update_flag | pixel_statistics_update_flag);
				if (queue_slot != WAVEFRONT_INVALID_SHADOW_QUEUE_SLOT)
				{
					WavefrontLearningToClusterShadowSidecar& sidecar = render_data.wavefront_data.learning_to_cluster_shadow_sidecars[queue_slot];

					if (update_training_samples)
					{
						sidecar.spatial_training_sample_index =
							render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
						sidecar.learning_to_cluster_sample_index =
							render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);
						sidecar.spatial_radiance_weight_if_visible = spatial_radiance_weight;
						sidecar.cluster_reward_if_visible		   = cluster_reward;
					}

					if (pixel_observation_target.record)
						sidecar.pixel_envmap_statistic_observation_if_visible = contribution_if_visible.luminance();
				}

				return ColorRGB32F(0.0f);
			}
		}
		else
		{
			NEEPlusPlusContext nee_plus_plus_context;
			nee_plus_plus_context.point_on_light = light_sample.point_on_light;
			nee_plus_plus_context.shaded_point	 = closest_hit_info.inter_point;

			shadow_ray_occluded = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																	nee_plus_plus_context, random_number_generator);
			if (!shadow_ray_occluded)
			{
				float solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
				if (solid_angle_pdf > 0.0f && hippt::is_finite(solid_angle_pdf))
				{
					spatial_training_sample.spatial_radiance_weight = (light_sample.emission / solid_angle_pdf).max_component();

					float bsdf_pdf							  = 0.0f;
					BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
					MicrofacetRegularization::RegularizationMode regularization_mode =
						use_MIS ? MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS
								: MicrofacetRegularization::RegularizationMode::REGULARIZATION_CLASSIC;
					BSDFContextT<BsdfMaterialType> bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal,
																shadow_direction, incident_light_info, ray_payload.volume_state, false, bsdf_material,
																ray_payload.accumulated_roughness, regularization_mode);
					ColorRGB32F bsdf_color = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);

					if (bsdf_pdf != 0.0f)
					{
						float cosine_term	  = hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));
						ColorRGB32F numerator = light_sample.emission * cosine_term * bsdf_color;
						ColorRGB32F estimator = numerator / solid_angle_pdf;

						float mis_weight = 1.0f;
						if constexpr (use_MIS)
							mis_weight = balance_heuristic(solid_angle_pdf, 1, bsdf_pdf, 1);

						ColorRGB32F mis_estimator = estimator * mis_weight;
						light_source_radiance += mis_estimator;

						// Cancel only cluster selection: Q learns the conditional contribution assigned to NEE by MIS.
						float cluster_reward									 = mis_estimator.luminance() * triangle_sample.cluster_probability;
						learning_to_cluster_training_sample.q_reward			 = cluster_reward;
						learning_to_cluster_training_sample.variance_observation = cluster_reward;

						if (pixel_observation_target.record)
							add_learning_to_cluster_pixel_envmap_statistic_observation(render_data, pixel_observation_target.pixel_index, false,
																					   mis_estimator.luminance());

						// Just a CPU-only sanity check
						sanity_check</* CPUOnly */ true>(render_data, light_source_radiance, 0, 0);
					}
				}
			}
		}
	}

	render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
	render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

	return light_source_radiance;
}

HIPRT_DEVICE IlluminationAwareKDTreeLearningToClusterTrainingSample
make_learning_to_cluster_training_sample(const float3_t& position,
										 unsigned int mesh_id,
										 const IlluminationAwareKDTreeSGShadingContext& shading_context,
										 const IlluminationAwareKDTreeLearningToClusterCutTriangleSample& cluster_sample)
{
	IlluminationAwareKDTreeLearningToClusterTrainingSample learning_to_cluster_training_sample{};
	learning_to_cluster_training_sample.position					   = position;
	learning_to_cluster_training_sample.mesh_id						   = mesh_id;
	learning_to_cluster_training_sample.shading_context				   = shading_context;
	learning_to_cluster_training_sample.selected_cluster_node_index	   = cluster_sample.cluster_node_index;
	learning_to_cluster_training_sample.emissive_triangle_global_index = cluster_sample.emissive_triangle_global_index;
	learning_to_cluster_training_sample.cluster_probability			   = cluster_sample.cluster_probability;
	learning_to_cluster_training_sample.sampled_lightcut_index		   = cluster_sample.lightcut_index;
	learning_to_cluster_training_sample.selected_lightcut_slot		   = cluster_sample.lightcut_slot;
	learning_to_cluster_training_sample.sampled_lightcut_size		   = cluster_sample.lightcut_size_at_sampling;
	learning_to_cluster_training_sample.valid_for_lightcut			   = true;

	return learning_to_cluster_training_sample;
}

/**
 * Stores how the NEE of the shading point chose between the envmap and the emissive lights, for the MIS weight of the BSDF sample of that
 * shading point. Only the estimator with MIS reads it
 */
HIPRT_DEVICE void set_learning_to_cluster_light_selection(NEEDeferredMISContext& out_nee_mis_context,
														  float envmap_selection_probability,
														  float emissive_selection_probability,
														  bool exclude_envmap_slot)
{
#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	out_nee_mis_context.set_last_light_selection(envmap_selection_probability, emissive_selection_probability, exclude_envmap_slot);
#endif
}

template <bool use_MIS, NEEShadowMode shadow_mode = NEE_SHADOW_INLINE, typename BsdfMaterialType, typename ProposalMaterialType>
HIPRT_DEVICE ColorRGB32F sample_one_light_SG_tree_learning_to_cluster(HIPRTRenderData& render_data,
																	  RayPayloadT<BsdfMaterialType>& ray_payload,
																	  BsdfMaterialType& bsdf_material,
																	  const HitInfo closest_hit_info,
																	  const float3_t& view_direction,
																	  const ProposalMaterialType& proposal_state,
																	  Xorshift32Generator& random_number_generator,
																	  int2_t pixel_coords,
																	  NEEDeferredMISContext& out_nee_mis_context,
																	  WavefrontShadowSink* shadow_sink = nullptr)
{
	// A shading point that samples no light leaves the BSDF sample as the only sampler of the envmap
	set_learning_to_cluster_light_selection(out_nee_mis_context, 0.0f, 1.0f, false);

	if (!bsdf_material.can_do_light_sampling())
		return ColorRGB32F(0.0f);

	IlluminationAwareKDTreeSGShadingContext shading_context = build_light_clustering_shading_context(
		closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, static_cast<const LightTreeSGProposalState&>(proposal_state));

	unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
	if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && closest_hit_info.primitive_index >= 0)
		mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[closest_hit_info.primitive_index];

	unsigned int guiding_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	unsigned int lightcut_index		= render_data.kd_tree_device.resolve_lightcut(shading_context, mesh_id, &guiding_node_index);

	unsigned int pixel_index					   = pixel_coords.x + pixel_coords.y * render_data.render_settings.render_resolution.x;
	LearningToClusterEnvmapNEEMode envmap_nee_mode = choose_learning_to_cluster_envmap_nee_mode(render_data, lightcut_index, ray_payload.bounce, pixel_index);

	// Only the NEE samples of the primary hit observe the per-pixel statistics
	LearningToClusterPixelObservationTarget pixel_observation_target;

	if (envmap_nee_mode == LEARNING_TO_CLUSTER_ENVMAP_NEE_LIGHTCUT_SLOT)
	{
#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
		set_learning_to_cluster_light_selection(out_nee_mis_context, pdf_of_envmap_cluster_learning_to_cluster(render_data, shading_context, lightcut_index),
												1.0f, false);
#endif

		IlluminationAwareKDTreeLearningToClusterCutTriangleSample triangle_sample =
			sample_one_emissive_triangle_learning_to_cluster(render_data, shading_context, lightcut_index, guiding_node_index, random_number_generator);
		if (!triangle_sample.valid())
		{
			append_failed_light_clustering_training_sample(render_data, closest_hit_info.inter_point, shading_context, mesh_id, triangle_sample);

			return ColorRGB32F(0.0f);
		}

		IlluminationAwareKDTreeLearningToClusterTrainingSample learning_to_cluster_training_sample =
			make_learning_to_cluster_training_sample(closest_hit_info.inter_point, mesh_id, shading_context, triangle_sample);

		if (triangle_sample.is_envmap())
			return sample_envmap_cluster_learning_to_cluster<use_MIS, shadow_mode>(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction,
																				   triangle_sample, learning_to_cluster_training_sample,
																				   pixel_observation_target, random_number_generator, shadow_sink);

		return sample_emissive_cluster_learning_to_cluster<use_MIS, shadow_mode>(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction,
																				 proposal_state, triangle_sample, learning_to_cluster_training_sample,
																				 pixel_observation_target, random_number_generator, shadow_sink);
	}

	// The envmap is sampled on its own and the emissive lights from the lightcut without its envmap slot: both with their own NEE sample,
	// or only one of them, chosen with the probability learned at the pixel
	float envmap_selection_probability	 = 1.0f;
	float emissive_selection_probability = 1.0f;
	bool sample_envmap					 = true;
	bool sample_emissive_lights			 = true;

	if (envmap_nee_mode == LEARNING_TO_CLUSTER_ENVMAP_NEE_PIXEL_PROBABILITY)
	{
		envmap_selection_probability   = learning_to_cluster_pixel_envmap_probability(render_data, pixel_index);
		emissive_selection_probability = 1.0f - envmap_selection_probability;

		sample_envmap		   = random_number_generator() < envmap_selection_probability;
		sample_emissive_lights = !sample_envmap;
	}

	set_learning_to_cluster_light_selection(out_nee_mis_context, envmap_selection_probability, emissive_selection_probability, true);

	pixel_observation_target.pixel_index = pixel_index;
	pixel_observation_target.record		 = learning_to_cluster_uses_pixel_envmap_statistics(render_data, ray_payload.bounce);

	// Counted here, before the visibility of the samples is known: an occluded or failed sample is an observation of 0
	if (pixel_observation_target.record)
		render_data.aux_buffers.learning_to_cluster_pixel_envmap_statistics[pixel_index].observation_count++;

	ColorRGB32F light_source_radiance(0.0f);

	if (sample_envmap)
	{
		IlluminationAwareKDTreeLearningToClusterCutTriangleSample envmap_sample{};
		envmap_sample.guiding_node_index			   = guiding_node_index;
		envmap_sample.cluster_node_index			   = IlluminationAwareKDTreeNode::ENVMAP_CLUSTER_NODE_INDEX;
		envmap_sample.cluster_probability			   = envmap_selection_probability;
		envmap_sample.conditional_triangle_probability = 1.0f;

		// The envmap sample observes the envmap slot of the lightcut learned at this shading point, like an envmap sample selected from that
		// lightcut: the learning of the envmap slot does not depend on how the envmap was selected
		IlluminationAwareKDTreeLearningToClusterTrainingSample envmap_training_sample =
			make_learning_to_cluster_training_sample(closest_hit_info.inter_point, mesh_id, shading_context, envmap_sample);

		light_source_radiance += sample_envmap_cluster_learning_to_cluster<use_MIS, shadow_mode>(
			render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, envmap_sample, envmap_training_sample, pixel_observation_target,
			random_number_generator, shadow_sink);
	}

	if (sample_emissive_lights)
	{
		IlluminationAwareKDTreeLearningToClusterCutTriangleSample triangle_sample = sample_one_emissive_triangle_learning_to_cluster(
			render_data, shading_context, lightcut_index, guiding_node_index, random_number_generator, /* exclude_envmap_slot */ true);
		triangle_sample.cluster_probability *= emissive_selection_probability;

		if (!triangle_sample.valid())
			append_failed_light_clustering_training_sample(render_data, closest_hit_info.inter_point, shading_context, mesh_id, triangle_sample);
		else
		{
			IlluminationAwareKDTreeLearningToClusterTrainingSample emissive_training_sample =
				make_learning_to_cluster_training_sample(closest_hit_info.inter_point, mesh_id, shading_context, triangle_sample);

			light_source_radiance += sample_emissive_cluster_learning_to_cluster<use_MIS, shadow_mode>(
				render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, proposal_state, triangle_sample, emissive_training_sample,
				pixel_observation_target, random_number_generator, shadow_sink);
		}
	}

	return light_source_radiance;
}

template <typename BsdfMaterialType, typename ProposalMaterialType, NEEShadowMode shadow_mode = NEE_SHADOW_INLINE>
HIPRT_DEVICE ColorRGB32F sample_one_light_no_MIS_SG_tree_learning_to_cluster(HIPRTRenderData& render_data,
																			 RayPayloadT<BsdfMaterialType>& ray_payload,
																			 BsdfMaterialType& bsdf_material,
																			 const HitInfo closest_hit_info,
																			 const float3_t& view_direction,
																			 const ProposalMaterialType& proposal_state,
																			 Xorshift32Generator& random_number_generator,
																			 int2_t pixel_coords,
																			 NEEDeferredMISContext& out_nee_mis_context,
																			 WavefrontShadowSink* shadow_sink = nullptr)
{
	return sample_one_light_SG_tree_learning_to_cluster<false, shadow_mode>(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction,
																			proposal_state, random_number_generator, pixel_coords, out_nee_mis_context,
																			shadow_sink);
}

template <typename BsdfMaterialType, typename ProposalMaterialType, NEEShadowMode shadow_mode = NEE_SHADOW_INLINE>
HIPRT_DEVICE ColorRGB32F sample_one_light_MIS_SG_tree_learning_to_cluster(HIPRTRenderData& render_data,
																		  RayPayloadT<BsdfMaterialType>& ray_payload,
																		  BsdfMaterialType& bsdf_material,
																		  const HitInfo closest_hit_info,
																		  const float3_t& view_direction,
																		  const ProposalMaterialType& proposal_state,
																		  Xorshift32Generator& random_number_generator,
																		  int2_t pixel_coords,
																		  NEEDeferredMISContext& out_nee_mis_context,
																		  WavefrontShadowSink* shadow_sink = nullptr)
{
	return sample_one_light_SG_tree_learning_to_cluster<true, shadow_mode>(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction,
																		   proposal_state, random_number_generator, pixel_coords, out_nee_mis_context,
																		   shadow_sink);
}

template <NEEShadowMode shadow_mode = NEE_SHADOW_INLINE, typename BsdfMaterialType, typename ProposalMaterialType>
HIPRT_DEVICE ColorRGB32F sample_multiple_emissive_geometry(HIPRTRenderData& render_data,
														   RayPayloadT<BsdfMaterialType>& ray_payload,
														   BsdfMaterialType& bsdf_material,
														   const HitInfo closest_hit_info,
														   const float3_t& view_direction,
														   int2_t pixel_coords,
														   NEEDeferredMISContext& out_nee_mis_context,
														   const ProposalMaterialType& proposal_state,
														   Xorshift32Generator& random_number_generator,
														   WavefrontShadowSink* shadow_sink = nullptr)
{
	ColorRGB32F direct_light_contribution;

	// Any of these light sampling strategy support sampling multiple lights
	// per each shading point, effectively "amortizing" camera and bounce rays
#if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_one_light_no_MIS<BsdfMaterialType, ProposalMaterialType, shadow_mode>(
		render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, proposal_state, random_number_generator, shadow_sink);
#elif DirectLightNEEEstimator == LSS_BSDF
	// No light sampling here: with LSS_BSDF, lights are only reached by the BSDF sample of the main path's next bounce
	// direct_light_contribution += sample_one_light_bsdf(render_data, ray_payload, closest_hit_info, view_direction, random_number_generator);
#elif DirectLightNEEEstimator == LSS_MIS_LIGHT_BSDF			 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_one_light_MIS_deferred_BSDF<BsdfMaterialType, ProposalMaterialType, shadow_mode>(
		render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, proposal_state, random_number_generator, shadow_sink);
#elif DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT		 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	RISReservoir reservoir = sample_lights_RIS_for_deferred_NEE_BSDF_MIS(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction,
																		 proposal_state, random_number_generator);

	out_nee_mis_context.fill_ris_reservoir(reservoir);
#elif DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER	 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_one_light_no_MIS_SG_tree_learning_to_cluster<BsdfMaterialType, ProposalMaterialType, shadow_mode>(
		render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, proposal_state, random_number_generator, pixel_coords, out_nee_mis_context,
		shadow_sink);
#elif DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS // #if DirectLightNEEEstimator == LSS_ONE_LIGHT
	direct_light_contribution = sample_one_light_MIS_SG_tree_learning_to_cluster<BsdfMaterialType, ProposalMaterialType, shadow_mode>(
		render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, proposal_state, random_number_generator, pixel_coords, out_nee_mis_context,
		shadow_sink);
#endif														 // #if DirectLightNEEEstimator == LSS_ONE_LIGHT

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
template <NEEShadowMode shadow_mode = NEE_SHADOW_INLINE, typename BsdfMaterialType, typename ProposalMaterialType>
HIPRT_DEVICE ColorRGB32F sample_emissive_geometry(HIPRTRenderData& render_data,
												  RayPayloadT<BsdfMaterialType>& ray_payload,
												  BsdfMaterialType& bsdf_material,
												  const HitInfo closest_hit_info,
												  const float3_t& view_direction,
												  int2_t pixel_coords,
												  NEEDeferredMISContext& out_nee_mis_context,
												  const ProposalMaterialType& proposal_state,
												  Xorshift32Generator& random_number_generator,
												  WavefrontShadowSink* shadow_sink = nullptr)
{
#if DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)
	// The envmap can be the only cluster of the lightcuts of learning to cluster, without any emissive geometry
	if (!learning_to_cluster_has_light_to_sample(render_data))
		return ColorRGB32F(0.0f);
#else
	if (render_data.buffers.emissive_triangles_count == 0)
		// No emissive geometry in the scene to sample
		// so return black
		return ColorRGB32F(0.0f);
#endif // #if DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)

	if (render_data.bsdfs_data.white_furnace_mode && render_data.bsdfs_data.white_furnace_mode_turn_off_emissives)
		return ColorRGB32F(0.0f);

	ColorRGB32F direct_light_contribution;
#if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING
	direct_light_contribution = ColorRGB32F(0.0f);
#else  // A light sampling strategy is used

	// A light sampling strategy can sample more than one light per path vertex.
	direct_light_contribution =
		sample_multiple_emissive_geometry<shadow_mode>(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, pixel_coords,
													   out_nee_mis_context, proposal_state, random_number_generator, shadow_sink);
#endif // #if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING

	return direct_light_contribution;
}

/**
 * The x & y parameters are retained for the renderer's direct-lighting interface.
 */
// Forced inline: see the comment above principled_bsdf_eval()
template <NEEShadowMode shadow_mode = NEE_SHADOW_INLINE, typename BsdfMaterialType>
HIPRT_DEVICE HIPRT_INLINE ColorRGB32F estimate_direct_lighting_from_emissive_contribution(HIPRTRenderData& render_data,
																						  RayPayloadT<BsdfMaterialType>& ray_payload,
																						  BsdfMaterialType& bsdf_material,
																						  ColorRGB32F ray_throughput,
																						  ColorRGB32F emissive_geometry_direct_contribution,
																						  HitInfo& closest_hit_info,
																						  float3_t view_direction,
																						  Xorshift32Generator& random_number_generator,
																						  WavefrontShadowSink* shadow_sink = nullptr)
{
	ColorRGB32F total_direct_lighting;

	ColorRGB32F envmap_direct_contribution;
#if DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)
	// When the envmap is a cluster of the lightcuts, it is sampled by the emissive geometry NEE of learning to cluster
	if (!render_data.kd_tree_device.learning_to_cluster.envmap_cluster_enabled)
#endif
		envmap_direct_contribution = sample_environment_map<BsdfMaterialType, shadow_mode>(render_data, ray_payload, bsdf_material, closest_hit_info,
																						   view_direction, random_number_generator, shadow_sink);

#if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING // No direct light sampling
	// This if() rejects backfacing lights if backfacing lights are disabled
	if (compute_cosine_term_at_light_source(closest_hit_info.original_geometric_normal(), view_direction) > 0.0f)
	{
		ColorRGB32F hit_emission = bsdf_material.get_emission();

		if (render_data.render_settings.enable_direct_lighting || ray_payload.bounce > 1)
			total_direct_lighting += hit_emission * ray_throughput;
	}
#else  // #if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING
	if (ray_payload.bounce == 0 && compute_cosine_term_at_light_source(closest_hit_info.original_geometric_normal(), view_direction) > 0.0f)
		// If we do have emissive geometry sampling, we only want to take
		// it into account on the first bounce, otherwise we would be
		// accounting for direct light sampling twice (bounce on emissive
		// geometry + direct light sampling). Otherwise, we don't check for bounce == 0
		total_direct_lighting += bsdf_material.get_emission();

	// Direct lighting from light sampling
	ColorRGB32F direct_lighting_contribution = (emissive_geometry_direct_contribution + envmap_direct_contribution) * ray_throughput;

	total_direct_lighting += direct_lighting_contribution;
#endif // #if DirectLightNEEEstimator == LSS_NO_DIRECT_LIGHT_SAMPLING

	return total_direct_lighting;
}

// Forced inline: see the comment above principled_bsdf_eval()
HIPRT_DEVICE HIPRT_INLINE ColorRGB32F estimate_direct_lighting_from_emissive_contribution(HIPRTRenderData& render_data,
																						  RayPayload& ray_payload,
																						  ColorRGB32F ray_throughput,
																						  ColorRGB32F emissive_geometry_direct_contribution,
																						  HitInfo& closest_hit_info,
																						  float3_t view_direction,
																						  Xorshift32Generator& random_number_generator)
{
	return estimate_direct_lighting_from_emissive_contribution(render_data, ray_payload, ray_payload.material, ray_throughput,
															   emissive_geometry_direct_contribution, closest_hit_info, view_direction,
															   random_number_generator);
}

template <NEEShadowMode shadow_mode = NEE_SHADOW_INLINE, typename BsdfMaterialType, typename ProposalMaterialType>
HIPRT_DEVICE ColorRGB32F estimate_direct_lighting(HIPRTRenderData& render_data,
												  RayPayloadT<BsdfMaterialType>& ray_payload,
												  BsdfMaterialType& bsdf_material,
												  ColorRGB32F ray_throughput,
												  HitInfo& closest_hit_info,
												  float3_t view_direction,
												  int x,
												  int y,
												  NEEDeferredMISContext& out_nee_mis_context,
												  const ProposalMaterialType& proposal_state,
												  Xorshift32Generator& random_number_generator,
												  WavefrontShadowSink* shadow_sink = nullptr)
{
	ColorRGB32F emissive_geometry_direct_contribution =
		sample_emissive_geometry<shadow_mode>(render_data, ray_payload, bsdf_material, closest_hit_info, view_direction, make_int2(x, y), out_nee_mis_context,
											  proposal_state, random_number_generator, shadow_sink);

	return estimate_direct_lighting_from_emissive_contribution<shadow_mode>(render_data, ray_payload, bsdf_material, ray_throughput,
																			emissive_geometry_direct_contribution, closest_hit_info, view_direction,
																			random_number_generator, shadow_sink);
}

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
	return estimate_direct_lighting(render_data, ray_payload, ray_payload.material, ray_throughput, closest_hit_info, view_direction, x, y, out_nee_mis_context,
									make_light_proposal_state<DirectLightSamplingStrategy, TrianglePointSamplingStrategy, BSDFModel::Principled>(
										make_light_proposal_inputs(ray_payload.material)),
									random_number_generator);
}

HIPRT_DEVICE RISReservoir deferred_NEE_MIS_add_one_RIS_BSDF_sample(HIPRTRenderData& render_data,
																   bool intersection_found,
																   HitInfo& main_path_ray_hit_info,
																   RayPayloadCommon& ray_payload,
																   const EffectiveMaterialEmission& current_hit_emission,
																   NEEDeferredMISContext& nee_deferred_MIS_context,
																   RISReservoir& reservoir,
																   Xorshift32Generator& random_number_generator,
																   WavefrontShadowSink* shadow_sink = nullptr)
{
#if !PathSamplingStrategyIsReSTIRPT

#if DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT
	int nb_light_candidates = render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.ris_settings.number_of_light_candidates;
	int nb_bsdf_candidates	= render_data.render_settings.do_render_low_resolution() ? 1 : render_data.render_settings.ris_settings.number_of_bsdf_candidates;

	if (!current_hit_emission.is_emissive() || !intersection_found || nb_bsdf_candidates == 0)
	{
		reservoir.end();

		return reservoir;
	}

	float bsdf_sample_pdf								= nee_deferred_MIS_context.last_bsdf_sample_pdf;
	const NEEDeferredLightProposalState& proposal_state = nee_deferred_MIS_context.get_last_light_proposal_state();

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
			ColorRGB32F hit_emission	   = current_hit_emission.get_emission();
			ColorRGB32F light_contribution = nee_deferred_MIS_context.last_bsdf_x_cos_theta * hit_emission;
			float target_function		   = light_contribution.luminance();

			float light_pdf =
				pdf_of_emissive_triangle_hit_solid_angle(render_data, nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction,
														 nee_deferred_MIS_context.last_shading_normal, proposal_state, main_path_ray_hit_info.primitive_index,
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

#endif // !PathSamplingStrategyIsReSTIRPT // #if !PathSamplingStrategyIsReSTIRPT

	return RISReservoir();
}

#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS && !PathSamplingStrategyIsReSTIRPT
/**
 * When the envmap is a cluster of the lightcuts, the bounce ray of the main path that misses the scene is the BSDF sample of the envmap: returns
 * the envmap contribution of that ray with its MIS weight against the envmap cluster of the lightcut of the previous vertex
 */
HIPRT_DEVICE ColorRGB32F deferred_NEE_MIS_envmap_miss_learning_to_cluster(HIPRTRenderData& render_data,
																		  const RayPayloadCommon& ray_payload,
																		  const NEEDeferredMISContext& nee_deferred_MIS_context)
{
	const float3_t& bsdf_sampled_direction = nee_deferred_MIS_context.last_bsdf_sampled_direction;

	float envmap_pdf_solid_angle;
	ColorRGB32F envmap_radiance = envmap_eval(render_data, bsdf_sampled_direction, envmap_pdf_solid_angle);

	float light_sampler_solid_angle_pdf = 0.0f;

	// The NEE of the previous vertex stored the probability with which it sampled the envmap: 1 when the envmap had its own NEE sample, the
	// probability learned at the pixel or the probability of the envmap slot of the lightcut otherwise
	if (nee_deferred_MIS_context.last_can_do_light_sampling)
		light_sampler_solid_angle_pdf = nee_deferred_MIS_context.last_envmap_selection_probability * envmap_pdf_solid_angle;

	// A previous vertex that cannot do light sampling sampled no envmap direction: the BSDF ray is then the only sampler of the envmap and its
	// weight is 1
	float bsdf_sample_mis_weight = balance_heuristic(nee_deferred_MIS_context.last_bsdf_sample_pdf, 1, light_sampler_solid_angle_pdf, 1);

	ColorRGB32F envmap_contribution =
		envmap_radiance * nee_deferred_MIS_context.last_bsdf_x_cos_theta / nee_deferred_MIS_context.last_bsdf_sample_pdf * bsdf_sample_mis_weight;

	return nee_deferred_MIS_context.last_ray_throughput * envmap_contribution;
}
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS && !PathSamplingStrategyIsReSTIRPT

/**
 * If the bounce ray of the main path hits an emissive light, computes the MIS weight for that emissive hit against the light sampler of the last hit and
 * returns the contribution of that emissive hit with that MIS weight.
 */
template <typename PreviousBSDFMaterialType, NEEShadowMode shadow_mode = NEE_SHADOW_INLINE>
[[nodiscard]] HIPRT_DEVICE ColorRGB32F do_deferred_NEE_MIS(HIPRTRenderData& render_data,
														   bool intersection_found,
														   RayPayloadCommon& ray_payload,
														   HitInfo& light_hit_info,
														   const EffectiveMaterialEmission& current_hit_emission,
														   NEEDeferredMISContext& nee_deferred_MIS_context,
														   const ResolvedMaterialUserControlsCache& previous_resolved_user_controls,
														   Xorshift32Generator& random_number_generator,
														   WavefrontShadowSink* shadow_sink = nullptr)
{
#if !PathSamplingStrategyIsReSTIRPT

#if !DirectLightNEEEstimatorHasBSDFSampling
	return ColorRGB32F(0.0f);
#else
	if (ray_payload.bounce == 0 && !render_data.render_settings.enable_direct_lighting)
		// Deferred NEE MIS for the primary hit but we're not doing direct lighting
		return ColorRGB32F(0.0f);
#if DirectLightNEEEstimator != LSS_RIS_BSDF_AND_LIGHT
	// RIS can still evaluate its stored light candidates when no valid BSDF candidate was sampled.
	else if (nee_deferred_MIS_context.last_bsdf_sample_pdf <= 0.0f)
		return ColorRGB32F(0.0f);
#endif // #if DirectLightNEEEstimator != LSS_RIS_BSDF_AND_LIGHT

#if DirectLightNEEEstimator == LSS_BSDF
	if (!current_hit_emission.is_emissive() || !intersection_found)
		return ColorRGB32F(0.0f);
	else if (compute_cosine_term_at_light_source(light_hit_info.original_geometric_normal(),
												 hippt::normalize(nee_deferred_MIS_context.last_shading_point - light_hit_info.inter_point)) <= 0.0f)
		// If the light is backfacing and backfacing lights are disabled, then we don't want to add its contribution
		return ColorRGB32F(0.0f);

	float bsdf_sample_mis_weight = 1.0f;

	return nee_deferred_MIS_context.last_ray_throughput * current_hit_emission.get_emission() * nee_deferred_MIS_context.last_bsdf_x_cos_theta /
		   nee_deferred_MIS_context.last_bsdf_sample_pdf * bsdf_sample_mis_weight;
#elif DirectLightNEEEstimator == LSS_MIS_LIGHT_BSDF || DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS // #if DirectLightNEEEstimator == LSS_BSDF
#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	if (!intersection_found && render_data.kd_tree_device.learning_to_cluster.envmap_cluster_enabled)
		return deferred_NEE_MIS_envmap_miss_learning_to_cluster(render_data, ray_payload, nee_deferred_MIS_context);
#endif

	if (!current_hit_emission.is_emissive() || !intersection_found)
		return ColorRGB32F(0.0f);
	else if (compute_cosine_term_at_light_source(light_hit_info.original_geometric_normal(),
												 hippt::normalize(nee_deferred_MIS_context.last_shading_point - light_hit_info.inter_point)) <= 0.0f)
		// If the light is backfacing and backfacing lights are disabled, then we don't want to add its contribution
		return ColorRGB32F(0.0f);

	ColorRGB32F hit_emission							= current_hit_emission.get_emission();
	const NEEDeferredLightProposalState& proposal_state = nee_deferred_MIS_context.get_last_light_proposal_state();

	float3_t ray_direction = light_hit_info.inter_point - nee_deferred_MIS_context.last_shading_point;
	float hit_distance	   = hippt::length(ray_direction);
	ray_direction /= hit_distance;

#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	float light_sampler_solid_angle_pdf = 0.0f;

	if (nee_deferred_MIS_context.last_can_do_light_sampling)
	{
		IlluminationAwareKDTreeSGShadingContext shading_context =
			build_light_clustering_shading_context(nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction,
												   nee_deferred_MIS_context.last_shading_normal, static_cast<const LightTreeSGProposalState&>(proposal_state));

		unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
		if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && nee_deferred_MIS_context.last_primitive_index >= 0)
			mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[nee_deferred_MIS_context.last_primitive_index];

		// Learning updates run after the megakernel, so this resolves the same cut and CDF used at the previous vertex.
		unsigned int lightcut_index	  = render_data.kd_tree_device.resolve_lightcut(shading_context, mesh_id);
		light_sampler_solid_angle_pdf = pdf_of_emissive_triangle_hit_solid_angle_learning_to_cluster(
			render_data, shading_context, lightcut_index, nee_deferred_MIS_context.last_can_do_light_sampling, proposal_state, light_hit_info.primitive_index,
			light_hit_info.inter_point, light_hit_info.original_geometric_normal(), nee_deferred_MIS_context.last_exclude_envmap_slot);
		// The emissive lights may have been one of two choices of the NEE of the previous vertex, the other one being the envmap
		light_sampler_solid_angle_pdf *= nee_deferred_MIS_context.last_emissive_selection_probability;
	}

	float bsdf_sample_mis_weight = balance_heuristic(nee_deferred_MIS_context.last_bsdf_sample_pdf, 1, light_sampler_solid_angle_pdf, 1);
#else													// #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
	float light_sampler_solid_angle_pdf = pdf_of_emissive_triangle_hit_solid_angle(
		render_data, nee_deferred_MIS_context.last_shading_point, nee_deferred_MIS_context.last_view_direction, nee_deferred_MIS_context.last_shading_normal,
		proposal_state, light_hit_info.primitive_index, light_hit_info.original_geometric_normal(), hit_distance, ray_direction);

	float bsdf_sample_mis_weight = balance_heuristic(nee_deferred_MIS_context.last_bsdf_sample_pdf, 1, light_sampler_solid_angle_pdf,
													 DirectLightIntegrationFactor<DirectLightSamplingStrategy>());
#endif													// #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS

	return nee_deferred_MIS_context.last_ray_throughput * hit_emission * nee_deferred_MIS_context.last_bsdf_x_cos_theta /
		   nee_deferred_MIS_context.last_bsdf_sample_pdf * bsdf_sample_mis_weight;
#elif DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT // #if DirectLightNEEEstimator == LSS_BSDF
	RISReservoir final_reservoir =
		deferred_NEE_MIS_add_one_RIS_BSDF_sample(render_data, intersection_found, light_hit_info, ray_payload, current_hit_emission, nee_deferred_MIS_context,
												 nee_deferred_MIS_context.ris_reservoir, random_number_generator);
	if (final_reservoir.UCW == 0.0f)
		return ColorRGB32F(0.0f);

	HitInfo last_hit_info;
	last_hit_info.inter_point	   = nee_deferred_MIS_context.last_shading_point;
	last_hit_info.geometric_normal = nee_deferred_MIS_context.last_geometric_normal;
	last_hit_info.shading_normal   = nee_deferred_MIS_context.last_shading_normal;
	last_hit_info.primitive_index  = nee_deferred_MIS_context.last_primitive_index;

	PreviousBSDFMaterialType last_hit_material;
	load_deferred_material_reference(render_data, nee_deferred_MIS_context, previous_resolved_user_controls, last_hit_material);

	RayPayloadT<PreviousBSDFMaterialType> last_hit_payload(NoInitTag{});
	static_cast<RayPayloadCommon&>(last_hit_payload) = ray_payload;
	last_hit_payload.bounce							 = ray_payload.bounce;
	last_hit_payload.material						 = last_hit_material;
	last_hit_payload.volume_state					 = nee_deferred_MIS_context.last_volume_state;

	ColorRGB32F last_hit_NEE_estimate = evaluate_RIS_reservoir_sample<PreviousBSDFMaterialType, shadow_mode>(
		render_data, last_hit_payload, last_hit_payload.material, last_hit_info, nee_deferred_MIS_context.last_view_direction, final_reservoir,
		random_number_generator, shadow_sink, WAVEFRONT_SHADOW_FINAL_COLOR, nee_deferred_MIS_context.last_ray_throughput);

	nee_deferred_MIS_context.ris_reservoir = RISReservoir();
	return last_hit_NEE_estimate * nee_deferred_MIS_context.last_ray_throughput;
#endif													// #if DirectLightNEEEstimator == LSS_BSDF

#endif // !PathSamplingStrategyIsReSTIRPT // #if !DirectLightNEEEstimatorHasBSDFSampling
#endif // DirectLightNEEEstimatorHasBSDFSampling // #if !PathSamplingStrategyIsReSTIRPT

	return ColorRGB32F();
}

[[nodiscard]] HIPRT_DEVICE ColorRGB32F do_deferred_NEE_MIS(HIPRTRenderData& render_data,
														   bool intersection_found,
														   RayPayloadCommon& ray_payload,
														   HitInfo& light_hit_info,
														   const EffectiveMaterialEmission& current_hit_emission,
														   NEEDeferredMISContext& nee_deferred_MIS_context,
														   Xorshift32Generator& random_number_generator)
{
	ResolvedMaterialUserControlsCache previous_resolved_user_controls;
	previous_resolved_user_controls.validity_mask = 0;
	return do_deferred_NEE_MIS<DeviceUnpackedPrincipledFullMaterial>(render_data, intersection_found, ray_payload, light_hit_info, current_hit_emission,
																	 nee_deferred_MIS_context, previous_resolved_user_controls, random_number_generator);
}

[[nodiscard]] HIPRT_DEVICE ColorRGB32F do_deferred_NEE_MIS(HIPRTRenderData& render_data,
														   bool intersection_found,
														   RayPayload& ray_payload,
														   HitInfo& light_hit_info,
														   NEEDeferredMISContext& nee_deferred_MIS_context,
														   Xorshift32Generator& random_number_generator)
{
	EffectiveMaterialEmission current_hit_emission;
	current_hit_emission.emission		= ray_payload.material.get_emission();
	current_hit_emission.emission_flags = ray_payload.material.is_emissive() ? EffectiveMaterialEmissionIsEmissive : 0u;
	return do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, light_hit_info, current_hit_emission, nee_deferred_MIS_context,
							   random_number_generator);
}

[[nodiscard]] HIPRT_DEVICE ColorRGB32F do_last_deferred_NEE_MIS(HIPRTRenderData& render_data,
																hiprtRay ray,
																RayPayloadCommon& ray_payload,
																HitInfo& closest_hit_info,
																Xorshift32Generator& random_number_generator,
																NEEDeferredMISContext& nee_deferred_MIS_context)
{
#if !PathSamplingStrategyIsReSTIRPT
	if (ray_payload.next_ray_state == RayState::MISSED)
		// The miss path has already gathered the envmap contribution, so there is no valid ray to retrace for deferred NEE MIS.
		return ColorRGB32F(0.0f);

#if DirectLightNEEEstimatorHasBSDFSampling
	// We will have one more bounce than necessary when getting here and this can throw off the 'max bounce' of alpha testing so we need to substract one bounce
	// here
	ray_payload.bounce--;
	bool intersection_found = path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, random_number_generator);
	// And add it back before deferred NEE MIS so that the code inside deferred NEE MIS receives the bounce index that it expects
	ray_payload.bounce++;

	EffectiveMaterialEmission current_hit_emission;

	if (intersection_found)
	{
		int material_index = render_data.buffers.material_indices[closest_hit_info.primitive_index];
		load_effective_emission(render_data, material_index, closest_hit_info.texcoords, current_hit_emission);
	}

	return do_deferred_NEE_MIS(render_data, intersection_found, ray_payload, closest_hit_info, current_hit_emission, nee_deferred_MIS_context,
							   random_number_generator);
#endif // #if DirectLightNEEEstimatorHasBSDFSampling

	return ColorRGB32F(0.0f);

#endif // !PathSamplingStrategyIsReSTIRPT // #if !PathSamplingStrategyIsReSTIRPT

	return ColorRGB32F();
}

#endif // #ifndef DEVICE_NEE_ESTIMATORS_H
