/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_PT_INITIAL_CANDIDATES_H
#define KERNELS_RESTIR_PT_INITIAL_CANDIDATES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/NEEEstimators.h"
#include "Device/includes/ReSTIR/PT/InitialCandidatesUtils.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/SPMIS/SPMISUtils.h"
#include "Device/includes/SanityCheck.h"

#include "HostDeviceCommon/Xorshift.h"

#if PathSamplingStrategyIsReSTIRPT && DirectLightNEEEstimator != LSS_LEARNING_TO_CLUSTER_MIS
#error "ReSTIR PT only supports the LSS_LEARNING_TO_CLUSTER_MIS NEE estimator."
#endif

#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
/**
 * Light candidate of learning to cluster when the selected cluster of the lightcut is the envmap: samples a direction from the envmap distribution,
 * traces the shadow ray towards the envmap, adds the candidate to the reservoir and appends the learning-to-cluster and spatial training samples
 * of that light sample.
 *
 * 'learning_to_cluster_training_sample' is already filled with the cluster selection of 'cluster_sample'
 */
HIPRT_DEVICE void ReSTIR_PT_add_learning_to_cluster_envmap_light_candidate(
	HIPRTRenderData& render_data,
	float3_t view_direction,
	RayPayload& ray_payload,
	ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
	ColorRGB32F path_weighted_throughput_after_rc_vertex,
	ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
	ReSTIRPTReservoirSample& restir_pt_initial_sample,
	HitInfo& closest_hit_info,
	Xorshift32Generator& random_number_generator,
	const IlluminationAwareKDTreeLearningToClusterCutTriangleSample& cluster_sample,
	IlluminationAwareKDTreeLearningToClusterTrainingSample& learning_to_cluster_training_sample,
	int x,
	int y)
{
	float envmap_pdf_solid_angle;
	float3_t sampled_direction;
	ColorRGB32F envmap_emission = envmap_sample(render_data.world_settings, sampled_direction, envmap_pdf_solid_angle, random_number_generator);

	float solid_angle_pdf = envmap_pdf_solid_angle * cluster_sample.cluster_probability;
	if (!(solid_angle_pdf > 0.0f))
	{
		// Keep a failed direction proposal observable to learning-to-cluster as a zero-reward observation, as for the emissive clusters
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return;
	}

	IlluminationAwareKDTreeDirectIlluminationTrainingSample spatial_training_sample{};
	spatial_training_sample.position				   = closest_hit_info.inter_point;
	spatial_training_sample.incoming_direction		   = sampled_direction;
	spatial_training_sample.valid_for_spatial_training = true;
	spatial_training_sample.cached_guiding_node_index  = cluster_sample.guiding_node_index;

	hiprtRay shadow_ray;
	shadow_ray.origin	 = closest_hit_info.inter_point;
	shadow_ray.direction = sampled_direction;

	NEEPlusPlusContext nee_plus_plus_context;
	nee_plus_plus_context.envmap		 = true;
	nee_plus_plus_context.point_on_light = sampled_direction;
	nee_plus_plus_context.shaded_point	 = closest_hit_info.inter_point;

	if (ray_payload.bounce == 0)
		restir_pt_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;

	bool shadow_ray_occluded =
		evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, 1.0e35f, closest_hit_info.primitive_index, nee_plus_plus_context, random_number_generator);
	if (!shadow_ray_occluded)
	{
		spatial_training_sample.spatial_radiance_weight = (envmap_emission / solid_angle_pdf).max_component();

		float bsdf_pdf							  = 0.0f;
		BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
		BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_direction, incident_light_info,
								 ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
								 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
		ColorRGB32F bsdf_color		= bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
		ColorRGB32F bsdf_throughput = bsdf_color * hippt::abs(hippt::dot(closest_hit_info.shading_normal, sampled_direction));

		if (bsdf_pdf > 0.0f)
		{
			// MIS weighted against the BSDF sample of the main path, the only BSDF candidate of the vertex
			float light_candidate_mis_weight = balance_heuristic(solid_angle_pdf, bsdf_pdf);

			float weight = light_candidate_mis_weight * (ray_payload.throughput * bsdf_throughput / solid_angle_pdf * envmap_emission).luminance();
			ReSTIR_PT_add_candidate(ray_payload.bounce, sampled_direction, sampled_direction, make_float3(0.0f, 0.0f, 0.0f), -1, envmap_emission,
									bsdf_throughput, BSDFIncidentLightInfo::NO_INFO, path_unweighted_throughput_up_to_rc_vertex,
									path_weighted_throughput_after_rc_vertex, weight, restir_pt_initial_reservoir, restir_pt_initial_sample,
									random_number_generator, x, y);

			ColorRGB32F learning_estimator							 = envmap_emission * bsdf_throughput / solid_angle_pdf * light_candidate_mis_weight;
			learning_to_cluster_training_sample.q_reward			 = learning_estimator.luminance() * cluster_sample.cluster_probability;
			learning_to_cluster_training_sample.variance_observation = learning_to_cluster_training_sample.q_reward;
		}
	}

	render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
	render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);
}

HIPRT_DEVICE void ReSTIR_PT_add_learning_to_cluster_light_candidate(HIPRTRenderData& render_data,
																	float3_t view_direction,
																	RayPayload& ray_payload,
																	ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
																	ColorRGB32F path_weighted_throughput_after_rc_vertex,
																	ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
																	ReSTIRPTReservoirSample& restir_pt_initial_sample,
																	HitInfo& closest_hit_info,
																	Xorshift32Generator& random_number_generator,
																	unsigned int lightcut_index,
																	unsigned int guiding_node_index,
																	int x,
																	int y)
{
	if (!ray_payload.material.can_do_light_sampling())
		return;

	IlluminationAwareKDTreeSGShadingContext shading_context =
		build_light_clustering_shading_context(closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, ray_payload.material);

	unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
	if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && closest_hit_info.primitive_index >= 0)
		mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[closest_hit_info.primitive_index];

	IlluminationAwareKDTreeLearningToClusterCutTriangleSample triangle_sample =
		sample_one_emissive_triangle_learning_to_cluster(render_data, shading_context, lightcut_index, guiding_node_index, random_number_generator);
	if (!triangle_sample.valid())
	{
		append_failed_light_clustering_training_sample(render_data, closest_hit_info.inter_point, shading_context, mesh_id, triangle_sample);

		return;
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

	if (triangle_sample.is_envmap())
	{
		ReSTIR_PT_add_learning_to_cluster_envmap_light_candidate(render_data, view_direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
																 path_weighted_throughput_after_rc_vertex, restir_pt_initial_reservoir,
																 restir_pt_initial_sample, closest_hit_info, random_number_generator, triangle_sample,
																 learning_to_cluster_training_sample, x, y);

		return;
	}

	LightSamplePointInformation light_sample =
		sample_point_on_light_and_fill_light_sample_information(render_data, closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal,
																ray_payload.material, triangle_sample.emissive_triangle_global_index, random_number_generator);
	light_sample.area_measure_pdf *= triangle_sample.triangle_probability();

	if (!(light_sample.area_measure_pdf > 0.0f))
	{
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return;
	}

	IlluminationAwareKDTreeDirectIlluminationTrainingSample spatial_training_sample{};
	spatial_training_sample.position				   = closest_hit_info.inter_point;
	spatial_training_sample.valid_for_spatial_training = true;
	spatial_training_sample.cached_guiding_node_index  = triangle_sample.guiding_node_index;

	float3_t shadow_direction = light_sample.point_on_light - closest_hit_info.inter_point;

	float distance_to_light = hippt::length(shadow_direction);
	if (!(distance_to_light > 0.0f))
	{
		render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
		render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);

		return;
	}
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

		if (ray_payload.bounce == 0)
			restir_pt_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;

		shadow_ray_occluded = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, distance_to_light, closest_hit_info.primitive_index,
																nee_plus_plus_context, random_number_generator);
		if (!shadow_ray_occluded)
		{
			float solid_angle_pdf = area_to_solid_angle_pdf(light_sample.area_measure_pdf, distance_to_light, dot_light_source);
			if (solid_angle_pdf > 0.0f)
			{
				spatial_training_sample.spatial_radiance_weight = (light_sample.emission / solid_angle_pdf).max_component();

				float bsdf_pdf							  = 0.0f;
				BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
				BSDFContext bsdf_context(view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, shadow_direction,
										 incident_light_info, ray_payload.volume_state, false, ray_payload.material, ray_payload.accumulated_roughness,
										 MicrofacetRegularization::RegularizationMode::REGULARIZATION_MIS);
				ColorRGB32F bsdf_color		= bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, random_number_generator);
				ColorRGB32F bsdf_throughput = bsdf_color * hippt::abs(hippt::dot(closest_hit_info.shading_normal, shadow_direction));

				if (bsdf_pdf > 0.0f)
				{
					// MIS weighted against the BSDF sample of the main path, the only BSDF candidate of the vertex
					float light_candidate_mis_weight = balance_heuristic(solid_angle_pdf, bsdf_pdf);

					float weight =
						light_candidate_mis_weight * (ray_payload.throughput * bsdf_throughput / solid_angle_pdf * light_sample.emission).luminance();
					ReSTIR_PT_add_candidate(ray_payload.bounce, shadow_direction, light_sample.point_on_light, light_sample.light_source_normal,
											triangle_sample.emissive_triangle_global_index, light_sample.emission, bsdf_throughput,
											BSDFIncidentLightInfo::NO_INFO, path_unweighted_throughput_up_to_rc_vertex,
											path_weighted_throughput_after_rc_vertex, weight, restir_pt_initial_reservoir, restir_pt_initial_sample,
											random_number_generator, x, y);

					ColorRGB32F learning_estimator				 = light_sample.emission * bsdf_throughput / solid_angle_pdf * light_candidate_mis_weight;
					learning_to_cluster_training_sample.q_reward = learning_estimator.luminance() * triangle_sample.cluster_probability;
					learning_to_cluster_training_sample.variance_observation = learning_to_cluster_training_sample.q_reward;
				}
			}
		}
	}

	render_data.kd_tree_device.core.append_direct_illumination_training_sample(spatial_training_sample);
	render_data.kd_tree_device.learning_to_cluster.append_learning_to_cluster_training_sample(learning_to_cluster_training_sample);
}

/**
 * Light candidate of the current vertex: one learning-to-cluster light candidate, which also samples the envmap when the envmap is a cluster of
 * the lightcuts.
 *
 * The only BSDF candidate of the vertex is the BSDF sample that continues the main path: it is MIS weighted against this light candidate when
 * the next vertex is found, by ReSTIR_PT_do_deferred_NEE_MIS()
 */
HIPRT_DEVICE void ReSTIR_PT_stream_NEE(HIPRTRenderData& render_data,
									   float3_t view_direction,
									   RayPayload& ray_payload,
									   ColorRGB32F path_unweighted_throughput_up_to_rc_vertex,
									   ColorRGB32F path_weighted_throughput_after_rc_vertex,
									   ReSTIRPTInitialCandidatesReservoir& restir_pt_initial_reservoir,
									   ReSTIRPTReservoirSample& restir_pt_initial_sample,
									   HitInfo& closest_hit_info,
									   NEEDeferredMISContext& nee_deferred_MIS_context,
									   Xorshift32Generator& random_number_generator,
									   int x,
									   int y)
{
	// The lightcut of this shading point is resolved once and used by the light candidate and by the deferred PDF of the next vertex, which are
	// both evaluated at this same shading point.
	//
	// It is resolved before the early exit for disabled direct lighting because the deferred PDF of the next vertex is evaluated even then. It
	// stays unresolved only when no PDF reads it: when the lightcuts hold no cluster (no emissive triangles and no envmap cluster) or for a
	// material that cannot do light sampling, the PDFs return before reading the lightcut
	unsigned int lightcut_index		= IlluminationAwareKDTreeNode::UNRESOLVED_LIGHTCUT_INDEX;
	unsigned int guiding_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	bool has_light_to_sample = learning_to_cluster_has_light_to_sample(render_data);
	if (has_light_to_sample && ray_payload.material.can_do_light_sampling())
	{
		IlluminationAwareKDTreeSGShadingContext shading_context =
			build_light_clustering_shading_context(closest_hit_info.inter_point, view_direction, closest_hit_info.shading_normal, ray_payload.material);

		unsigned int mesh_id = IlluminationAwareKDTreeLearningToClusterLightcutSet::INVALID_MESH_ID;
		if (render_data.buffers.global_triangle_index_to_mesh_index != nullptr && closest_hit_info.primitive_index >= 0)
			mesh_id = render_data.buffers.global_triangle_index_to_mesh_index[closest_hit_info.primitive_index];

		lightcut_index = render_data.kd_tree_device.resolve_lightcut(shading_context, mesh_id, &guiding_node_index);
	}

	// Also written when unresolved so that the deferred PDF of the next vertex never reads the lightcut of an older vertex
	nee_deferred_MIS_context.set_last_lightcut_index(lightcut_index);

	if (ray_payload.bounce == 0 && !render_data.render_settings.enable_direct_lighting)
		return;

	// When the envmap is a cluster of the lightcuts, the learning-to-cluster light candidate also samples it
	if (has_light_to_sample)
		ReSTIR_PT_add_learning_to_cluster_light_candidate(render_data, view_direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
														  path_weighted_throughput_after_rc_vertex, restir_pt_initial_reservoir, restir_pt_initial_sample,
														  closest_hit_info, random_number_generator, lightcut_index, guiding_node_index, x, y);
}
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char RESTIR_PT_RENDER_DATA[sizeof(HIPRTRenderData)];

	// LOCAL DEBUG TOGGLE (HIPRT_INITIAL_CANDIDATES_BANDS, see ReSTIRPTRenderPass::launch_initial_candidates_pass()): first row of the horizontal
	// band processed by this launch. Zero when the image is launched in one go
	__device__ unsigned int RESTIR_PT_INITIAL_CANDIDATES_BAND_Y_OFFSET;
}
GLOBAL_KERNEL_SIGNATURE(void) __attribute__((amdgpu_num_vgpr(144))) __launch_bounds__(64) ReSTIR_PT_InitialCandidates()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PT_InitialCandidates(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(RESTIR_PT_RENDER_DATA);

	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y + RESTIR_PT_INITIAL_CANDIDATES_BAND_Y_OFFSET;
#endif // #ifdef __KERNELCC__
	if (x >= render_data.render_settings.render_resolution.x || y >= render_data.render_settings.render_resolution.y)
		return;

	uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;
	if (!render_data.aux_buffers.pixel_active[pixel_index])
		// Pixel isn't active because of adaptive sampling or render resolution scaling
		return;

	int bounce_count = render_data.render_settings.nb_bounces;
	if (render_data.render_settings.do_render_low_resolution())
		// Reducing the number of bounces to 3 if rendering at low resolution
		// for better interactivity
		bounce_count = hippt::min(3, bounce_count);

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
	// Resetting splatting samples
	for (int bounce = 0; bounce < bounce_count; bounce++)
		render_data.render_settings.restir_pg_settings.invalidate_splatting_sample(render_data.render_settings.render_resolution, x, y, bounce);
#endif

	Xorshift32Generator random_number_generator(render_data.get_updated_random_seed(pixel_index));

	ReSTIRPTInitialCandidatesReservoir restir_pt_initial_reservoir(render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer,
																   pixel_index);

	for (int candidate = 0; candidate < ReSTIR_PT_InitialPathTreesCount; candidate++)
	{
		HitInfo closest_hit_info;
		closest_hit_info.inter_point	  = render_data.g_buffer.primary_hit_position[pixel_index];
		closest_hit_info.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		closest_hit_info.shading_normal	  = render_data.g_buffer.shading_normals[pixel_index].unpack();
		closest_hit_info.primitive_index  = render_data.g_buffer.first_hit_prim_index[pixel_index];

		bool intersection_found = closest_hit_info.primitive_index != -1;

		hiprtRay ray;
		ray.direction = -render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

		RayPayload ray_payload;
		ray_payload.next_ray_state = RayState::BOUNCE;
		ray_payload.material	   = render_data.g_buffer.materials[pixel_index].unpack();

		// Because this is the camera hit (and assuming the camera isn't inside volumes for now),
		// the ray volume state after the camera hit is just an empty interior stack but with
		// the material index that we hit pushed onto the stack. That's it. Because it is that
		// simple, we don't have the ray volume state in the GBuffer but rather we can
		// reconstruct the ray volume state on the fly
		ray_payload.volume_state.reconstruct_first_hit(ray_payload.material, render_data.buffers.material_indices,
													   render_data.g_buffer.first_hit_prim_index[pixel_index], random_number_generator);

		// The sample of the path lives in memory, not in registers: the reconnection vertex fields written at bounces 0 and 1 would otherwise stay live
		// across every later NEE and traced ray, which costs the kernel resident waves. Memory is only read back when a candidate is selected (copy into
		// the output reservoir)
		ReSTIRPTReservoirSample& restir_pt_initial_sample =
			render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_working_samples_buffer[pixel_index];
		restir_pt_initial_sample			 = ReSTIRPTReservoirSample();
		restir_pt_initial_sample.pixel_index = pixel_index;

		// Bounce throughput from rc_vertex to the next vertex included
		ColorRGB32F path_unweighted_throughput_up_to_rc_vertex = ColorRGB32F(1.0f);
		// BSDF * cosine / PDF of the bounces after the reconnection vertex. The unweighted BSDF * cosine of a diffuse bounce is around 0.2: its product
		// over ~45 bounces falls below the smallest normal float, the target function of the sample becomes denormal or zero and its UCW infinite.
		// Divided by the PDF, the factors are around the albedo and the product stays in range. The sample stores this suffix as its incident
		// radiance: the suffix PDF is the same in every shifted copy of the sample (the reconnection shift keeps all the vertices after the
		// reconnection vertex), so it scales the target function and the UCW of the sample by inverse factors and the contributions do not change
		ColorRGB32F path_weighted_throughput_after_rc_vertex = ColorRGB32F(1.0f);

		// Keep the clamped continuation factor separate until deferred MIS has consumed the segment's starting throughputs.
		// Unweighted for the bounces of the prefix, divided by the BSDF PDF for the bounces after the reconnection vertex
		ColorRGB32F pending_throughput = ColorRGB32F(1.0f);

		if (intersection_found)
			store_denoiser_AOVs(render_data, pixel_index, closest_hit_info.shading_normal, ray_payload.material.base_color);

		// + 1 to nb_bounces here because we want "0" bounces to still act as one
		// hit and to return some color
		NEEDeferredMISContext nee_deferred_MIS_context;

		// Bounce counter needs to be in its own variable here because HIPRTC 7.2 codegen issues otherwise, yet again...
		int bounce = ray_payload.bounce;
		for (; bounce < bounce_count + 1; bounce++)
		{
			ray_payload.bounce = bounce;

			if (ray_payload.next_ray_state != RayState::MISSED)
			{
				if (bounce > 0)
				{
					if (bounce == 1)
						restir_pt_initial_sample.visible_to_sample_point_alpha_test_random_seed = random_number_generator.m_state.seed;

					intersection_found =
						path_tracing_find_indirect_bounce_intersection(render_data, ray, ray_payload, closest_hit_info, random_number_generator);
					if (intersection_found)
					{
						// We're checking for intersection found here because otherwise this would add envmap contribution but the envmap contribution on miss
						// is already added by ReSTIR_PT_do_last_deferred_NEE_MIS
						ReSTIR_PT_do_deferred_NEE_MIS(render_data, intersection_found, ray.direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
													  path_weighted_throughput_after_rc_vertex, restir_pt_initial_reservoir, restir_pt_initial_sample,
													  closest_hit_info, nee_deferred_MIS_context, random_number_generator);

						// Hardcoded to bounce 2 for a simple reconnection shift at the first indirect vertex.
						// The pending factor belongs to bounce - 1, so the first two factors belong to the prefix.
						if (bounce > 2)
							path_weighted_throughput_after_rc_vertex *= pending_throughput;
						else
							path_unweighted_throughput_up_to_rc_vertex *= pending_throughput;
					}
				}

				if (intersection_found)
				{
					if (bounce == 1)
						ReSTIR_PT_rc_vertex_fill_information(render_data, ray_payload, closest_hit_info, restir_pt_initial_sample);

					// Compiled out for the other NEE estimators: this kernel is only built with them when ReSTIR PT is not the path sampling strategy,
					// and is then never launched
#if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS
					ReSTIR_PT_stream_NEE(render_data, -ray.direction, ray_payload, path_unweighted_throughput_up_to_rc_vertex,
										 path_weighted_throughput_after_rc_vertex, restir_pt_initial_reservoir, restir_pt_initial_sample, closest_hit_info,
										 nee_deferred_MIS_context, random_number_generator, x, y);
#endif // #if DirectLightNEEEstimator == LSS_LEARNING_TO_CLUSTER_MIS

					float bsdf_pdf;
					BSDFIncidentLightInfo incident_light_info;

					unsigned int primary_gbuffer_path_index = bounce == 0 ? pixel_index : NEE_DEFERRED_INVALID_PATH_INDEX;
					pending_throughput =
						ReSTIR_PT_compute_next_indirect_bounce(render_data, ray_payload, closest_hit_info, -ray.direction, ray, random_number_generator,
															   incident_light_info, bsdf_pdf, nee_deferred_MIS_context, primary_gbuffer_path_index);
					if (pending_throughput == ReSTIR_PT_invalid_throughput)
					{
						// Bad BSDF sample (under the surface), killed by russian roulette, ...
						bounce++;
						break;
					}

					// This factor goes into the suffix after the reconnection vertex (see the multiplication with the pending factor above)
					if (bounce >= 2)
						pending_throughput /= bsdf_pdf;

					if (bounce == 0)
						restir_pt_initial_sample.incident_light_info_at_visible_point = incident_light_info;
					else if (bounce == 1)
					{
						restir_pt_initial_sample.incident_light_info_at_sample_point = incident_light_info;
						// TODO remove this line
						restir_pt_initial_sample.rc_vertex_incident_light_direction = ray.direction;
						// Paths continuing past the reconnection vertex must store the BSDF * cosine of their continuation direction, not the value
						// left by the NEE sample of this bounce: the spatial-reuse neighbor selection target uses it in place of the reconnection
						// vertex throughput
						restir_pt_initial_sample.bsdf_throughput_luminance_at_sample_point = pending_throughput.luminance();
					}

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
					// Not the last bounce
					if (bounce != bounce_count)
					{
						ReSTIRPGSplattingSample sample;
						sample.position			  = closest_hit_info.inter_point;
						sample.normal			  = closest_hit_info.geometric_normal;
						sample.incident_direction = ray.direction;

						render_data.render_settings.restir_pg_settings.splatting_samples_soa.store_sample(sample, render_data.render_settings.render_resolution,
																										  x, y, bounce);
					}
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_TRUE
				}
				else
					break;
			}
		}

		// ReSTIR_PT_do_last_deferred_NEE_MIS() expects the bounce index one past the last traced vertex, as the loop leaves it
		ray_payload.bounce = bounce;

		ReSTIR_PT_do_last_deferred_NEE_MIS(render_data, ray, ray_payload, path_unweighted_throughput_up_to_rc_vertex, path_weighted_throughput_after_rc_vertex,
										   restir_pt_initial_reservoir, restir_pt_initial_sample, closest_hit_info, nee_deferred_MIS_context,
										   intersection_found, random_number_generator);
	}

	// The end of the kernel reads per-pixel buffers that the beginning of the kernel already read (G-buffer, random seeds). Hidden from the
	// optimizer, the pixel index makes the compiler compute these addresses again here instead of keeping the ones of the beginning of the kernel
	// in registers across the whole path, and so do the pixel coordinates recomputed from it
	pixel_index = hippt::hide_from_optimizer(pixel_index);
	int pixel_y = pixel_index / render_data.render_settings.render_resolution.x;
	int pixel_x = pixel_index - pixel_y * render_data.render_settings.render_resolution.x;

	// Computed where it is first needed: a pointer computed before the bounce loop would be kept in two registers across the whole path
	ReSTIRPTReservoir* restir_pt_initial_reservoir_output =
		&render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer[pixel_index];

	render_data.store_updated_random_seed(pixel_index, random_number_generator.m_state.seed);

	// If we got here, this means that we still have at least one ray active
	// This is a concurrent write by the way but we don't really care, everyone is writing
	// the same value
	render_data.aux_buffers.still_one_ray_active[0] = 1;

	restir_pt_initial_reservoir.end_with_normalization(1.0f, ReSTIR_PT_InitialPathTreesCount);
	restir_pt_initial_reservoir_output->sanity_check(make_int2(pixel_x, pixel_y));

	if (ReSTIR_PT_MIS_target_function_bsdf_approximation_enabled(render_data))
	{
		// Final MIS target function BSDF approximation flags of the sample (the material part was written with the reconnection vertex). The flags depend only
		// on the reconnection data of the sample, they stay valid in every copy of the sample
		unsigned char mis_target_function_bsdf_approximation_flags = 0;
		if (restir_pt_initial_reservoir_output->UCW > 0.0f)
			mis_target_function_bsdf_approximation_flags =
				ReSTIR_PT_MIS_target_function_bsdf_approximation_sample_flags(restir_pt_initial_reservoir_output->sample);

		restir_pt_initial_reservoir_output->sample.mis_target_function_bsdf_approximation_flags = mis_target_function_bsdf_approximation_flags;
	}

#if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||                                                                          \
	ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE
	float3_t pixel_first_hit_point			  = render_data.g_buffer.primary_hit_position[pixel_index];
	float3_t pixel_first_hit_geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
	ReSTIR_PT_spmis_insert_pixel_hash(render_data, pixel_x, pixel_y, pixel_first_hit_point, pixel_first_hit_geometric_normal);
#endif // #if ReSTIR_PT_MISWeightsType == RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS || ReSTIR_PT_MISWeightsType ==
	   // RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE

	if (render_data.render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_SHADE_ONLY_INITIAL_CANDIDATES &&
		ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled)
	{
		ReSTIRPTSurface initial_surface;
		initial_surface.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		initial_surface.shading_normal	 = render_data.g_buffer.shading_normals[pixel_index].unpack();
		initial_surface.primitive_index	 = render_data.g_buffer.first_hit_prim_index[pixel_index];
		initial_surface.material		 = render_data.g_buffer.materials[pixel_index].unpack();
		initial_surface.ray_volume_state.reconstruct_first_hit(initial_surface.material, render_data.buffers.material_indices, initial_surface.primitive_index,
															   random_number_generator);
		initial_surface.shading_point  = render_data.g_buffer.primary_hit_position[pixel_index];
		initial_surface.view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

		float3_t to_light_direction = restir_pt_initial_reservoir_output->sample.is_envmap_path()
										  ? restir_pt_initial_reservoir_output->sample.rc_vertex
										  : hippt::normalize(restir_pt_initial_reservoir_output->sample.rc_vertex - initial_surface.shading_point);
		BSDFContext first_hit_eval_context(initial_surface.view_direction, initial_surface.shading_normal, initial_surface.geometric_normal, to_light_direction,
										   restir_pt_initial_reservoir_output->sample.incident_light_info_at_visible_point, initial_surface.ray_volume_state,
										   false, initial_surface.material, 0.0f);

		float trash_pdf;
		ColorRGB32F bsdf_first_hit = bsdf_dispatcher_eval(render_data, first_hit_eval_context, trash_pdf, random_number_generator) *
									 hippt::abs(hippt::dot(initial_surface.shading_normal, to_light_direction));

		ColorRGB32F radiance_to_camera;
		if (restir_pt_initial_reservoir_output->sample.is_envmap_path())
			radiance_to_camera =
				bsdf_first_hit * restir_pt_initial_reservoir_output->sample.rc_vertex_incident_radiance * restir_pt_initial_reservoir_output->UCW;
		else if (!restir_pt_initial_reservoir_output->sample.di_sample)
		{
			// TODO the ray volume state should be updated here
			float3_t view_direction					 = hippt::normalize(initial_surface.shading_point - restir_pt_initial_reservoir_output->sample.rc_vertex);
			float3_t to_light_direction_sample_point = restir_pt_initial_reservoir_output->sample.rc_vertex_incident_light_direction;

			int rc_vertex_material_index = render_data.buffers.material_indices[restir_pt_initial_reservoir_output->sample.rc_vertex_primitive_index];
			DeviceUnpackedPrincipledFullMaterial rc_vertex_material =
				get_intersection_material(render_data, rc_vertex_material_index,
										  make_float2(restir_pt_initial_reservoir_output->sample.rc_vertex_texcoords_u,
													  restir_pt_initial_reservoir_output->sample.rc_vertex_texcoords_v));
			float3_t rc_vertex_shading_normal	= restir_pt_initial_reservoir_output->sample.rc_vertex_shading_normal.unpack();
			float3_t rc_vertex_geometric_normal = restir_pt_initial_reservoir_output->sample.rc_vertex_geometric_normal.unpack();
			BSDFContext secondary_hit_eval_context(view_direction, rc_vertex_shading_normal, rc_vertex_geometric_normal, to_light_direction_sample_point,
												   restir_pt_initial_reservoir_output->sample.incident_light_info_at_sample_point,
												   initial_surface.ray_volume_state, false, rc_vertex_material, 0.0f);

			ColorRGB32F bsdf_secondary_hit =
				bsdf_dispatcher_eval(render_data, secondary_hit_eval_context, trash_pdf, random_number_generator) *
				hippt::abs(hippt::dot(restir_pt_initial_reservoir_output->sample.rc_vertex_shading_normal.unpack(), to_light_direction_sample_point));
			radiance_to_camera = bsdf_first_hit * bsdf_secondary_hit * restir_pt_initial_reservoir_output->sample.rc_vertex_incident_radiance *
								 restir_pt_initial_reservoir_output->UCW;
		}
		else
			radiance_to_camera =
				bsdf_first_hit * restir_pt_initial_reservoir_output->sample.rc_vertex_incident_radiance * restir_pt_initial_reservoir_output->UCW;

		render_data.buffers.accumulated_ray_colors[pixel_index] = radiance_to_camera;
	}
}

#endif // #ifndef KERNELS_RESTIR_PT_INITIAL_CANDIDATES_H
