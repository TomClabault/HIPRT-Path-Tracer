/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_TEMPORAL_REUSE_H
#define DEVICE_RESTIR_DI_TEMPORAL_REUSE_H 

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/ReSTIR_DI_Surface.h"
#include "Device/includes/ReSTIR/ReSTIR_DI_Utils.h"
#include "Device/includes/Sampling.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

 /** References:
 *
 * [1] [Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting] https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
 * [2] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time] https://intro-to-restir.cwyman.org/
 * [3] [A Gentle Introduction to ReSTIR: Path Reuse in Real-time - SIGGRAPH 2023 Presentation Video] https://dl.acm.org/doi/10.1145/3587423.3595511#sec-supp
 * [4] [NVIDIA RTX DI SDK - Github] https://github.com/NVIDIAGameWorks/RTXDI
 * [5] [Generalized Resampled Importance Sampling Foundations of ReSTIR] https://research.nvidia.com/publication/2022-07_generalized-resampled-importance-sampling-foundations-restir
 * [6] [Uniform disk sampling] https://rh8liuqy.github.io/Uniform_Disk.html
 * [7] [Reddit Post for the Jacobian Term needed] https://www.reddit.com/r/GraphicsProgramming/comments/1eo5hqr/restir_di_light_sample_pdf_confusion/
 * [8] [Rearchitecting Spatiotemporal Resampling for Production] https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production
 * [9] [Adventures in Hybrid Rendering] https://diharaw.github.io/post/adventures_in_hybrid_rendering/
 * [10] [NVIDIA ReBLUR - Fast Denoising with Self Stabilizing Recurrent Blurs] https://developer.nvidia.com/gtc/2020/video/s22699-vid
 */

// By convention, the temporal neighbor is the first one to be resampled in for loops 
// (for looping over the neighbors when resampling / computing MIS weights)
// So instead of hardcoding 0 everywhere in the code, we just basically give it a name
// with a #define
#define TEMPORAL_NEIGHBOR_RESAMPLE 0
// Same when resampling the initial candidates
#define INITIAL_CANDIDATES_RESAMPLE 1

/**
 * Returns true if the two given points pass the plane distance check, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool plane_distance_heuristic(const float3& temporal_world_space_point, const float3& current_point, const float3& current_surface_normal, float plane_distance_threshold)
{
	float3 direction_between_points = temporal_world_space_point - current_point;
	float distance_to_plane = hippt::abs(hippt::dot(direction_between_points, current_surface_normal));

	return distance_to_plane < plane_distance_threshold;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool normal_similarity_heuristic(const float3& current_normal, const float3& neighbor_normal, float threshold)
{
	return hippt::dot(current_normal, neighbor_normal) > threshold;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool roughness_similarity_heuristic(float neighbor_roughness, float center_pixel_roughness, float threshold)
{
	// We don't want to temporally reuse on materials smoother than 0.075f because this
	// causes near-specular/glossy reflections to darken when camera ray jittering is used.
	// 
	// This glossy reflections darkening only happens with confidence weights and 
	// ray jittering but I'm not sure why. Probably because samples from one pixel (or sub-pixel location)
	// cannot efficiently be reused at another pixel (or sub-pixel location through jittering)
	// but confidence weights overweight these bad neighbor samples --> you end up using these
	// bad samples --> the shading loses in energy since we're now shading with samples that
	// don't align well with the glossy reflection direction
	return hippt::abs(neighbor_roughness - center_pixel_roughness) < threshold;
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool check_similarity_heuristics(const HIPRTRenderData& render_data, int temporal_neighbor_index, int center_pixel_index, const float3& current_shading_point, const float3& current_normal)
{
	float3 temporal_neighbor_point = render_data.g_buffer.first_hits[temporal_neighbor_index];

	float temporal_neighbor_roughness = render_data.g_buffer.materials[temporal_neighbor_index].roughness;
	float current_material_roughness = render_data.g_buffer.materials[center_pixel_index].roughness;

	bool plane_distance_passed = plane_distance_heuristic(temporal_neighbor_point, current_shading_point, current_normal, render_data.render_settings.restir_di_settings.plane_distance_threshold);
	bool normal_similarity_passed = normal_similarity_heuristic(current_normal, render_data.g_buffer.shading_normals[temporal_neighbor_index], render_data.render_settings.restir_di_settings.normal_similarity_angle_precomp);
	bool roughness_similarity_passed = roughness_similarity_heuristic(temporal_neighbor_roughness, current_material_roughness, render_data.render_settings.restir_di_settings.roughness_similarity_threshold);

	return plane_distance_passed && normal_similarity_passed && roughness_similarity_passed;
}

/**
 * Returns the linear index that can be used directly to index a buffer
 * of render_data for getting data of the temporal neighbor
 */
HIPRT_HOST_DEVICE HIPRT_INLINE int find_temporal_neighbor(const HIPRTRenderData& render_data, const float3& current_shading_point, const float3& current_normal, int2 resolution, int center_pixel_index, float center_pixel_roughness, Xorshift32Generator& random_number_generator)
{
	float3 previous_screen_space_point_xyz = matrix_X_point(render_data.prev_camera.view_projection, current_shading_point);
	float2 previous_screen_space_point = make_float2(previous_screen_space_point_xyz.x, previous_screen_space_point_xyz.y);
	// Bringing back in [0, 1] from [-1, 1]
	previous_screen_space_point += make_float2(1.0f, 1.0f);
	previous_screen_space_point *= make_float2(0.5f, 0.5f);

	float2 pixel_pos_float = make_float2(previous_screen_space_point.x * resolution.x, previous_screen_space_point.y * resolution.y);

	// Trying to find a neighbor that isn't too far away in world space in the neighborhood of the reprojected 
	// screen space position

	if (render_data.render_settings.restir_di_settings.temporal_pass.neighbor_search_strategy == 0)
	{
		float smallest_dist = 1000000.0f;
		int smallest_index = -1;

		int2 neighbor_1 = make_int2(floorf(pixel_pos_float.x), floorf(pixel_pos_float.y));
		int neighbor_1_index = neighbor_1.x + neighbor_1.y * resolution.x;
		if (neighbor_1.x > 0 && neighbor_1.x < resolution.x && neighbor_1.y > 0 && neighbor_1.y < resolution.y)
		{
			float dist_neighbor_1 = hippt::length2(current_shading_point - render_data.g_buffer.first_hits[neighbor_1_index]);
			smallest_dist = dist_neighbor_1;
			smallest_index = neighbor_1_index;
		}

		int2 neighbor_2 = make_int2(floorf(pixel_pos_float.x), ceilf(pixel_pos_float.y));
		int neighbor_2_index = neighbor_2.x + neighbor_2.y * resolution.x;
		if (neighbor_2.x > 0 && neighbor_2.x < resolution.x && neighbor_2.y > 0 && neighbor_2.y < resolution.y)
		{
			float dist_neighbor_2 = hippt::length2(current_shading_point - render_data.g_buffer.first_hits[neighbor_2_index]);

			if (smallest_dist > dist_neighbor_2)
			{
				smallest_dist = dist_neighbor_2;
				smallest_index = neighbor_2_index;
			}
		}

		int2 neighbor_3 = make_int2(ceilf(pixel_pos_float.x), floorf(pixel_pos_float.y));
		int neighbor_3_index = neighbor_3.x + neighbor_3.y * resolution.x;
		if (neighbor_3.x > 0 && neighbor_3.x < resolution.x && neighbor_3.y > 0 && neighbor_3.y < resolution.y)
		{
			float dist_neighbor_3 = hippt::length2(current_shading_point - render_data.g_buffer.first_hits[neighbor_3_index]);

			if (smallest_dist > dist_neighbor_3)
			{
				smallest_dist = dist_neighbor_3;
				smallest_index = neighbor_3_index;
			}
		}

		int2 neighbor_4 = make_int2(ceilf(pixel_pos_float.x), ceilf(pixel_pos_float.y));
		int neighbor_4_index = neighbor_4.x + neighbor_4.y * resolution.x;
		if (neighbor_4.x > 0 && neighbor_4.x < resolution.x && neighbor_4.y > 0 && neighbor_4.y < resolution.y)
		{
			float dist_neighbor_4 = hippt::length2(current_shading_point - render_data.g_buffer.first_hits[neighbor_4_index]);

			if (smallest_dist > dist_neighbor_4)
			{
				smallest_dist = dist_neighbor_4;
				smallest_index = neighbor_4_index;
			}
		}

		if (smallest_index != -1)
			if (!check_similarity_heuristics(render_data, smallest_index, center_pixel_index, current_shading_point, current_normal))
				smallest_index = -1;

		return smallest_index;
	}
	else if (render_data.render_settings.restir_di_settings.temporal_pass.neighbor_search_strategy == 1)
	{
		// We're going to randomly look for an acceptable neighbor around the back-projected pixel location to find
		// in a given radius
		int temporal_neighbor_index = -1;
		for (int i = 0; i < render_data.render_settings.restir_di_settings.temporal_pass.max_neighbor_search_count + 1; i++)
		{
			float2 offset = make_float2(0.0f, 0.0f);
			if (i > 0)
				// Only randomly looking after we've at least checked whether or not the exact temporally reprojected location
				// is valid or not
				offset = make_float2(random_number_generator() - 0.5f, random_number_generator() - 0.5f) * render_data.render_settings.restir_di_settings.temporal_pass.neighbor_search_radius;

			int2 temporal_neighbor_screen_pixel_pos = make_int2(pixel_pos_float.x + offset.x, pixel_pos_float.y + offset.y);
			if (temporal_neighbor_screen_pixel_pos.x < 0 || temporal_neighbor_screen_pixel_pos.x >= resolution.x || temporal_neighbor_screen_pixel_pos.y < 0 || temporal_neighbor_screen_pixel_pos.y >= resolution.y)
				// Previous pixel is out of the current viewport
				continue;

			temporal_neighbor_index = temporal_neighbor_screen_pixel_pos.x + temporal_neighbor_screen_pixel_pos.y * resolution.x;
			bool neighbor_heuristics_valid = check_similarity_heuristics(render_data, temporal_neighbor_index, center_pixel_index, current_shading_point, current_normal);
			if (neighbor_heuristics_valid)
				// We found a good neighbor
				break;

			// We didn't break so we didn't find a good neighbor
			temporal_neighbor_index = -1;
		}

		return temporal_neighbor_index;
	}

	return -1;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_temporal_reuse_resampling_MIS_weight(
	const HIPRTRenderData& render_data,
	const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& temporal_neighbor_reservoir,
	const ReSTIRDISurface& temporal_neighbor_surface, const ReSTIRDISurface& center_pixel_surface,
	int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index,
	Xorshift32Generator& random_number_generator)
{
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
	return reservoir_being_resampled.M;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
	// No resampling MIS weights for this. Everything is computed in the last step where
	// we check which neighbors could have produced the sample that we picked
	return 1.0f;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS
	float nume = 0.0f;
	// We already have the target function at the center pixel, adding it to the denom
	float denom = 0.0f;

	// Hardcoding 2 in the loop for temporal reuse since we're only reusing the initial candidate
	// at our pixel and our temporal neighbor which makes 2 candidates
	for (int j = 0; j < 2; j++)
	{
		int neighbor_pixel_index;
		if (j == TEMPORAL_NEIGHBOR_RESAMPLE)
		{
			// Resampling the temporal neighbor on the first iteration
			neighbor_pixel_index = temporal_neighbor_pixel_index;
			if (neighbor_pixel_index == -1)
				continue;
		}
		else
			// Resampling at our center pixel on the second iteration
			neighbor_pixel_index = center_pixel_index;

		// Evaluating the sample that we're resampling at the neighor locations (using the neighbors surfaces)
		float target_function_at_neighbor;
		if (j == TEMPORAL_NEIGHBOR_RESAMPLE) 
			target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, reservoir_being_resampled.sample, temporal_neighbor_surface);
		else
			target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, reservoir_being_resampled.sample, center_pixel_surface);

		int M = 1;

		if (j == TEMPORAL_NEIGHBOR_RESAMPLE)
			M = temporal_neighbor_reservoir.M;

		if (M == 0)
			// No temporal history, no resampling to do, skipping this reservoir
			continue;

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH
		// No confidence weights, using M = 1.
		// 
		// Note that we still have to go through the piece of code above because it checks if
		// the temporal neighbor is valid. If the temporal neighbor isn't valid, then
		// we skip to reusing the initial candidates and we never get here.
		//
		// For this reason, we cannot just set M = 1 if not using confidence weights
		// and do the piece of code above only for confidence weights, we have to check
		// for the validity of the temporal neighbor in both cases and so we have to
		// go through the code above in both cases
		M = 1;
#endif

		denom += target_function_at_neighbor * M;
		if (j == current_neighbor)
			nume = target_function_at_neighbor * M;
		}

	if (denom == 0.0f)
		return 0.0f;
	else
		return nume / denom;
#else
#error "Unsupported bias correction mode in ReSTIR DI spatial reuse get_resampling_MIS_weight"
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE void get_temporal_reuse_normalization_denominator_numerator(
	const HIPRTRenderData& render_data, 
	const ReSTIRDIReservoir& reservoir, const ReSTIRDIReservoir& initial_candidates_reservoir, const ReSTIRDIReservoir& temporal_neighbor_reservoir, 
	const ReSTIRDISurface& center_pixel_surface, const ReSTIRDISurface& temporal_neighbor_surface, 
	int selected_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index, 
	Xorshift32Generator& random_number_generator,
	float& out_normalization_nume, float& out_normalization_denom)
{
	if (reservoir.weight_sum <= 0)
	{
		// Invalid reservoir, returning directly
		out_normalization_nume = 1.0;
		out_normalization_denom = 1.0f;

		return;
	}

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
	// 1/M MIS weights are basically confidence weights only i.e. c_i / sum(c_j) with
	// c_i = r_i.M

	out_normalization_nume = 1.0f;
	// We're simply going to divide by the sum of all the M values of all the neighbors we resampled (including the center pixel)
	// so we're only going to set the denominator to that and the numerator isn't going to change
	out_normalization_denom = 0.0f;
	for (int neighbor_index = 0; neighbor_index < 2; neighbor_index++)
	{
		ReSTIRDIReservoir neighbor_reservoir;
		if (neighbor_index == TEMPORAL_NEIGHBOR_RESAMPLE)
			neighbor_reservoir = temporal_neighbor_reservoir;
		else
			neighbor_reservoir = initial_candidates_reservoir;

		out_normalization_denom += neighbor_reservoir.M;
	}
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
	// Checking how many of our neighbors could have produced the sample that we just picked
	// and we're going to divide by the sum of M values of those neighbors
	out_normalization_denom = 0.0f;

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	out_normalization_nume = 1.0f;
#else
	out_normalization_nume = 0.0f;
#endif

	for (int neighbor_index = 0; neighbor_index < 2; neighbor_index++)
	{
		float target_function_at_neighbor;
		if (neighbor_index == TEMPORAL_NEIGHBOR_RESAMPLE)
			target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, reservoir.sample, temporal_neighbor_surface);
		else
			target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, reservoir.sample, center_pixel_surface);

		if (target_function_at_neighbor > 0.0f)
		{
			// If the neighbor could have produced this sample...
			ReSTIRDIReservoir neighbor_reservoir;
			if (neighbor_index == TEMPORAL_NEIGHBOR_RESAMPLE)
			{
				neighbor_reservoir = temporal_neighbor_reservoir;

				if (neighbor_reservoir.M == 0)
				{
					// No temporal neighbor, no taking it into account.
					// 
					// Note that no temporal neighbor isn't the same as a temporal with an UCW of 0
					// (in which case it would mean that this is a neighbor that resampled a bunch
					// of samples but couldn't find any good sample for itself)
					//
					// When M == 0 here, this means that we have no temporal history at all (either very
					// first frame of the render or after a disocclusion for example) in which case
					// we shouldn't take it into account in the MIS weight computation as, in MIS terms, 
					// this doesn't constitute a valid sampling technique (there's no sample, no technique, nothing.)

					continue;
				}
			}
			else
				neighbor_reservoir = initial_candidates_reservoir;

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
			out_normalization_denom += neighbor_reservoir.M;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
			if (neighbor_index == selected_neighbor)
				out_normalization_nume += target_function_at_neighbor;
			out_normalization_denom += target_function_at_neighbor;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
			if (neighbor_index == selected_neighbor)
				// Not multiplying by M in the numerator because this is already included in the MIS weight when resampling
				out_normalization_nume += target_function_at_neighbor;
			out_normalization_denom += target_function_at_neighbor * neighbor_reservoir.M;
#endif
		}
	}
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS
	// Nothing more to normalize, everything is already handled when resampling the neighbors with balance heuristic MIS weights in the m_i terms
	out_normalization_nume = 1.0f;
	out_normalization_denom = 1.0f;
#else
#error "Unsupported bias correction mode in ReSTIR DI spatial reuse get_normalization_denominator_numerator()"
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE void temporal_visibility_reuse(const HIPRTRenderData& render_data, ReSTIRDIReservoir& reservoir, float3 shading_point)
{
	if (reservoir.UCW == 0.0f)
		return;

	float distance_to_light;
	float3 sample_direction = reservoir.sample.point_on_light_source - shading_point;
	sample_direction /= (distance_to_light = hippt::length(sample_direction));

	hiprtRay shadow_ray;
	shadow_ray.origin = shading_point;
	shadow_ray.direction = sample_direction;

	bool visible = !evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);
	if (!visible)
		reservoir.UCW = 0.0f;
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_DI_TemporalReuse(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_TemporalReuse(HIPRTRenderData render_data, int2 res, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	uint32_t center_pixel_index = (x + y * res.x);
	if (center_pixel_index >= res.x * res.y)
		return;

	// Initializing the random generator
	unsigned int seed;
	if (render_data.render_settings.freeze_random)
		seed = wang_hash(center_pixel_index + 1);
	else
		seed = wang_hash((center_pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_seed);
	Xorshift32Generator random_number_generator(seed);

	// Surface data of the center pixel
	ReSTIRDISurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index);

	int temporal_neighbor_pixel_index = find_temporal_neighbor(render_data, center_pixel_surface.shading_point, center_pixel_surface.shading_normal, res, center_pixel_index, center_pixel_surface.material.roughness, random_number_generator);
	if (temporal_neighbor_pixel_index == -1)
	{
		// Temporal occlusion / disoccusion, temporal neighbor is invalid,
		// we're only going to resample the initial candidates so let's set that as
		// the output right away

		// The output of this temporal pass is just the initial candidates reservoir
		render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs[center_pixel_index] = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[center_pixel_index];

		return;
	}

	// Resampling the initial candidates
	ReSTIRDIReservoir new_reservoir;
	ReSTIRDIReservoir initial_candidates_reservoir = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[center_pixel_index];
	ReSTIRDIReservoir temporal_neighbor_reservoir = render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs[temporal_neighbor_pixel_index];
	ReSTIRDISurface temporal_neighbor_surface = get_pixel_surface(render_data, temporal_neighbor_pixel_index);
	
	// Will keep the index of the neighbor that has been selected by resampling. 
	// Either 0 or 1 for the temporal resampling pass
	int selected_neighbor = 0;
	float init_cand_mis_weight = get_temporal_reuse_resampling_MIS_weight(render_data, 
		initial_candidates_reservoir, temporal_neighbor_reservoir, 
		temporal_neighbor_surface, center_pixel_surface, 
		/* indicating that we're currently resampling the initial candidates */ INITIAL_CANDIDATES_RESAMPLE,
		center_pixel_index, temporal_neighbor_pixel_index, 
		random_number_generator);

	if (new_reservoir.combine_with(initial_candidates_reservoir, init_cand_mis_weight, initial_candidates_reservoir.sample.target_function, /* jacobian is 1 when reusing at the exact same spot */ 1.0f, random_number_generator))
		selected_neighbor = INITIAL_CANDIDATES_RESAMPLE;
	new_reservoir.sanity_check(make_int2(x, y));


	// ---
	// The rest of the code resamples the temporal neighbor
	// ---

	if (temporal_neighbor_reservoir.M > 0)
	{
		// M-capping the temporal neighbor
		if (render_data.render_settings.restir_di_settings.m_cap > 0)
			temporal_neighbor_reservoir.M = hippt::min(temporal_neighbor_reservoir.M, render_data.render_settings.restir_di_settings.m_cap);

		int temporal_neighbor_M = temporal_neighbor_reservoir.M;

		float target_function_at_center = 0.0f;
		if (temporal_neighbor_reservoir.UCW != 0.0f)
			// Only resampling if the temporal neighbor isn't empty
			//
			// If the temporal neiughor's reservoir is empty, then we do not get
			// inside that if() and the target function stays at 0.0f which eliminates
			// most of the computations afterwards
			target_function_at_center = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_TargetFunctionVisibility>(render_data, temporal_neighbor_reservoir.sample, center_pixel_surface);

		float jacobian_determinant = 1.0f;
		// If the neighbor reservoir is invalid, do not compute the jacobian
		if (target_function_at_center > 0.0f && temporal_neighbor_reservoir.UCW != 0.0f)
		{
			// The reconnection shift is what is implicitely used in ReSTIR DI. We need this because
			// the initial light sample candidates that we generate on the area of the lights have an
			// area measure PDF. This area measure PDF is converted to solid angle in the initial candidates
			// sampling routine by multiplying by the distance squared and dividing by the cosine
			// angle at the light source. However, a PDF in solid angle measure is only viable at a
			// given point. We say "solid angle with respect to the shading point". This means that
			// reusing a light sample with PDF (the UCW of the neighbor reservoir) in solid angle
			// from a neighbor is invalid since that PDF is only valid at the neighbor point, not
			// at the point we're resampling from (the center pixel). We thus need to convert from the
			// "solid angle PDF at the neighbor" to the solid angle at the center pixel and we do
			// that by multiplying by the jacobian determinant of the reconnection shift in solid
			// angle, Eq. 52 of 2022, "Generalized Resampled Importance Sampling".
			jacobian_determinant = get_jacobian_determinant_reconnection_shift(render_data, 
				temporal_neighbor_reservoir, 
				center_pixel_surface.shading_point,
				/* recomputing the point without the normal offset */ temporal_neighbor_surface.shading_point - temporal_neighbor_surface.shading_normal * 1.0e-4f);

			if (jacobian_determinant == -1.0f)
				// Sample too dissimilar, not going to resample it so setting
				// the jacobian to 0.0f so that the reservoir combination fails
				// for this sample
				jacobian_determinant = 0.0f;
		}

		float mis_weight = 1.0f;
		if (target_function_at_center > 0.0f)
			// No need to compute the MIS weight if the target function is 0.0f because we're never going to pick
			// that sample anyway when combining the reservoir since the resampling weight will be 0.0f because of
			// the multiplication by the target function that is 0.0f
			mis_weight = get_temporal_reuse_resampling_MIS_weight(render_data, 
				temporal_neighbor_reservoir, temporal_neighbor_reservoir, 
				temporal_neighbor_surface, center_pixel_surface, 
				/* indicating that we're currently resampling the temporal neighbor */ TEMPORAL_NEIGHBOR_RESAMPLE, 
				center_pixel_index, temporal_neighbor_pixel_index, 
				random_number_generator);

		// Combining as in Alg. 6 of the paper
		if (new_reservoir.combine_with(temporal_neighbor_reservoir, mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			selected_neighbor = 0;

		new_reservoir.sanity_check(make_int2(x, y));
	}

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	get_temporal_reuse_normalization_denominator_numerator(render_data, 
		new_reservoir, initial_candidates_reservoir, temporal_neighbor_reservoir,
		center_pixel_surface, temporal_neighbor_surface, 
		selected_neighbor,
		center_pixel_index, temporal_neighbor_pixel_index,
		random_number_generator, normalization_numerator, normalization_denominator);

	new_reservoir.end_with_normalization(normalization_numerator, normalization_denominator);
	new_reservoir.sanity_check(make_int2(x, y));

#if ReSTIR_DI_DoVisibilityReuse && ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	temporal_visibility_reuse(render_data, new_reservoir, center_pixel_surface.shading_point);
#endif

	render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs[center_pixel_index] = new_reservoir;
}

#endif
