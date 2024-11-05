/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIOTEMPORAL_REUSE_H
#define DEVICE_RESTIR_DI_SPATIOTEMPORAL_REUSE_H

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/DI/SpatiotemporalMISWeight.h"
#include "Device/includes/ReSTIR/DI/SpatiotemporalNormalizationWeight.h"
#include "Device/includes/ReSTIR/DI/Surface.h"
#include "Device/includes/ReSTIR/DI/Utils.h"
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
 * [7] [Reddit Post for the Jacobian term needed] https://www.reddit.com/r/GraphicsProgramming/comments/1eo5hqr/restir_di_light_sample_pdf_confusion/
 * [8] [Rearchitecting Spatiotemporal Resampling for Production] https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production
 */

#define TEMPORAL_NEIGHBOR_ID 0

HIPRT_HOST_DEVICE HIPRT_INLINE bool do_include_spatial_visibility_term_or_not(const HIPRTRenderData& render_data, int current_neighbor_index)
{
	const SpatialPassSettings& spatial_settings = render_data.render_settings.restir_di_settings.spatial_pass;
	bool visibility_only_on_last_pass = spatial_settings.do_visibility_only_last_pass;
	bool is_last_pass = spatial_settings.spatial_pass_index == spatial_settings.number_of_passes - 1;

	// Only using the visibility term on the last pass if so desired
	bool include_target_function_visibility = visibility_only_on_last_pass && is_last_pass;
	// Also allowing visibility if we want it at every pass
	include_target_function_visibility |= !spatial_settings.do_visibility_only_last_pass;

	// Only doing visibility for a few neighbors depending on 'neighbor_visibility_count'
	include_target_function_visibility &= current_neighbor_index < spatial_settings.neighbor_visibility_count;

	// Only doing visibility if we want it at all
	include_target_function_visibility &= ReSTIR_DI_SpatialTargetFunctionVisibility;

	// We don't want visibility for the center pixel because we're going to reuse the
	// target function stored in the reservoir anyways
	// Note: the center pixel has index 'spatial_settings.reuse_neighbor_count'
	// while actual *neighbors* have index between [0, spatial_settings.reuse_neighbor_count - 1]
	include_target_function_visibility &= current_neighbor_index != spatial_settings.reuse_neighbor_count;

	return include_target_function_visibility;
}

/**
 * Returns -1 if there is no valid temporal neighbor.
 * The linear buffer index of the temporal neighbor otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE int3 load_temporal_neighbor_data(const HIPRTRenderData& render_data, const ReSTIRDISurface& center_pixel_surface, int center_pixel_index, int2 res, 
	ReSTIRDIReservoir& out_temporal_neighbor_reservoir, ReSTIRDISurface& out_temporal_neighbor_surface, Xorshift32Generator& random_number_generator)
{
	int3 temporal_neighbor_pixel_index_and_pos = find_temporal_neighbor_index(render_data, render_data.g_buffer.first_hits[center_pixel_index], center_pixel_surface.shading_normal, res, center_pixel_index, random_number_generator);
	if (temporal_neighbor_pixel_index_and_pos.x == -1 || render_data.render_settings.freeze_random)
		// Temporal occlusion / disoccusion --> temporal neighbor is invalid,
		// we're only going to resample the initial candidates so let's set that as
		// the output right away
		//
		// We're also 'disabling' temporal accumulation if the renderer's random is frozen otherwise
		// very strong correlations will creep up, corrupt the render and potentially invalidate
		// performance measurements (which we're probably trying to measure since we froze the random)
		return temporal_neighbor_pixel_index_and_pos;

	out_temporal_neighbor_reservoir = render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs[temporal_neighbor_pixel_index_and_pos.x];
	if (out_temporal_neighbor_reservoir.M == 0)
		// No temporal neighbor
		return temporal_neighbor_pixel_index_and_pos;

	// Reading from the previous g-buffer or not depending on whether or not the prev g-buffer is available
	// (it may not be if we're accumulating because then, it's useless since there is no motion)
	out_temporal_neighbor_surface = get_pixel_surface(render_data, temporal_neighbor_pixel_index_and_pos.x, render_data.render_settings.use_prev_frame_g_buffer());

	if (out_temporal_neighbor_surface.material.is_emissive())
		// Can't resample the temporal neighbor if it's emissive
		return temporal_neighbor_pixel_index_and_pos;

	return temporal_neighbor_pixel_index_and_pos;
}

/**
 * Counts how many neighbors are eligible for reuse.
 * This is needed for proper normalization by pairwise MIS weights.
 *
 * A neighbor is not eligible if it is outside of the viewport or if
 * it doesn't satisfy the normal/plane/roughness heuristics
 *
 * 'out_valid_neighbor_M_sum' is the sum of the M values (confidences) of the
 * valid neighbors. Used by confidence-weights pairwise MIS weights
 *
 * The bits of 'out_neighbor_heuristics_cache' are 1 or 0 depending on whether or not
 * the corresponding neighbor was valid or not (can be reused later to avoid having to
 * re-evauate the heuristics). Neighbor 0 is LSB.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void count_valid_spatiotemporal_neighbors(const HIPRTRenderData& render_data, const ReSTIRDISurface& center_pixel_surface, int center_pixel_index, int2 temporal_neighbor_position, int2 res, float2 cos_sin_theta_rotation, int& out_valid_neighbor_count, int& out_valid_neighbor_M_sum, int& out_neighbor_heuristics_cache)
{
	int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;

	out_valid_neighbor_count = 0;
	for (int neighbor_index = 0; neighbor_index < reused_neighbors_count; neighbor_index++)
	{
		int neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, neighbor_index, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.reuse_radius, temporal_neighbor_position, res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport / invalid
			continue;

		if (!check_neighbor_similarity_heuristics(render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal, render_data.render_settings.use_prev_frame_g_buffer()))
			continue;

		out_valid_neighbor_M_sum += render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[neighbor_pixel_index].M;
		out_valid_neighbor_count++;
		out_neighbor_heuristics_cache |= (1 << neighbor_index);
	}
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) ReSTIR_DI_SpatiotemporalReuse(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_SpatiotemporalReuse(HIPRTRenderData render_data, int2 res, int x, int y)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
#endif
	if (x >= res.x || y >= res.y)
		return;

	uint32_t center_pixel_index = (x + y * res.x);

	if (!render_data.aux_buffers.pixel_active[center_pixel_index] || !render_data.g_buffer.camera_ray_hit[center_pixel_index])
		// Pixel inactive because of adaptive sampling, returning
		return;

	// Initializing the random generator
	unsigned int seed;
	if (render_data.render_settings.freeze_random)
		seed = wang_hash(center_pixel_index + 1);
	else
		seed = wang_hash((center_pixel_index + 1) * (render_data.render_settings.sample_number + 1) * render_data.random_seed);
	Xorshift32Generator random_number_generator(seed);

	int2 center_pixel_coords = make_int2(x, y);

	// Surface data of the center pixel
	ReSTIRDISurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index);
	if (center_pixel_surface.material.is_emissive())
		// Not doing ReSTIR on directly visible emissive materials
		return;

	if (render_data.render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested)
		// We requested a temporal buffer clear for ReSTIR DI
		render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs[center_pixel_index] = ReSTIRDIReservoir();

	ReSTIRDIReservoir temporal_neighbor_reservoir;
	ReSTIRDISurface temporal_neighbor_surface;
	int3 temporal_neighbor_pixel_index_and_pos = load_temporal_neighbor_data(render_data, center_pixel_surface, center_pixel_index, res, temporal_neighbor_reservoir, temporal_neighbor_surface, random_number_generator);
	if ((temporal_neighbor_pixel_index_and_pos.x == -1 || temporal_neighbor_reservoir.M <= 1) && render_data.render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost)
		// Increasing the number of spatial samples for disoclussions
		render_data.render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count = render_data.render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count;

	// Rotation that is going to be used to rotate the points generated by the Hammersley sampler
	// for generating the spatial neighbors location to resample
	float rotation_theta;
	if (render_data.render_settings.restir_di_settings.spatial_pass.do_neighbor_rotation)
		rotation_theta = M_TWO_PI * random_number_generator();
	else
		rotation_theta = 0.0f;

	float2 cos_sin_theta_rotation = make_float2(cos(rotation_theta), sin(rotation_theta));

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
	// Only used with MIS-like weight
	// 
	// Will keep the index of the neighbor that has been selected by resampling. 
	int selected_neighbor = 0;
#endif

	int neighbor_heuristics_cache = 0;
	int valid_neighbors_count = 0;
	int valid_neighbors_M_sum = 0;
	count_valid_spatiotemporal_neighbors(render_data, center_pixel_surface, center_pixel_index, make_int2(temporal_neighbor_pixel_index_and_pos.y, temporal_neighbor_pixel_index_and_pos.z), res, cos_sin_theta_rotation, valid_neighbors_count, valid_neighbors_M_sum, neighbor_heuristics_cache);
	if (temporal_neighbor_pixel_index_and_pos.x != -1 && temporal_neighbor_reservoir.M > 0)
	{
		// Adding the temporal neighbor to the count 
		valid_neighbors_count++;
		valid_neighbors_M_sum += temporal_neighbor_reservoir.M;
	}

	ReSTIRDIReservoir spatiotemporal_output_reservoir;
	ReSTIRDIReservoir initial_candidates_reservoir = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[center_pixel_index];
	ReSTIRDISpatiotemporalResamplingMISWeight<ReSTIR_DI_BiasCorrectionWeights> mis_weight_function;
	if (temporal_neighbor_pixel_index_and_pos.x != -1)
	{
		// Resampling the temporal neighbor

		if (temporal_neighbor_reservoir.M > 0)
		{
			float target_function_at_center = 0.0f;
			if (temporal_neighbor_reservoir.UCW > 0.0f)
				// Only resampling if the temporal neighbor isn't empty
				//
				// If the temporal neiughor's reservoir is empty, then we do not get
				// inside that if() and the target function stays at 0.0f which eliminates
				// most of the computations afterwards
				//
				// Matching the visibility used here with the bias correction mode for ease 
				// of use (and because manually handling the visibility in the target 
				// function of the temporal reuse is tricky for the user to use in 
				// combination with other parameters and on top of that, it makes little 
				// technical sense since our temporal neighbor is supposed to be unoccluded 
				// (unless geometry moves around in the scene but that's another problem)
				target_function_at_center = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, temporal_neighbor_reservoir.sample, center_pixel_surface, random_number_generator);

			float jacobian_determinant = 1.0f;
			// If the neighbor reservoir is invalid, do not compute the jacobian
			if (target_function_at_center > 0.0f && temporal_neighbor_reservoir.UCW > 0.0f && !(temporal_neighbor_reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE))
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

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
			float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(temporal_neighbor_reservoir.M);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
			float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(temporal_neighbor_reservoir.M);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
			float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, temporal_neighbor_reservoir.M);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH
			float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data,
				temporal_neighbor_reservoir, center_pixel_surface, temporal_neighbor_surface,
				TEMPORAL_NEIGHBOR_ID, initial_candidates_reservoir.M, temporal_neighbor_reservoir.M, 
				center_pixel_index, make_int2(temporal_neighbor_pixel_index_and_pos.y, temporal_neighbor_pixel_index_and_pos.z),
				res, cos_sin_theta_rotation, random_number_generator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
			bool update_mc = initial_candidates_reservoir.M > 0 && initial_candidates_reservoir.UCW > 0.0f;

			float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, temporal_neighbor_reservoir, initial_candidates_reservoir,
				target_function_at_center, temporal_neighbor_pixel_index_and_pos.x, valid_neighbors_count, valid_neighbors_M_sum, 
				update_mc, /* resample canonical */ false, random_number_generator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
			bool update_mc = initial_candidates_reservoir.M > 0 && initial_candidates_reservoir.UCW > 0.0f;

			float temporal_neighbor_resampling_mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, temporal_neighbor_reservoir,
				initial_candidates_reservoir, target_function_at_center, temporal_neighbor_pixel_index_and_pos.x, valid_neighbors_count, valid_neighbors_M_sum, 
				update_mc, /* resample canonical */ false, random_number_generator);
#else
#error "Unsupported bias correction mode"
#endif

			// Combining as in Alg. 6 of the paper
			if (spatiotemporal_output_reservoir.combine_with(temporal_neighbor_reservoir, temporal_neighbor_resampling_mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			{
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
				// Only used with MIS-like weight
				selected_neighbor = TEMPORAL_NEIGHBOR_ID;
#endif

				// Using ReSTIR_DI_BiasCorrectionUseVisibility here because that's what we use in the resampling target function
#if ReSTIR_DI_BiasCorrectionUseVisibility == KERNEL_OPTION_FALSE
				// We cannot be certain that the visibility of the temporal neighbor
				// chosen is exactly the same so we're clearing the unoccluded flag
				spatiotemporal_output_reservoir.sample.flags &= ~ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;
#else
				// However, if we're using the visibility in the target function, then
				// the temporal neighobr could never have been selected unless it is
				// unoccluded so we can add the flag
				spatiotemporal_output_reservoir.sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;
#endif
			}
		}

		spatiotemporal_output_reservoir.sanity_check(center_pixel_coords);
	}
















	ReSTIRDIReservoir* spatial_input_reservoir_buffer = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs;

	// Resampling the neighbors. Using neighbors + 1 here so that
	// we can use the last iteration of the loop to resample the *initial candidates reservoir*
	int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;
	int start_index = 0;
	if (valid_neighbors_M_sum == 0)
		// No spatial resampling to do, only the initial candidate reservoir (potentially)
		// so we can directly start there
		start_index = reused_neighbors_count;
	for (int spatial_neighbor_index = start_index; spatial_neighbor_index < reused_neighbors_count + 1; spatial_neighbor_index++)
	{
		// We can already check whether or not this neighbor is going to be
		// accepted at all by checking the heuristic cache
		if (spatial_neighbor_index < reused_neighbors_count && reused_neighbors_count <= 32)
			// If not the center pixel, we can check the heuristics, otherwise there's no need to,
			// we know that the center pixel will be accepted
			// 
			// Our heuristics cache is a 32bit int so we can only cache 32 values are we're
			// going to have issues if we try to read more than that.
			if ((neighbor_heuristics_cache & (1 << spatial_neighbor_index)) == 0)
				// Neighbor not passing the heuristics tests, skipping it right away
				continue;

		int neighbor_pixel_index = -1;
		if (spatial_neighbor_index == reused_neighbors_count)
			// Last iteration, resampling the initial candidates
			neighbor_pixel_index = center_pixel_index;
		else
			// Resampling around the temporal neighbor location
			neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, spatial_neighbor_index, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.reuse_radius, 
				make_int2(temporal_neighbor_pixel_index_and_pos.y, temporal_neighbor_pixel_index_and_pos.z), res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));

		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		if (spatial_neighbor_index < reused_neighbors_count && reused_neighbors_count > 32)
			// If not the center pixel, we can check the heuristics
			// 
			// Only checking the heuristic if we have more than 32 neighbors (does not fit in the heuristic cache)
			// If we have less than 32 neighbors, we've already checked the cache at the beginning of this for loop
			if (!check_neighbor_similarity_heuristics(render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal, render_data.render_settings.use_prev_frame_g_buffer()))
				continue;

		// Neighbor surface needed for roughness m-capping and jacobian determinant
		ReSTIRDIReservoir neighbor_reservoir;
		if (spatial_neighbor_index == reused_neighbors_count)
			// Last iteration, resampling the initial candidates
			neighbor_reservoir = initial_candidates_reservoir;
		else
			neighbor_reservoir = spatial_input_reservoir_buffer[neighbor_pixel_index];

		float target_function_at_center = 0.0f;
		bool do_neighbor_target_function_visibility = do_include_spatial_visibility_term_or_not(render_data, spatial_neighbor_index);
		if (neighbor_reservoir.UCW > 0.0f)
		{
			if (spatial_neighbor_index == reused_neighbors_count)
				// No need to evaluate the center sample at the center pixel, that's exactly
				// the target function of the center reservoir
				target_function_at_center = neighbor_reservoir.sample.target_function;
			else
			{
				if (do_neighbor_target_function_visibility)
					target_function_at_center = ReSTIR_DI_evaluate_target_function<KERNEL_OPTION_TRUE>(render_data, neighbor_reservoir.sample, center_pixel_surface, random_number_generator);
				else
					target_function_at_center = ReSTIR_DI_evaluate_target_function<KERNEL_OPTION_FALSE>(render_data, neighbor_reservoir.sample, center_pixel_surface, random_number_generator);
			}
		}

		float jacobian_determinant = 1.0f;
		// If the neighbor reservoir is invalid, do not compute the jacobian
		// Also, if this is the last neighbor resample (meaning that it is the sample pixel),
		// the jacobian is going to be 1.0f so no need to compute
		if (target_function_at_center > 0.0f && neighbor_reservoir.UCW > 0.0f && spatial_neighbor_index != reused_neighbors_count && !(neighbor_reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_ENVMAP_SAMPLE))
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
			ReSTIRDISurface neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, render_data.render_settings.use_prev_frame_g_buffer());

			jacobian_determinant = get_jacobian_determinant_reconnection_shift(render_data, neighbor_reservoir, center_pixel_surface.shading_point, neighbor_surface.shading_point);

			if (jacobian_determinant == -1.0f)
			{
				spatiotemporal_output_reservoir.M += neighbor_reservoir.M;

				// The sample was too dissimilar and so we're rejecting it (not
				// exactly rejecting but acting as if its contribution is 0)
				continue;
			}
		}

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(neighbor_reservoir.M);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(neighbor_reservoir.M);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir.M);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH
		// Using 'spatial_neighbor_index + 1' in this function call because the index
		// 0 is for the temporal neighbor so we start at 1 by using '+ 1'
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir, center_pixel_surface, temporal_neighbor_surface,
			spatial_neighbor_index + 1, initial_candidates_reservoir.M, temporal_neighbor_reservoir.M, 
			center_pixel_index, make_int2(temporal_neighbor_pixel_index_and_pos.y, temporal_neighbor_pixel_index_and_pos.z),
			res, cos_sin_theta_rotation, random_number_generator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
		bool update_mc = initial_candidates_reservoir.M > 0 && initial_candidates_reservoir.UCW > 0.0f;

		float mis_weight;
		if (neighbor_reservoir.UCW == 0.0f && !update_mc)
			mis_weight = 1.0f;
		else
			mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir, initial_candidates_reservoir,
				target_function_at_center, neighbor_pixel_index, valid_neighbors_count, valid_neighbors_M_sum,
				update_mc, spatial_neighbor_index == reused_neighbors_count, random_number_generator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
		bool update_mc = initial_candidates_reservoir.M > 0 && initial_candidates_reservoir.UCW > 0.0f;

		float mis_weight;
		if (neighbor_reservoir.UCW == 0.0f && !update_mc)
			mis_weight = 1.0f;
		else
			mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir, initial_candidates_reservoir, 
				target_function_at_center, neighbor_pixel_index, valid_neighbors_count, valid_neighbors_M_sum,
				update_mc, spatial_neighbor_index == reused_neighbors_count, random_number_generator);
#else
#error "Unsupported bias correction mode"
#endif

		// Combining as in Alg. 6 of the paper
		if (spatiotemporal_output_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
		{
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
			// Only used with MIS-like weight
			// 
			// + 1 here because we've already resampled the temporal neighbor so we need to account for that
			selected_neighbor = spatial_neighbor_index + 1;
#endif

			if (do_neighbor_target_function_visibility)
				// If we resampled the neighbor with visibility, then we are sure that we can set the flag
				spatiotemporal_output_reservoir.sample.flags |= ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;
			else
			{
				// If we didn't resample the neighbor with visibility
				if (spatial_neighbor_index == reused_neighbors_count)
					// If we just resampled the center pixel, then we can copy the visibility flag
					spatiotemporal_output_reservoir.sample.flags |= neighbor_reservoir.sample.flags & ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;
				else
					// This was not the center pixel, we cannot be certain what the visibility at the center
					// pixel of the neighbor sample we just resample is so we're clearing the bit
					spatiotemporal_output_reservoir.sample.flags &= ~ReSTIRDISampleFlags::RESTIR_DI_FLAGS_UNOCCLUDED;
			}
		}
		spatiotemporal_output_reservoir.sanity_check(center_pixel_coords);
	}

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	ReSTIRDISpatiotemporalNormalizationWeight<ReSTIR_DI_BiasCorrectionWeights> normalization_function;
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
	normalization_function.get_normalization(render_data, spatiotemporal_output_reservoir, initial_candidates_reservoir, center_pixel_surface, 
		temporal_neighbor_reservoir.M, center_pixel_index, make_int2(temporal_neighbor_pixel_index_and_pos.y, temporal_neighbor_pixel_index_and_pos.z),
		res, cos_sin_theta_rotation, normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	normalization_function.get_normalization(render_data, 
		spatiotemporal_output_reservoir, center_pixel_surface, temporal_neighbor_surface,
		initial_candidates_reservoir.M, temporal_neighbor_reservoir.M, center_pixel_index, 
		make_int2(temporal_neighbor_pixel_index_and_pos.y, temporal_neighbor_pixel_index_and_pos.z), res, cos_sin_theta_rotation, 
		normalization_numerator, normalization_denominator, random_number_generator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
	normalization_function.get_normalization(render_data, spatiotemporal_output_reservoir, center_pixel_surface, temporal_neighbor_surface, 
		selected_neighbor, initial_candidates_reservoir.M, temporal_neighbor_reservoir.M, center_pixel_index, make_int2(temporal_neighbor_pixel_index_and_pos.y, temporal_neighbor_pixel_index_and_pos.z),
		res, cos_sin_theta_rotation, normalization_numerator, normalization_denominator, 
		random_number_generator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#else
#error "Unsupported bias correction mode"
#endif

	spatiotemporal_output_reservoir.end_with_normalization(normalization_numerator, normalization_denominator);
	spatiotemporal_output_reservoir.sanity_check(center_pixel_coords);

	// Only these 3 weighting schemes are affected
#if (ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z \
	|| ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS \
	|| ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE) \
	&& ReSTIR_DI_BiasCorrectionUseVisibility == KERNEL_OPTION_TRUE \
	&& (ReSTIR_DI_DoVisibilityReuse == KERNEL_OPTION_TRUE || (ReSTIR_DI_InitialTargetFunctionVisibility == KERNEL_OPTION_TRUE && ReSTIR_DI_SpatialTargetFunctionVisibility == KERNEL_OPTION_TRUE))
	// Why is this needed?
	//
	// Picture the case where we have visibility reuse (at the end of the initial candidates sampling pass),
	// visibility term in the bias correction target function (when counting the neighbors that could
	// have produced the picked sample) and 2 spatial reuse passes.
	//
	// The first spatial reuse pass reuses from samples that were produced with visibility in mind
	// (because of the visibility reuse pass that discards occluded samples). This means that we need
	// the visibility in the target function used when counting the neighbors that could have produced
	// the picked sample otherwise we may think that our neighbor could have produced the picked
	// sample where actually it couldn't because the sample is occluded at the neighbor. We would
	// then have a Z denominator (with 1/Z weights) that is too large and we'll end up with darkening.
	//
	// Now at the end of the first spatial reuse pass, the center pixel ends up with a sample that may
	// or may not be occluded from the center's pixel point of view. We didn't include the visibility
	// in the target function when resampling the neighbors (only when counting the "correct" neighbors
	// but that's all) so we are not giving a 0 weight to occluded resampled neighbors --> it is possible
	// that we picked an occluded sample.
	//
	// In the second spatial reuse pass, we are now going to resample from our neighbors and get some
	// samples that were not generated with occlusion in mind (because the resampling target function of
	// the first spatial reuse doesn't include visibility). Yet, we are going to weight them with occlusion
	// in mind. This means that we are probably going to discard samples because of occlusion that could
	// have been generated because they are generated without occlusion test. We end up discarding too many
	// samples --> brightening bias.
	//
	// With the visibility reuse at the end of each spatial pass, we force samples at the end of each
	// spatial reuse to take visibility into account so that when we weight them with visibility testing,
	// everything goes well
	//
	// As an optimization, we also do this for the pairwise MIS because pairwise MIS evaluates the target function
	// of reservoirs at their own location. Doing the visibility reuse here ensures that a reservoir sample at its own location
	// includes visibility and so we do not need to recompute the target function of the neighbors in this case. We can just
	// reuse the target function stored in the reservoir
	//
	// We also give the user the choice to remove bias using this option or not as it introduces very little bias
	// in practice (but noticeable when switching back and forth between reference image/biased image)
	//
	// We only need this if we're going to temporally reuse (because then the output of the spatial reuse must be correct
	// for the temporal reuse pass) or if we have multiple spatial reuse passes and this is not the last spatial pass
	if (render_data.render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass || render_data.render_settings.restir_di_settings.spatial_pass.number_of_passes - 1 != render_data.render_settings.restir_di_settings.spatial_pass.spatial_pass_index)
		ReSTIR_DI_visibility_reuse(render_data, spatiotemporal_output_reservoir, center_pixel_surface.shading_point, center_pixel_surface.last_hit_primitive_index, random_number_generator);
#endif

	// M-capping so that we don't have to M-cap when reading reservoirs on the next frame
	if (render_data.render_settings.restir_di_settings.m_cap > 0)
		// M-capping the temporal neighbor if an M-cap has been given
		spatiotemporal_output_reservoir.M = hippt::min(spatiotemporal_output_reservoir.M, render_data.render_settings.restir_di_settings.m_cap);

	render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs[center_pixel_index] = spatiotemporal_output_reservoir;
}

#endif
