/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIAL_REUSE_H
#define DEVICE_RESTIR_DI_SPATIAL_REUSE_H 

#include "Device/includes/Dispatcher.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Hash.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightUtils.h"
#include "Device/includes/ReSTIR/DI/SpatialMISWeight.h"
#include "Device/includes/ReSTIR/DI/SpatialNormalizationWeight.h"
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
HIPRT_HOST_DEVICE HIPRT_INLINE void count_valid_neighbors(const HIPRTRenderData& render_data, const ReSTIRDISurface& center_pixel_surface, int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation, int& out_valid_neighbor_count, int& out_valid_neighbor_M_sum, int& out_neighbor_heuristics_cache)
{
	int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * res.x;
	int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;

	out_valid_neighbor_count = 0;
	for (int neighbor_index = 0; neighbor_index < reused_neighbors_count; neighbor_index++)
	{
		int neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, neighbor_index, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport / invalid
			continue;

		if (!check_neighbor_similarity_heuristics(render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
			continue;

		out_valid_neighbor_M_sum += render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[neighbor_pixel_index].M;
		out_valid_neighbor_count++;
		out_neighbor_heuristics_cache |= (1 << neighbor_index);
	}
}


#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_DI_SpatialReuse(HIPRTRenderData render_data, int2 res)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_DI_SpatialReuse(HIPRTRenderData render_data, int2 res, int x, int y)
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

	ReSTIRDIReservoir* input_reservoir_buffer = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs;

	ReSTIRDIReservoir new_reservoir;
	// Center pixel coordinates
	int2 center_pixel_coords = make_int2(x, y);
	// Surface data of the center pixel
	ReSTIRDISurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index);
	ReSTIRDIReservoir center_pixel_reservoir = input_reservoir_buffer[center_pixel_index];

	// Rotation that is going to be used to rotate the points generated by the Hammersley sampler
	// for generating the neighbors location to resample
	float rotation_theta;
	if (render_data.render_settings.restir_di_settings.spatial_pass.do_neighbor_rotation)
		rotation_theta = 2.0f * M_PI * random_number_generator();
	else
		rotation_theta = 0.0f;

	float2 cos_sin_theta_rotation = make_float2(cos(rotation_theta), sin(rotation_theta));

	int selected_neighbor = 0;
	int neighbor_heuristics_cache = 0;
	int valid_neighbors_count = 0;
	int valid_neighbors_M_sum = 0;
	count_valid_neighbors(render_data, center_pixel_surface, center_pixel_coords, res, cos_sin_theta_rotation, valid_neighbors_count, valid_neighbors_M_sum, neighbor_heuristics_cache);

	ReSTIRDISpatialResamplingMISWeight<ReSTIR_DI_BiasCorrectionWeights> mis_weight_function;
	// Resampling the neighbors. Using neighbors + 1 here so that
	// we can use the last iteration of the loop to resample ourselves (the center pixel)
	// 
	// See the implementation of get_spatial_neighbor_pixel_index() in ReSTIR/DI/Utils.h
	int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;
	for (int neighbor_index = 0; neighbor_index < reused_neighbors_count + 1; neighbor_index++)
	{
		int neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, neighbor_index, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		if (neighbor_index < reused_neighbors_count)
		{
			// If not the center pixel, we can check the heuristics, otherwise there's no need to

			if (reused_neighbors_count <= 32)
			{
				// Our heuristics cache is a 32bit int so we can only cache 32 values are we're
				// going to have issues if we try to read more than that.
				if ((neighbor_heuristics_cache & (1 << neighbor_index)) == 0)
					continue;
			}
			else 
			{
				// So if we have more than 32 neighbors, falling back to default heuristic redundant check
				if (!check_neighbor_similarity_heuristics(render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
			 		continue;
			}
		}

		ReSTIRDIReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];
		float target_function_at_center = 0.0f;
		if (neighbor_reservoir.UCW > 0.0f)
		{
			if (neighbor_index == reused_neighbors_count)
				// No need to evaluate the center sample at the center pixel, that's exactly
				// the target function of the center reservoir
				target_function_at_center = neighbor_reservoir.sample.target_function;
			else
				target_function_at_center = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_TargetFunctionVisibility>(render_data, neighbor_reservoir.sample, center_pixel_surface);
		}

		float jacobian_determinant = 1.0f;
		// If the neighbor reservoir is invalid, do not compute the jacobian
		// Also, if this is the last neighbor resample (meaning that it is the sample pixel), 
		// the jacobian is going to be 1.0f so no need to compute
		if (target_function_at_center > 0.0f && neighbor_reservoir.UCW != 0.0f && neighbor_index != reused_neighbors_count)
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
			jacobian_determinant = get_jacobian_determinant_reconnection_shift(render_data, neighbor_reservoir, center_pixel_surface.shading_point, neighbor_pixel_index);

			if (jacobian_determinant == -1.0f)
			{
				new_reservoir.M += neighbor_reservoir.M;

				// The sample was too dissimilar and so we're rejecting it
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
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir,
			center_pixel_surface, neighbor_index, center_pixel_coords, res, cos_sin_theta_rotation);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir,
			center_pixel_reservoir, target_function_at_center, neighbor_index, neighbor_pixel_index, valid_neighbors_count, valid_neighbors_M_sum);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
		float mis_weight = mis_weight_function.get_resampling_MIS_weight(render_data, neighbor_reservoir,
			center_pixel_reservoir, target_function_at_center, neighbor_index, neighbor_pixel_index, valid_neighbors_count, valid_neighbors_M_sum);
#else
#error "Unsupported bias correction mode"
#endif

		// Combining as in Alg. 6 of the paper
		if (new_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			selected_neighbor = neighbor_index;
		new_reservoir.sanity_check(center_pixel_coords);
	}

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	ReSTIRDISpatialNormalizationWeight<ReSTIR_DI_BiasCorrectionWeights> normalization_function;
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
	normalization_function.get_normalization(render_data, new_reservoir,
		center_pixel_surface, center_pixel_coords, res, cos_sin_theta_rotation, normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	normalization_function.get_normalization(render_data, new_reservoir,
		center_pixel_surface, center_pixel_coords, res, cos_sin_theta_rotation, normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
	normalization_function.get_normalization(render_data, new_reservoir,
		center_pixel_surface, selected_neighbor, center_pixel_coords, res, cos_sin_theta_rotation, normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE
	normalization_function.get_normalization(normalization_numerator, normalization_denominator);
#else
#error "Unsupported bias correction mode"
#endif

	new_reservoir.end_with_normalization(normalization_numerator, normalization_denominator);
	new_reservoir.sanity_check(center_pixel_coords);

#if ReSTIR_DI_DoVisibilityReuse && ReSTIR_DI_RaytraceSpatialReuseReservoirs && \
	(ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z \
	|| ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS \
	|| ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE)
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
		ReSTIR_DI_visibility_reuse(render_data, new_reservoir, center_pixel_surface.shading_point);
#else
	// There are also some edge cases that we want to cover which would cause bias explosion and lead to
	// unusable renders. So this #else part is basically forcing the visibility reuse even if the user
	// doesn't want it (but that's for their own good hehe)
#if ReSTIR_DI_DoVisibilityReuse == KERNEL_OPTION_FALSE \
	&& ReSTIR_DI_BiasCorrectionUseVisiblity == KERNEL_OPTION_TRUE \
	&& ReSTIR_DI_TargetFunctionVisibility == KERNEL_OPTION_FALSE
	// Without visibility reuse, samples that are occluded can be produced. If we're not discarding
	// them here, they may be discarded in the subsequent temporal/spatial reuse
	// (which use visibility bias correction because ReSTIR_DI_BiasCorrectionUseVisiblity == KERNEL_OPTION_TRUE)
	// and this will cause brightening bias
	//
	// Using visibility in the target function completely stops the bias anyways because reservoir
	// always account for visibility. That's it. No issues of reusing/discarding/occluded/unoccluded/... samples. No bias.

	if (render_data.render_settings.restir_di_settings.use_confidence_weights)
	{
		// With confidence weights, the bias will compound so hard that it will blow up and completely corrupt the render
		// so we need to visibility-reuse the reservoir here anyways
		ReSTIR_DI_visibility_reuse(render_data, new_reservoir, center_pixel_surface.shading_point);
	}
#elif ReSTIR_DI_DoVisibilityReuse == KERNEL_OPTION_TRUE \
	&& ReSTIR_DI_BiasCorrectionUseVisiblity == KERNEL_OPTION_TRUE \
	&& ReSTIR_DI_RaytraceSpatialReuseReservoirs == KERNEL_OPTION_FALSE \
	&& ReSTIR_DI_TargetFunctionVisibility == KERNEL_OPTION_FALSE
	// This is almost the same situation as above: we're resampling neighbors into the center pixel
	// but we don't ray trace the final reservoir (because ReSTIR_DI_RaytraceSpatialReuseReservoirs == KERNEL_OPTION_FALSE).
	// This means that we may now have a reservoir in the center pixel that is actually occluded:
	// the center pixel is now able to produce samples (through resampling its neighbors) that
	// are actually occluded. In the next temporal/spatial pass, this center pixel may be
	// resampled itself by one of its neighbors but with visibility in mind
	// (because && ReSTIR_DI_BiasCorrectionUseVisiblity == KERNEL_OPTION_TRUE) and this may cause
	// that center pixel to be wrongly discarded --> brightening bias
	//
	// Using visibility in the target function completely stops the bias anyways because reservoir
	// always account for visibility. That's it. No issues of reusing/discarding/occluded/unoccluded/... samples. No bias.
	if (render_data.render_settings.restir_di_settings.use_confidence_weights)
	{
		// With confidence weights, the bias will compound so hard that it will blow up and completely corrupt the render
		// so we need to visibility-reuse the reservoir here anyways
		ReSTIR_DI_visibility_reuse(render_data, new_reservoir, center_pixel_surface.shading_point);
	}
#endif

#endif

	render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs[center_pixel_index] = new_reservoir;
}

#endif
