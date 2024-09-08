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
 */
HIPRT_HOST_DEVICE HIPRT_INLINE int count_valid_neighbors(const HIPRTRenderData& render_data, const ReSTIRDISurface& center_pixel_surface, int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation)
{
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS || \
	ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_CONFIDENCE_WEIGHTS || \
	ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE || \
	ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE_CONFIDENCE_WEIGHTS

	int valid_neighbors_count = 0;
	int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * res.x;
	int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;

	for (int neighbor_index = 0; neighbor_index < reused_neighbors_count; neighbor_index++)
	{
		int neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, neighbor_index, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport / invalid
			continue;

		if (!check_neighbor_similarity_heuristics(render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
			continue;

		valid_neighbors_count++;
	}

	return valid_neighbors_count;
#else
	// Counting the valid neighbors is only necessary for proper normalization with pairwise MIS.
	// Thus, if we're not using pairwise MIS, we can just return -1 because we don't care about the
	// value returned by that function anyways
	return -1;
#endif
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
	uint32_t center_pixel_index = (x + y * res.x);
	if (center_pixel_index >= res.x * res.y)
		return;

	if (!render_data.aux_buffers.pixel_active[center_pixel_index])
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
	int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;
	int valid_neighbors_count = count_valid_neighbors(render_data, center_pixel_surface, center_pixel_coords, res, cos_sin_theta_rotation);

	ReSTIRDISpatialResamplingMISWeight<ReSTIR_DI_BiasCorrectionWeights> resampling_mis_weight;
	// Resampling the neighbors. Using neighbors + 1 here so that
	// we can use the last iteration of the loop to resample ourselves (the center pixel)
	// 
	// See the implementation of get_spatial_neighbor_pixel_index() in ReSTIR/DI/Utils.h
	for (int neighbor = 0; neighbor < reused_neighbors_count + 1; neighbor++)
	{
		int neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, neighbor, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		if (!check_neighbor_similarity_heuristics(render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
			continue;

		ReSTIRDISurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index);
		ReSTIRDIReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];
		float target_function_at_center = 0.0f;
		if (neighbor_reservoir.UCW > 0.0f)
		{
			if (neighbor == reused_neighbors_count)
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
		if (target_function_at_center > 0.0f && neighbor_reservoir.UCW != 0.0f && neighbor != reused_neighbors_count)
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

		float mis_weight = resampling_mis_weight.get_resampling_MIS_weight(render_data,
			neighbor_reservoir, center_pixel_reservoir,
			center_pixel_surface, neighbor_pixel_surface,
			target_function_at_center,
			neighbor, reused_neighbors_count, valid_neighbors_count,
			center_pixel_coords, res,
			cos_sin_theta_rotation,
			random_number_generator);

		// Combining as in Alg. 6 of the paper
		if (new_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			selected_neighbor = neighbor;
		new_reservoir.sanity_check(center_pixel_coords);
	}

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	ReSTIRDISpatialNormalizationWeight<ReSTIR_DI_BiasCorrectionWeights> normalization_weight;
	normalization_weight.get_normalization(render_data,
		new_reservoir,
		center_pixel_surface,
		selected_neighbor, reused_neighbors_count,
		center_pixel_coords, res,
		cos_sin_theta_rotation,
		random_number_generator, normalization_numerator, normalization_denominator);

	new_reservoir.end_with_normalization(normalization_numerator, normalization_denominator);
	new_reservoir.sanity_check(center_pixel_coords);

#if ReSTIR_DI_DoVisibilityReuse && (ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS)
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
	// of reservoirs at their own location so doing the visibility reuse here ensure that we do not run into
	// "invalid visibility issues" that would cause bias.
	//
	// We need this at every spatial reuse pass expect the last one because we're not going to re-use anything
	// after the last pass.
	if (render_data.render_settings.restir_di_settings.spatial_pass.number_of_passes - 1 != render_data.render_settings.restir_di_settings.spatial_pass.spatial_pass_index)
		ReSTIR_DI_visibility_reuse(render_data, new_reservoir, center_pixel_surface.shading_point);
#endif

	render_data.render_settings.restir_di_settings.spatial_pass.output_reservoirs[center_pixel_index] = new_reservoir;
}

#endif
