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
#include "Device/includes/ReSTIR/ReSTIR_DI_Utils.H"
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
 */

/**
 * Returns the linear index that can be used directly to index a buffer
 * of render_data for getting data of the temporal neighbor
 */
HIPRT_HOST_DEVICE HIPRT_INLINE int find_temporal_neighbor(const HIPRTCamera& previous_frame_camera, int center_pixel_index)
{
	return center_pixel_index;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_temporal_reuse_resampling_MIS_weight(const HIPRTRenderData& render_data, const ReSTIRDIReservoir& neighbor_reservoir, int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index, Xorshift32Generator& random_number_generator)
{
#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_M || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	int neighbor_M = neighbor_reservoir.M;
	if (current_neighbor == 0)
		// M-capping the temporal neighbor
		if (render_data.render_settings.restir_di_settings.m_cap > 0)
			neighbor_M = hippt::min(neighbor_M, render_data.render_settings.restir_di_settings.m_cap);

	return neighbor_M;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
	// No resampling MIS weights for this. Everything is computed in the last step where
	// we check which neighbors could have produced the sample that we picked
	return 1.0f;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH || ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS
	float nume = 0.0f;
	// We already have the target function at the center pixel, adding it to the denom
	float denom = 0.0f;

	// Hardocding 2 in the loop for temporal reuse since we're only reusing the initial candidate
	// at our pixel and our temporal neighbor which makes 2 candidates
	for (int j = 0; j < 2; j++)
	{
		int neighbor_pixel_index;
		if (j == 0)
		{
			// Resampling the temporal neighbor on the first iteration
			neighbor_pixel_index = temporal_neighbor_pixel_index;
			if (neighbor_pixel_index == -1)
				continue;
		}
		else
			// Resampling at our center pixel on the second iteration
			neighbor_pixel_index = center_pixel_index;

		ReSTIRDISurface neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index);

		float target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, neighbor_reservoir.sample, neighbor_surface);

		int M = 1;

		ReSTIRDIReservoir* input_reservoirs;
		if (j == 0)
			// First neighbor is the temporal neighbor
			input_reservoirs = render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs;
		else
			// Second neighbor is the center pixel itself, from the initial candidates pass
			input_reservoirs = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;

		if (j == 0)
		{
			// M-capping the temporal neighbor
			if (render_data.render_settings.restir_di_settings.m_cap > 0)
			{
				int temporal_neighbor_M = hippt::min(input_reservoirs[neighbor_pixel_index].M, render_data.render_settings.restir_di_settings.m_cap);
				if (temporal_neighbor_M == 0)
					// No temporal history, no taking this into account in the MIS weight
					continue;
				else
					M = temporal_neighbor_M;
			}
		}

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

HIPRT_HOST_DEVICE HIPRT_INLINE void get_temporal_reuse_normalization_denominator_numerator(float& out_normalization_nume, float& out_normalization_denom, const HIPRTRenderData& render_data, const ReSTIRDIReservoir& new_reservoir, int selected_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index, Xorshift32Generator& random_number_generator)
{
	if (new_reservoir.weight_sum <= 0)
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
	for (int neighbor = 0; neighbor < 2; neighbor++)
	{
		int neighbor_pixel_index;
		if (neighbor == 0)
		{
			// Resampling the temporal neighbor on the first iteration
			neighbor_pixel_index = temporal_neighbor_pixel_index;
			if (neighbor_pixel_index == -1)
				continue;
		}
		else
			// Resampling at our center pixel on the second iteration
			neighbor_pixel_index = center_pixel_index;

		ReSTIRDIReservoir* input_reservoirs;
		if (neighbor == 0)
			// First neighbor is the temporal neighbor
			input_reservoirs = render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs;
		else
			// Second neighbor is the center pixel itself, from the initial candidates pass
			input_reservoirs = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;

		ReSTIRDIReservoir neighbor_reservoir = input_reservoirs[neighbor_pixel_index];

		int neighbor_M = neighbor_reservoir.M;
		if (neighbor == 0)
			if (render_data.render_settings.restir_di_settings.m_cap > 0)
				// M-capping the temporal neighbor
				neighbor_M = hippt::min(neighbor_M, render_data.render_settings.restir_di_settings.m_cap);

		out_normalization_denom += neighbor_M;
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

	for (int neighbor = 0; neighbor < 2; neighbor++)
	{
		int neighbor_pixel_index;
		if (neighbor == 0)
		{
			// Resampling the temporal neighbor on the first iteration
			neighbor_pixel_index = temporal_neighbor_pixel_index;
			if (neighbor_pixel_index == -1)
				continue;
		}
		else
			// Resampling at our center pixel on the second iteration
			neighbor_pixel_index = center_pixel_index;

		// Getting the surface data at the neighbor
		ReSTIRDISurface neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index);
		float target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_SpatialReuseBiasUseVisiblity>(render_data, new_reservoir.sample, neighbor_surface);

		if (target_function_at_neighbor > 0.0f)
		{
			// If the neighbor could have produced this sample...
			ReSTIRDIReservoir neighbor_reservoir;
			if (neighbor == 0)
			{
				neighbor_reservoir = render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs[neighbor_pixel_index];
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
				neighbor_reservoir = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs[neighbor_pixel_index];

#if ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
			int neighbor_M = neighbor_reservoir.M;
			if (neighbor == 0)
				if (render_data.render_settings.restir_di_settings.m_cap > 0)
					// M-capping the temporal neighbor
					neighbor_M = hippt::min(neighbor_M, render_data.render_settings.restir_di_settings.m_cap);

			out_normalization_denom += neighbor_M;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
			if (neighbor == selected_neighbor)
				out_normalization_nume += target_function_at_neighbor;
			out_normalization_denom += target_function_at_neighbor;
#elif ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS
			int neighbor_M = neighbor_reservoir.M;
			if (neighbor == 0)
				if (render_data.render_settings.restir_di_settings.m_cap > 0)
					// M-capping the temporal neighbor
					neighbor_M = hippt::min(neighbor_M, render_data.render_settings.restir_di_settings.m_cap);

			if (neighbor == selected_neighbor)
				out_normalization_nume += target_function_at_neighbor * neighbor_M;
			out_normalization_denom += target_function_at_neighbor * neighbor_M;
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

	ReSTIRDIReservoir new_reservoir;
	// Surface data of the center pixel
	ReSTIRDISurface center_pixel_surface = get_pixel_surface(render_data, center_pixel_index);
	// Center pixel coordinates
	int2 center_pixel_coords = make_int2(x, y);

	int temporal_neighbor_pixel_index = find_temporal_neighbor(render_data.prev_camera, center_pixel_index);

	int selected_neighbor = 0;
	// Resampling 2 reservoirs for the temporal reuse pass: our temporal neighbor and the initial candidate
	// reservoir for our pixel (from the initial candidates pass)
	for (int neighbor = 0; neighbor < 2; neighbor++)
	{
		ReSTIRDIReservoir* input_reservoir_buffer;
		int neighbor_pixel_index;
		if (neighbor == 0)
		{
			neighbor_pixel_index = temporal_neighbor_pixel_index;
			if (neighbor_pixel_index == -1)
				// No temporal neighbor found, this is a disocclusion, nothing to temporally resample, skipping
				continue;

			// If resampling the temporal neighbor, reading from the temporal input buffer
			input_reservoir_buffer = render_data.render_settings.restir_di_settings.temporal_pass.input_reservoirs;
		}
		else
		{
			// Resampling the initial candidate at our pixel
			neighbor_pixel_index = center_pixel_index;

			// Resampling from the initial candidates buffer
			input_reservoir_buffer = render_data.render_settings.restir_di_settings.initial_candidates.output_reservoirs;
		}

		ReSTIRDIReservoir neighbor_reservoir = input_reservoir_buffer[neighbor_pixel_index];
		if (neighbor_reservoir.UCW == 0.0f)
		{
			// Nothing to do here, just take the M of the resampled neighbor into account.
			// This is basically euiqvalent to combining the reservoir with the
			// new_reservoir.combine_with() function knowing that the target function will
			// be 0.0f (because there's no neighbor reservoir sample)

			int neighbor_M = neighbor_reservoir.M;
			if (neighbor == 0)
			{
				// M-capping the temporal neighbor
				if (render_data.render_settings.restir_di_settings.m_cap > 0)
					neighbor_M = hippt::min(neighbor_M, render_data.render_settings.restir_di_settings.m_cap);
			}

			new_reservoir.M += neighbor_M;

			continue;
		}

		float target_function_at_center = 0.0f;
		if (neighbor == 1)
			// neighbor == 1 in the temporal reuse pass means that we're resampling 
			// the initial candidate at the center pixel
			// 
			// No need to evaluate the center sample at the center pixel, that's exactly
			// the target function of the center reservoir
			target_function_at_center = neighbor_reservoir.sample.target_function;
		else
			target_function_at_center = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_TargetFunctionVisibility>(render_data, neighbor_reservoir.sample, center_pixel_surface);

		float jacobian_determinant = 1.0f;
		// If the neighbor reservoir is invalid, do not compute the jacobian
		// Also, if this is the last neighbor resample (meaning that it is the sample pixel), 
		// the jacobian is going to be 1.0f so no need to compute
		if (target_function_at_center > 0.0f && neighbor_reservoir.UCW != 0.0f && neighbor == 0)
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
				int neighbor_M = neighbor_reservoir.M;
				if (render_data.render_settings.restir_di_settings.m_cap > 0)
					// M-capping the temporal neighbor
					neighbor_M = hippt::min(neighbor_M, render_data.render_settings.restir_di_settings.m_cap);

				new_reservoir.M += neighbor_M;

				// Sample too dissimilar, not resampling it
				continue;
			}
		}


		float mis_weight = 1.0f;
		if (target_function_at_center > 0.0f)
			// No need to compute the MIS weight if the target function is 0.0f because we're never going to pick
			// that sample anyway when combining the reservoir since the resampling weight will be 0.0f because of
			// the multiplication by the target function that is 0.0f
			mis_weight = get_temporal_reuse_resampling_MIS_weight(render_data, neighbor_reservoir, neighbor, center_pixel_index, temporal_neighbor_pixel_index, random_number_generator);

		// Combining as in Alg. 6 of the paper
		if (new_reservoir.combine_with(neighbor_reservoir, mis_weight, target_function_at_center, jacobian_determinant, random_number_generator))
			selected_neighbor = neighbor;
		new_reservoir.sanity_check(center_pixel_coords);
	}

	float normalization_numerator = 1.0f;
	float normalization_denominator = 1.0f;

	get_temporal_reuse_normalization_denominator_numerator(normalization_numerator, normalization_denominator, render_data, new_reservoir, selected_neighbor, center_pixel_index, temporal_neighbor_pixel_index, random_number_generator);

	new_reservoir.end_normalized(normalization_numerator, normalization_denominator);
	new_reservoir.sanity_check(center_pixel_coords);

#if ReSTIR_DI_DoVisibilityReuse && ReSTIR_DI_BiasCorrectionWeights == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
	temporal_visibility_reuse(render_data, new_reservoir, center_pixel_surface.shading_point);
#endif

	render_data.render_settings.restir_di_settings.temporal_pass.output_reservoirs[center_pixel_index] = new_reservoir;
}

#endif
