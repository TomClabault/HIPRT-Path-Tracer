/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_MIS_WEIGHT_H
#define DEVICE_RESTIR_DI_MIS_WEIGHT_H

#include "Device/includes/ReSTIR/DI/Utils.h"
#include "HostDeviceCommon/KernelOptions.h"

 // By convention, the temporal neighbor is the first one to be resampled in for loops 
 // (for looping over the neighbors when resampling / computing MIS weights)
 // So instead of hardcoding 0 everywhere in the code, we just basically give it a name
 // with a #define
#define TEMPORAL_NEIGHBOR_RESAMPLE 0
// Same when resampling the initial candidates
#define INITIAL_CANDIDATES_RESAMPLE 1

/**
 * This structure here is only meant to encapsulate one method that
 * returns the resampling MIS weight used by the temporal resampling pass.
 * 
 * This whole file basically defines the functions to compute the different resampling
 * MIS weights that the renderer supports.
 * 
 * This is cleaner that having a single function with a ton of 
 * 
 * #if BiasCorrectionmode == 1_OVER_M
 * #elif BiasCorrectionmode == 1_OVER_Z
 * #elif BiasCorrectionmode == MIS_LIKE
 * ....
 * 
 * We now have one structure per MIS weight computation mode instead of one #if / #elif
 */
template <int BiasCorrectionMode>
struct ReSTIRDITemporalResamplingMISWeight {};

template<>
struct ReSTIRDITemporalResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_M>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& temporal_neighbor_reservoir,
		const ReSTIRDISurface& temporal_neighbor_surface, const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index,
		Xorshift32Generator& random_number_generator)
	{
		// 1/M MIS Weights are basically confidence weights only so we only need to return
		// the confidence of the reservoir

		return reservoir_being_resampled.M;
	}
};

template<>
struct ReSTIRDITemporalResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_Z>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& temporal_neighbor_reservoir,
		const ReSTIRDISurface& temporal_neighbor_surface, const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index,
		Xorshift32Generator& random_number_generator)
	{
		// 1/Z MIS Weights are basically confidence weights only so we only need to return
		// the confidence of the reservoir. The difference with 1/M weights is how we're going
		// to normalize the reservoir at the end of the temporal/spatial resampling pass

		return reservoir_being_resampled.M;
	}
};





template<>
struct ReSTIRDITemporalResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& temporal_neighbor_reservoir,
		const ReSTIRDISurface& temporal_neighbor_surface, const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index,
		Xorshift32Generator& random_number_generator)
	{
		// MIS-like MIS weights without confidence weights do not weight the neighbor reservoirs
		// during resampling (the same goes with any MIS weights that doesn't use confidence
		// weights). We're thus returning 1.0f.
		// 
		// The bulk of the work of the MIS-like weights is done in during the normalization of the reservoir

		return 1.0f;
	}
};

template<>
struct ReSTIRDITemporalResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& temporal_neighbor_reservoir,
		const ReSTIRDISurface& temporal_neighbor_surface, const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index,
		Xorshift32Generator& random_number_generator)
	{
		// MIS-like MIS weights with confidence weights are basically a mix of 1/Z 
		// and MIS like for the normalization so we're just returning the confidence here
		// so that a reservoir that is being resampled gets a bigger weight depending on its 
		// confidence weight (M).

		return reservoir_being_resampled.M;
	}
};





template<>
struct ReSTIRDITemporalResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& temporal_neighbor_reservoir,
		const ReSTIRDISurface& temporal_neighbor_surface, const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index,
		Xorshift32Generator& random_number_generator)
	{
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

			if (j == TEMPORAL_NEIGHBOR_RESAMPLE && temporal_neighbor_reservoir.M == 0)
				// No temporal history, no resampling to do, skipping this reservoir
				continue;

			// No confidence weight so always using M = 1
			int M = 1;

			denom += target_function_at_neighbor * M;
			if (j == current_neighbor)
				nume = target_function_at_neighbor * M;
		}

		if (denom == 0.0f)
			return 0.0f;
		else
			return nume / denom;
	}
};

template<>
struct ReSTIRDITemporalResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& temporal_neighbor_reservoir,
		const ReSTIRDISurface& temporal_neighbor_surface, const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int center_pixel_index, int temporal_neighbor_pixel_index,
		Xorshift32Generator& random_number_generator)
	{
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

			int M;
			if (j == TEMPORAL_NEIGHBOR_RESAMPLE)
				M = temporal_neighbor_reservoir.M;
			else
				// If this is not the temporal neighbor, then we're resampling the initial candidates
				// and initial candidates M value is always 1
				// 
				// TODO: this may not be true anymore if we vary the number of initial candidates 
				// (light candidates most notably) based on the surface's roughness
				M = 1;

			if (M == 0)
				// No temporal history, no resampling to do, skipping this reservoir
				continue;

			denom += target_function_at_neighbor * M;
			if (j == current_neighbor)
				nume = target_function_at_neighbor * M;
		}

		if (denom == 0.0f)
			return 0.0f;
		else
			return nume / denom;
	}
};

#endif
