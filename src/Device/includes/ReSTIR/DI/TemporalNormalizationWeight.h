/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_NORMALIZATION_WEIGHT_H
#define DEVICE_RESTIR_DI_NORMALIZATION_WEIGHT_H

#include "Device/includes/ReSTIR/DI/Utils.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"

 // By convention, the temporal neighbor is the first one to be resampled in for loops 
 // (for looping over the neighbors when resampling / computing MIS weights)
 // So instead of hardcoding 0 everywhere in the code, we just basically give it a name
 // with a #define
#define TEMPORAL_NEIGHBOR_ID 0
// Same when resampling the initial candidates
#define INITIAL_CANDIDATES_ID 1

/**
 * This structure here is only meant to encapsulate one method that
 * returns the numerator and denominator for normalizing a reservoir at
 * the end of the temporal / spatial reuse pass.
 * 
 * This is cleaner that having a single function with a ton of 
 * 
 * #if BiasCorrectionmode == 1_OVER_M
 * #elif BiasCorrectionmode == 1_OVER_Z
 * #elif BiasCorrectionmode == MIS_LIKE
 * ....
 * 
 * We now have one structure per bias correction method one #if / #elif
 */
template <int BiasCorrectionMode, bool IsReSTIRGI>
struct ReSTIRDITemporalNormalizationWeight {};

template <bool IsReSTIRGI>
struct ReSTIRDITemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_M, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const ReSTIRDIReservoir& final_reservoir,
		int initial_candidates_M, int temporal_neighbor_M,
		float& out_normalization_nume, float& out_normalization_denom)
	{
		if (final_reservoir.weight_sum <= 0)
		{
			// Invalid reservoir, returning directly
			out_normalization_nume = 1.0f;
			out_normalization_denom = 1.0f;

			return;
		}

		// 1/M MIS weights are basically confidence weights only i.e. c_i / sum(c_j) with
		// c_i = r_i.M

		out_normalization_nume = 1.0f;
		// We're simply going to divide by the sum of all the M values of all the neighbors we resampled (including the center pixel)
		// so we're only going to set the denominator to that and the numerator isn't going to change
		out_normalization_denom = initial_candidates_M + temporal_neighbor_M;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRDITemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_Z, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& final_reservoir, 
		int initial_candidates_M, int temporal_neighbor_M,
		ReSTIRDISurface& center_pixel_surface, ReSTIRDISurface& temporal_neighbor_surface,
		float& out_normalization_nume, float& out_normalization_denom,
		Xorshift32Generator& random_number_generator)
	{
		if (final_reservoir.weight_sum <= 0)
		{
			// Invalid reservoir, returning directly
			out_normalization_nume = 1.0f;
			out_normalization_denom = 1.0f;

			return;
		}

		out_normalization_nume = 1.0f;
		// Checking how many of our neighbors could have produced the sample that we just picked
		// and we're going to divide by the sum of M values of those neighbors
		out_normalization_denom = 0.0f;

		// We're resampling from two reservoirs (the initial candidates and the temporal neighbor).
		// Either of these two reservoirs could have potentially produced the sample that we retained
		// in the 'reservoir' parameter.
		// 
		// The question is: how many neighbors could have produced that sample?
		// The sample could have been produced by a neighbor if the target function of the neighbor with
		// that sample is > so we're going to check both target function here.

		// Evaluating the target function at the center pixel because this is the pixel of the initial candidates
		float center_pixel_target_function = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir.sample, center_pixel_surface, random_number_generator);
		// if the sample contained in our final reservoir (the 'reservoir' parameter) could have been produced by the center
		// pixel, we're adding the confidence of that pixel to the denominator for normalization
		out_normalization_denom += (center_pixel_target_function > 0) * initial_candidates_M;

		if (temporal_neighbor_M > 0)
		{
			// We only want to check if the temporal could have produced the sample if we actually have a temporal neighbor
			float temporal_neighbor_target_function = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir.sample, temporal_neighbor_surface, random_number_generator);
			out_normalization_denom += (temporal_neighbor_target_function > 0) * temporal_neighbor_M;
		}
	}
};

template <bool IsReSTIRGI>
struct ReSTIRDITemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		const ReSTIRSampleType<IsReSTIRGI>& final_reservoir_sample, float final_reservoir_weight_sum,
		int initial_candidates_M, int temporal_neighbor_M,
		ReSTIRDISurface& center_pixel_surface, ReSTIRDISurface& temporal_neighbor_surface,
		int selected_neighbor,
		float& out_normalization_nume, float& out_normalization_denom,
		Xorshift32Generator& random_number_generator)
	{
		if (final_reservoir_weight_sum <= 0)
		{
			// Invalid reservoir, returning directly
			out_normalization_nume = 1.0f;
			out_normalization_denom = 1.0f;

			return;
		}

		float center_pixel_target_function;
		if constexpr (IsReSTIRGI)
			// ReSTIR GI target function
			center_pixel_target_function = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, center_pixel_surface, random_number_generator);
		else
			// ReSTIR DI target function
			center_pixel_target_function = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, center_pixel_surface, random_number_generator);

		float temporal_neighbor_target_function = 0.0f;
		if (temporal_neighbor_M > 0)
		{
			// Only evaluating the target function if we actually have a temporal neighbor because if we don't,
			// this means that no temporal neighbor contributed to the resampling of the sample in 'reservoir'
			// and if the temporal neighbor didn't contribute to the resampling, then this is not, in MIS terms,
			// a sampling technique/strategy to take into account in the MIS weight
			if constexpr (IsReSTIRGI)
				// ReSTIR GI target function
				temporal_neighbor_target_function = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, temporal_neighbor_surface, random_number_generator);
			else
				// ReSTIR DI target function
				temporal_neighbor_target_function = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, temporal_neighbor_surface, random_number_generator);
		}

		if (selected_neighbor == INITIAL_CANDIDATES_ID)
			// The point of the MIS-like MIS weights is to have the weight of the sample that we picked
			// in the numerator and the sum of everyone in the denominator.
			//
			// So if this is the sample that we picked, we're putting its target function value in the numerator
			//
			// Not multiplying by M here because this is done already during the resampling (in the resampling MIS weights)
			out_normalization_nume = center_pixel_target_function;
		else
			// Otherwise, if the sample that we picked is from the temporal neighbor, then the temporal
			// neighbor's target function is the one in the numerator
			out_normalization_nume = temporal_neighbor_target_function;

		if (!render_data.render_settings.restir_di_settings.use_confidence_weights)
		{
			// If not using confidence weights, settings the weights to 1 so that everyone has the same weight
			initial_candidates_M = 1;
			temporal_neighbor_M = 1;
		}

		out_normalization_denom = center_pixel_target_function * initial_candidates_M + temporal_neighbor_target_function * temporal_neighbor_M;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRDITemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the 
		// neighbors with balance heuristic MIS weights in the m_i terms
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRDITemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the 
		// neighbors. Everything is already in the MIS weights m_i.
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRDITemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the 
		// neighbors. Everything is already in the MIS weights m_i.
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

#endif
