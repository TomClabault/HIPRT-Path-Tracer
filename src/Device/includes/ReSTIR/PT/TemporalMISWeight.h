/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_MIS_WEIGHT_H
#define DEVICE_RESTIR_PT_MIS_WEIGHT_H

#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/PT/Utils.h"
#include "Device/includes/ReSTIR/SymmetricMISCommon.h"

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
struct ReSTIRPTTemporalResamplingMISWeight
{
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const ReSTIRPTReservoir& reservoir_being_resampled)
	{
		// 1/M MIS Weights are basically confidence weights only so we only need to return
		// the confidence of the reservoir

		return reservoir_being_resampled.confidence;
	}
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const ReSTIRPTReservoir& reservoir_being_resampled)
	{
		// 1/Z MIS Weights are basically confidence weights only so we only need to return
		// the confidence of the reservoir. The difference with 1/M weights is how we're going
		// to normalize the reservoir at the end of the temporal/spatial resampling pass

		return reservoir_being_resampled.confidence;
	}
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_MIS_LIKE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data, const ReSTIRPTReservoir& reservoir_being_resampled)
	{
		// MIS-like MIS weights with confidence weights are basically a mix of 1/Z
		// and MIS like for the normalization so we're just returning the confidence here
		// so that a reservoir that is being resampled gets a bigger weight depending on its
		// confidence weight (M).

		return reservoir_being_resampled.confidence;
	}
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_MIS_GBH>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,

													  const ReSTIRPTReservoirSample& reservoir_being_resampled_sample,
													  float initial_candidates_reservoir_confidence,

													  ReSTIRSurface& temporal_neighbor_surface,
													  ReSTIRSurface& center_pixel_surface,
													  int temporal_neighbor_reservoir_confidence,
													  int current_neighbor_index,
													  Xorshift32Generator& random_number_generator)
	{
		float nume = 0.0f;
		// We already have the target function at the center pixel, adding it to the denom
		float denom = 0.0f;

		// Evaluating the sample that we're resampling at the neighor locations (using the neighbors surfaces)
		float target_function_at_temporal_neighbor = 0.0f;
		if (temporal_neighbor_reservoir_confidence != 0)
		{
			// Only computing the target function if we do have a temporal neighbor

			target_function_at_temporal_neighbor =
				ReSTIR_PT_evaluate_target_function<true>(render_data, reservoir_being_resampled_sample, temporal_neighbor_surface, random_number_generator);
		}

		if (current_neighbor_index == TEMPORAL_NEIGHBOR_ID && target_function_at_temporal_neighbor == 0.0f)
			// If we're currently computing the MIS weight for the temporal neighbor,
			// this means that we're going to have the temporal neighbor weight
			// (target function) in the numerator. But if the target function
			// at the temporal neighbor is 0.0f, then we're going to have 0.0f
			// in the numerator --> 0.0f MIS weight anyways --> no need to
			// compute anything else, we can already return 0.0f for the MIS weight.
			return 0.0f;

		float target_function_at_center =
			ReSTIR_PT_evaluate_target_function<true>(render_data, reservoir_being_resampled_sample, center_pixel_surface, random_number_generator);

		int temporal_confidence			= temporal_neighbor_reservoir_confidence;
		int center_reservoir_confidence = initial_candidates_reservoir_confidence;

		if (current_neighbor_index == TEMPORAL_NEIGHBOR_ID)
			nume = target_function_at_temporal_neighbor * temporal_confidence;
		else
			nume = target_function_at_center * center_reservoir_confidence;

		denom = target_function_at_temporal_neighbor * temporal_confidence + target_function_at_center * center_reservoir_confidence;

		if (denom == 0.0f)
			return 0.0f;
		else
			return nume / denom;
	}
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS>
{
	static constexpr float NO_NEIGHBOR_RESAMPLING = -1.0f;

	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
													  ReSTIRPTReservoir& temporal_neighbor_reservoir,
													  ReSTIRPTReservoir& initial_candidates_reservoir,
													  ReSTIRSurface& center_pixel_surface,
													  ReSTIRSurface& temporal_neighbor_surface,

													  float neighbor_sample_target_function_at_center,
													  int current_neighbor_index,

													  Xorshift32Generator& random_number_generator)
	{
		if (current_neighbor_index == TEMPORAL_NEIGHBOR_ID)
		{
			// Setting to 0.0f to remove the 'NO_NEIGHBOR_RESAMPLING' value
			mc = 0.0f;

			// Resampling the temporal neighbor

			float target_function_at_neighbor = temporal_neighbor_reservoir.sample.target_function;
			float target_function_at_center	  = neighbor_sample_target_function_at_center;

			float temporal_neighbor_confidence = temporal_neighbor_reservoir.confidence;
			float center_reservoir_confidence  = initial_candidates_reservoir.confidence;
			float neighbors_confidence_sum	   = temporal_neighbor_confidence;

			float nume	= target_function_at_neighbor * temporal_neighbor_confidence;
			float denom = target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center * center_reservoir_confidence;
			float mi	= denom == 0.0f ? 0.0f : (nume / denom);

			float target_function_center_sample_at_neighbor =
				ReSTIR_PT_evaluate_target_function<true>(render_data, initial_candidates_reservoir.sample, temporal_neighbor_surface, random_number_generator);

			// Because we're using the target function as a PDF here, we need to scale the PDF
			// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

			// Only doing this if we at least have a target function to scale by the jacobian
			if (target_function_center_sample_at_neighbor > 0.0f)
			{
				// If this is an envmap path the jacobian is just 1 so this is not needed
				if (!initial_candidates_reservoir.sample.is_envmap_path())
				{
					float jacobian = get_jacobian_determinant_reconnection_shift(
						initial_candidates_reservoir.sample.rc_vertex, initial_candidates_reservoir.sample.rc_vertex_geometric_normal.unpack(),
						temporal_neighbor_surface.shading_point, center_pixel_surface.shading_point,
						render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

					if (jacobian == 0.0f)
						// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
						// and has been rejected), the target function is set to 0
						target_function_center_sample_at_neighbor = 0.0f;
					else
						target_function_center_sample_at_neighbor *= jacobian;
				}
			}

			float target_function_center_sample_at_center = initial_candidates_reservoir.sample.target_function;

			float nume_mc = target_function_center_sample_at_center * center_reservoir_confidence;
			float denom_mc =
				target_function_center_sample_at_neighbor * neighbors_confidence_sum + target_function_center_sample_at_center * center_reservoir_confidence;

			float confidence_multiplier = temporal_neighbor_confidence / neighbors_confidence_sum;

			if (denom_mc != 0.0f)
				mc += nume_mc / denom_mc * confidence_multiplier;

			return mi;
		}
		else
		{
			// Resampling the center pixel (initial candidates)

			if (mc == NO_NEIGHBOR_RESAMPLING)
				// If there was no neighbor resampling (and mc hasn't been accumulated),
				// then the MIS weight should be 1 for the center pixel. It gets all the weight
				// since no neighbor was resampled
				return 1.0f;
			else if (mc == 0.0f)
				return 0.0f;
			else
				// Returning the weight accumulated so far when resampling the neighbors.
				//
				// !!! This assumes that the center pixel is resampled last (which it is in this ReSTIR implementation) !!!
				return mc;
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = NO_NEIGHBOR_RESAMPLING;
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
													  ReSTIRPTReservoir& temporal_neighbor_reservoir,
													  ReSTIRPTReservoir& initial_candidates_reservoir,
													  ReSTIRSurface& center_pixel_surface,
													  ReSTIRSurface& temporal_neighbor_surface,

													  float neighbor_sample_target_function_at_center,
													  int current_neighbor_index,

													  Xorshift32Generator& random_number_generator)
	{
		if (current_neighbor_index == TEMPORAL_NEIGHBOR_ID)
		{
			// Resampling the temporal neighbor

			float target_function_at_neighbor = temporal_neighbor_reservoir.sample.target_function;
			float target_function_at_center	  = neighbor_sample_target_function_at_center;

			float temporal_neighbor_confidence = temporal_neighbor_reservoir.confidence;
			float center_reservoir_confidence  = initial_candidates_reservoir.confidence;
			float neighbors_confidence_sum	   = temporal_neighbor_confidence;

			float nume	= target_function_at_neighbor * temporal_neighbor_confidence;
			float denom = target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center * center_reservoir_confidence;
			float mi	= denom == 0.0f ? 0.0f : (nume / denom);
			// Eq 7.8
			mi *= neighbors_confidence_sum / (neighbors_confidence_sum + center_reservoir_confidence);

			float target_function_center_sample_at_neighbor =
				ReSTIR_PT_evaluate_target_function<true>(render_data, initial_candidates_reservoir.sample, temporal_neighbor_surface, random_number_generator);

			// Because we're using the target function as a PDF here, we need to scale the PDF
			// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

			// Only doing this if we at least have a target function to scale by the jacobian
			if (target_function_center_sample_at_neighbor > 0.0f)
			{
				// If this is an envmap path the jacobian is just 1 so this is not needed
				if (!initial_candidates_reservoir.sample.is_envmap_path())
				{
					float jacobian = get_jacobian_determinant_reconnection_shift(
						initial_candidates_reservoir.sample.rc_vertex, initial_candidates_reservoir.sample.rc_vertex_geometric_normal.unpack(),
						temporal_neighbor_surface.shading_point, center_pixel_surface.shading_point,
						render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

					if (jacobian == 0.0f)
						// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
						// and has been rejected), the target function is set to 0
						target_function_center_sample_at_neighbor = 0.0f;
					else
						target_function_center_sample_at_neighbor *= jacobian;
				}
			}

			float target_function_center_sample_at_center = initial_candidates_reservoir.sample.target_function;

			float nume_mc = target_function_center_sample_at_center * center_reservoir_confidence;
			float denom_mc =
				target_function_center_sample_at_neighbor * neighbors_confidence_sum + target_function_center_sample_at_center * center_reservoir_confidence;
			float confidence_multiplier = neighbors_confidence_sum / (neighbors_confidence_sum + center_reservoir_confidence);

			if (denom_mc != 0.0f)
				mc += nume_mc / denom_mc * confidence_multiplier;

			return mi;
		}
		else
		{
			// Resampling the center pixel (initial candidates)

			if (mc == 0.0f)
				// If there was no neighbor resampling (and mc hasn't been accumulated),
				// then the MIS weight should be 1 for the center pixel. It gets all the weight
				// since no neighbor was resampled
				return 1.0f;
			else
			{
				// Returning the weight accumulated so far when resampling the neighbors.
				//
				// !!! This assumes that the center pixel is resampled last (which it is in this ReSTIR implementation) !!!

				// In the defensive formulation, we want to divide by M, not M-1.
				// (Eq. 7.6 of "A Gentle Introduction to ReSTIR")
				return mc + static_cast<float>(initial_candidates_reservoir.confidence) /
								static_cast<float>(initial_candidates_reservoir.confidence + temporal_neighbor_reservoir.confidence);
			}
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = 0.0f;
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
													  ReSTIRPTReservoir& temporal_neighbor_reservoir,
													  ReSTIRPTReservoir& initial_candidates_reservoir,
													  ReSTIRSurface& center_pixel_surface,
													  ReSTIRSurface& temporal_neighbor_surface,

													  float neighbor_sample_target_function_at_center,
													  int current_neighbor_index,

													  Xorshift32Generator& random_number_generator)
	{
		if (current_neighbor_index == TEMPORAL_NEIGHBOR_ID)
		{
			// Resampling the temporal neighbor

			float target_function_neighbor_sample_at_neighbor = temporal_neighbor_reservoir.sample.target_function;
			float target_function_neighbor_sample_at_center	  = neighbor_sample_target_function_at_center;

			float temporal_neighbor_confidence = temporal_neighbor_reservoir.confidence;
			float center_reservoir_confidence  = initial_candidates_reservoir.confidence;
			float neighbors_confidence_sum	   = temporal_neighbor_confidence;

			// Eq. 15 of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, 2024] generalized
			// with confidence weights
			float difference_function =
				symmetric_ratio_MIS_weights_difference_function(target_function_neighbor_sample_at_center, target_function_neighbor_sample_at_neighbor,
																render_data.render_settings.restir_pt_settings.symmetric_ratio_mis_weights_beta_exponent);
			float nume_mi  = difference_function * temporal_neighbor_confidence;
			float denom_mi = center_reservoir_confidence + neighbors_confidence_sum * difference_function;
			float mi	   = nume_mi / denom_mi;

			float target_function_center_sample_at_neighbor =
				ReSTIR_PT_evaluate_target_function<true>(render_data, initial_candidates_reservoir.sample, temporal_neighbor_surface, random_number_generator);

			// Because we're using the target function as a PDF here, we need to scale the PDF
			// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

			// Only doing this if we at least have a target function to scale by the jacobian
			if (target_function_center_sample_at_neighbor > 0.0f)
			{
				// If this is an envmap path the jacobian is just 1 so this is not needed
				if (!initial_candidates_reservoir.sample.is_envmap_path())
				{
					float jacobian = get_jacobian_determinant_reconnection_shift(
						initial_candidates_reservoir.sample.rc_vertex, initial_candidates_reservoir.sample.rc_vertex_geometric_normal.unpack(),
						temporal_neighbor_surface.shading_point, center_pixel_surface.shading_point,
						render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

					if (jacobian == 0.0f)
						// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
						// and has been rejected), the target function is set to 0
						target_function_center_sample_at_neighbor = 0.0f;
					else
						target_function_center_sample_at_neighbor *= jacobian;
				}
			}

			float target_function_center_sample_at_center = initial_candidates_reservoir.sample.target_function;

			float nume_mc  = center_reservoir_confidence;
			float denom_mc = center_reservoir_confidence +
							 neighbors_confidence_sum * symmetric_ratio_MIS_weights_difference_function(
															target_function_center_sample_at_neighbor, target_function_center_sample_at_center,
															render_data.render_settings.restir_pt_settings.symmetric_ratio_mis_weights_beta_exponent);

			float confidence_weights_multiplier;
			if (neighbors_confidence_sum == 0.0f)
				confidence_weights_multiplier = 0.0f;
			else
				confidence_weights_multiplier = temporal_neighbor_confidence / neighbors_confidence_sum;

			mc += confidence_weights_multiplier * nume_mc / denom_mc;

			return mi;
		}
		else
		{
			// Resampling the center pixel (initial candidates)

			if (mc == 0.0f)
				// If there was no neighbor resampling (and mc hasn't been accumulated),
				// then the MIS weight should be 1 for the center pixel. It gets all the weight
				// since no neighbor was resampled
				return 1.0f;
			else
				// Returning the weight accumulated so far when resampling the neighbors.
				//
				// !!! This assumes that the center pixel is resampled last (which it is in this ReSTIR implementation) !!!
				return mc;
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = 0.0f;
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
													  ReSTIRPTReservoir& temporal_neighbor_reservoir,
													  ReSTIRPTReservoir& initial_candidates_reservoir,
													  ReSTIRSurface& center_pixel_surface,
													  ReSTIRSurface& temporal_neighbor_surface,

													  float neighbor_sample_target_function_at_center,
													  int current_neighbor_index,

													  Xorshift32Generator& random_number_generator)
	{
		if (current_neighbor_index == TEMPORAL_NEIGHBOR_ID)
		{
			// Resampling a neighbor

			float target_function_neighbor_sample_at_neighbor = temporal_neighbor_reservoir.sample.target_function;
			float target_function_center_sample_at_center	  = initial_candidates_reservoir.sample.target_function;

			float temporal_neighbor_confidence = temporal_neighbor_reservoir.confidence;
			float center_reservoir_confidence  = initial_candidates_reservoir.confidence;
			float neighbors_confidence_sum	   = temporal_neighbor_confidence;

			// Eq. 15 of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, 2024] generalized
			// with confidence weights
			float difference_function =
				symmetric_ratio_MIS_weights_difference_function(neighbor_sample_target_function_at_center, target_function_neighbor_sample_at_neighbor,
																render_data.render_settings.restir_pt_settings.symmetric_ratio_mis_weights_beta_exponent);
			float nume_mi, denom_mi;

			// Eq. 16 of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, 2024] generalized
			// with confidence weights
			if (neighbor_sample_target_function_at_center <= target_function_neighbor_sample_at_neighbor)
			{
				nume_mi	 = difference_function * temporal_neighbor_confidence;
				denom_mi = center_reservoir_confidence + neighbors_confidence_sum * difference_function;
			}
			else
			{
				nume_mi	 = difference_function * temporal_neighbor_confidence;
				denom_mi = center_reservoir_confidence + neighbors_confidence_sum;
			}

			float mi = nume_mi / denom_mi;

			float target_function_center_sample_at_neighbor =
				ReSTIR_PT_evaluate_target_function<true>(render_data, initial_candidates_reservoir.sample, temporal_neighbor_surface, random_number_generator);

			// Because we're using the target function as a PDF here, we need to scale the PDF
			// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

			// Only doing this if we at least have a target function to scale by the jacobian
			if (target_function_center_sample_at_neighbor > 0.0f)
			{
				// If this is an envmap path the jacobian is just 1 so this is not needed
				if (!initial_candidates_reservoir.sample.is_envmap_path())
				{
					float jacobian = get_jacobian_determinant_reconnection_shift(
						initial_candidates_reservoir.sample.rc_vertex, initial_candidates_reservoir.sample.rc_vertex_geometric_normal.unpack(),
						temporal_neighbor_surface.shading_point, center_pixel_surface.shading_point,
						render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

					if (jacobian == 0.0f)
						// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
						// and has been rejected), the target function is set to 0
						target_function_center_sample_at_neighbor = 0.0f;
					else
						target_function_center_sample_at_neighbor *= jacobian;
				}
			}

			float nume_mc, denom_mc;

			float difference_function_mc =
				symmetric_ratio_MIS_weights_difference_function(target_function_center_sample_at_neighbor, target_function_center_sample_at_center,
																render_data.render_settings.restir_pt_settings.symmetric_ratio_mis_weights_beta_exponent);
			if (target_function_center_sample_at_center <= target_function_center_sample_at_neighbor)
			{
				nume_mc	 = difference_function_mc * temporal_neighbor_confidence;
				denom_mc = center_reservoir_confidence + neighbors_confidence_sum * difference_function_mc;
			}
			else
			{
				nume_mc	 = difference_function_mc * temporal_neighbor_confidence;
				denom_mc = center_reservoir_confidence + neighbors_confidence_sum;
			}

			mc += nume_mc / denom_mc;

			return mi;
		}
		else
		{
			// Resampling the center pixel

			if (mc == 0.0f)
				// If there was no neighbor resampling (and mc hasn't been accumulated),
				// then the MIS weight should be 1 for the center pixel. It gets all the weight
				// since no neighbor was resampled
				return 1.0f;
			else
				// Returning the weight accumulated so far when resampling the neighbors.
				//
				// !!! This assumes that the center pixel is resampled last (which it is in this ReSTIR implementation) !!!
				//
				// This is Eq. 16 of the paper: y not in R: m_i(y) = 1 - Sum(...) / |R|
				// mc here is the sum
				// and |R| is 1
				return 1.0f - mc;
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = 0.0f;
};

// For temporal reuse, we're not using SPMIS (we're not importance sampling the temporal neighbor) so we just use traditional pairwise MIS weights
template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS>
	: public ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS>
{
};

template <>
struct ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE>
	: public ReSTIRPTTemporalResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE>
{
};

#endif
