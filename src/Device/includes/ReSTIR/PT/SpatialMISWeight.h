/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_MIS_WEIGHT_H
#define DEVICE_RESTIR_PT_SPATIAL_MIS_WEIGHT_H

#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/PT/Utils.h"
#include "Device/includes/ReSTIR/SymmetricMISCommon.h"
#include "Device/includes/TriangleLoadUtils.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"

template <int BiasCorrectionMode>
struct ReSTIRPTSpatialResamplingMISWeight
{
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(int reservoir_being_resampled_M)
	{
		return reservoir_being_resampled_M;
	}
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(int reservoir_being_resampled_M)
	{
		return reservoir_being_resampled_M;
	}
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_MIS_LIKE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data, int reservoir_being_resampled_M)
	{
		return render_data.render_settings.restir_pt_settings.use_confidence_weights ? reservoir_being_resampled_M : 1;
	}
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_MIS_GBH>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,

													  float reservoir_being_resampled_UCW,
													  const ReSTIRPTReservoirSample& reservoir_being_resampled_sample,

													  const ReSTIRSurface& center_pixel_surface,
													  int current_neighbor_index,
													  int2_t center_pixel_coords,
													  Xorshift32Generator& random_number_generator)
	{
		if (reservoir_being_resampled_UCW <= 0.0f)
			// Reservoir that doesn't contain any sample, returning
			// 0.0f MIS weight beacuse there's no point resampling that one
			return 0.0f;

		float nume	= 0.0f;
		float denom = 0.0f;

		unsigned int backup_seed = random_number_generator.m_state.seed;

		random_number_generator.m_state.seed = render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_neighbors_rng_seed;

		for (int j = 0; j < render_data.render_settings.restir_pt_settings.common_spatial_pass.reuse_neighbor_count + 1; j++)
		{
			int neighbor_index_j = get_spatial_neighbor_pixel_index<ReSTIR_VARIANT_PT>(render_data, j, center_pixel_coords, random_number_generator);
			if (neighbor_index_j == -1)
				// Invalid neighbor, skipping
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
			if (!check_neighbor_similarity_heuristics<ReSTIR_VARIANT_PT>(
					render_data, neighbor_index_j, center_pixel_index, center_pixel_surface.shading_point,
					ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<ReSTIR_VARIANT_PT>(render_data, center_pixel_surface)))
				// Neighbor too dissimilar according to heuristics, skipping
				continue;

			ReSTIRSurface neighbor_surface = get_pixel_surface(render_data, neighbor_index_j, random_number_generator);

			float target_function_at_j;
			if (j == current_neighbor_index)
				target_function_at_j =
					ReSTIR_PT_evaluate_target_function<true>(render_data, reservoir_being_resampled_sample, neighbor_surface, random_number_generator);
			else
				target_function_at_j =
					ReSTIR_PT_evaluate_target_function<true>(render_data, reservoir_being_resampled_sample, neighbor_surface, random_number_generator);

			if (!reservoir_being_resampled_sample.is_envmap_path())
				// Applying the jacobian to get "p_hat_from_i"
				target_function_at_j *=
					hippt::max(0.0f, get_jacobian_determinant_reconnection_shift(
										 reservoir_being_resampled_sample.rc_vertex, reservoir_being_resampled_sample.rc_vertex_geometric_normal.unpack(),
										 center_pixel_surface.shading_point, neighbor_surface.shading_point,
										 render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold()));

			int M = 1;
			if (render_data.render_settings.restir_pt_settings.use_confidence_weights)
				M = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs[neighbor_index_j].M;
			denom += target_function_at_j * M;
			if (j == current_neighbor_index)
				nume = target_function_at_j * M;
		}

		random_number_generator.m_state.seed = backup_seed;

		if (denom == 0.0f)
			return 0.0f;
		else
			return nume / denom;
	}
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,

													  int reservoir_being_resampled_M,
													  float reservoir_being_resampled_target_function,
													  ReSTIRPTReservoirSample& center_pixel_reservoir_sample,
													  int center_pixel_reservoir_M,
													  float center_pixel_reservoir_target_function,

													  ReSTIRSurface& center_pixel_surface,
													  float target_function_at_center,
													  int neighbor_pixel_index,
													  int valid_neighbors_count,
													  int valid_neighbors_M_sum,
													  bool update_mc,
													  bool resampling_canonical,
													  Xorshift32Generator& random_number_generator)
	{
		if (!resampling_canonical)
		{
			// Resampling a neighbor

			// The target function of the neighbor reservoir's sample at the neighbor surface is just
			// the target function stored in the neighbor's reservoir.
			//
			// Care must be taken however because this is not necessarily true anymore after multiple spatial
			// reuse passes: a given pixel may now hold a sample from another pixel and that means that the visibility
			// doesn't match anymore.
			//
			// However, this ReSTIR implementation does a visibility reuse pass at the end of each spatial reuse pass
			// so that we know that the visibility is correct and thus we do not run into any issues and we can just
			// reuse the target function stored in the neighbor's reservoir
			float target_function_at_neighbor			  = reservoir_being_resampled_target_function;
			float target_function_center_sample_at_center = center_pixel_reservoir_target_function;

			bool use_confidence_weights	   = ReSTIRSettingsHelper::get_restir_settings<ReSTIR_VARIANT_PT>(render_data).use_confidence_weights;
			float reservoir_resampled_M	   = use_confidence_weights ? reservoir_being_resampled_M : 1;
			float center_reservoir_M	   = use_confidence_weights ? center_pixel_reservoir_M : 1;
			float neighbors_confidence_sum = use_confidence_weights ? valid_neighbors_M_sum : 1;
			// We only want to divide by M-1 if we're not using confidence weights.
			// (Eq. 7.6 and 7.7 of "A Gentle Introduction to ReSTIR")
			float valid_neighbor_division_term = use_confidence_weights ? 1 : valid_neighbors_count;

			float nume = target_function_at_neighbor * reservoir_resampled_M;
			float denom =
				target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center / valid_neighbor_division_term * center_reservoir_M;
			float mi = denom == 0.0f ? 0.0f : (nume / denom);

			if (update_mc)
			{
				ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);
				float target_function_center_sample_at_neighbor =
					ReSTIR_PT_evaluate_target_function<true>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

				// Because we're using the target function as a PDF here, we need to scale the PDF
				// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

				// Only doing this if we at least have a target function to scale by the jacobian
				if (target_function_center_sample_at_neighbor > 0.0f)
				{
					// If this is an envmap path the jacobian is just 1 so this is not needed
					if (!center_pixel_reservoir_sample.is_envmap_path())
					{
						float jacobian = get_jacobian_determinant_reconnection_shift(
							center_pixel_reservoir_sample.rc_vertex, center_pixel_reservoir_sample.rc_vertex_geometric_normal.unpack(),
							neighbor_pixel_surface.shading_point, center_pixel_surface.shading_point,
							render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

						if (jacobian == 0.0f)
							// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
							// and has been rejected), the target function is set to 0
							target_function_center_sample_at_neighbor = 0.0f;
						else
							target_function_center_sample_at_neighbor *= jacobian;
					}
				}

				float nume_mc  = target_function_center_sample_at_center / valid_neighbor_division_term * center_reservoir_M;
				float denom_mc = target_function_center_sample_at_neighbor * neighbors_confidence_sum +
								 target_function_center_sample_at_center / valid_neighbor_division_term * center_reservoir_M;

				float confidence_weights_multiplier;
				if (use_confidence_weights)
				{
					if (neighbors_confidence_sum == 0.0f)
						confidence_weights_multiplier = 0.0f;
					else
						confidence_weights_multiplier = reservoir_resampled_M / neighbors_confidence_sum;
				}
				else
					confidence_weights_multiplier = 1.0f;

				// (Eq. 7.7 of "A Gentle Introduction to ReSTIR"), c_j / (Sum_{k!=c}^M c_k)
				if (denom_mc != 0.0f)
					mc += nume_mc / denom_mc / valid_neighbor_division_term * confidence_weights_multiplier;
			}

			return mi / valid_neighbor_division_term;
		}
		else
		{
			// Resampling the center pixel
			if (mc == 0.0f)
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
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,

													  int reservoir_being_resampled_M,
													  float reservoir_being_resampled_target_function,
													  ReSTIRPTReservoirSample& center_pixel_reservoir_sample,
													  int center_pixel_reservoir_M,
													  float center_pixel_reservoir_target_function,

													  ReSTIRSurface& center_pixel_surface,
													  float target_function_at_center,
													  int neighbor_pixel_index,
													  int valid_neighbors_count,
													  int valid_neighbors_M_sum,
													  bool update_mc,
													  bool resampling_canonical,
													  Xorshift32Generator& random_number_generator)
	{
		if (!resampling_canonical)
		{
			// Resampling a neighbor

			// The target function of the neighbor reservoir's sample at the neighbor surface is just
			// the target function stored in the neighbor's reservoir.
			//
			// Care must be taken however because this is not necessarily true anymore after multiple spatial
			// reuse passes: a given pixel may now hold a sample from another pixel and that means that the visibility
			// doesn't match anymore.
			//
			// However, this ReSTIR DI implementation does a visibility reuse pass at the end of each spatial reuse pass
			// so that we know that the visibility is correct and thus we do not run into any issues and we can just
			// reuse the target function stored in the neighbor's reservoir
			float target_function_at_neighbor = reservoir_being_resampled_target_function;

			bool use_confidence_weights	   = render_data.render_settings.restir_pt_settings.use_confidence_weights;
			float reservoir_resampled_M	   = use_confidence_weights ? reservoir_being_resampled_M : 1;
			float center_reservoir_M	   = use_confidence_weights ? center_pixel_reservoir_M : 1;
			float neighbors_confidence_sum = use_confidence_weights ? valid_neighbors_M_sum : 1;
			// We only want to divide by M-1 if we're not using confidence weights.
			// (Eq. 7.6 and 7.7 of "A Gentle Introduction to ReSTIR")
			float valid_neighbor_division_term = use_confidence_weights ? 1 : valid_neighbors_count;

			float nume = target_function_at_neighbor * reservoir_resampled_M;
			float denom =
				target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center / valid_neighbor_division_term * center_reservoir_M;
			float mi = denom == 0.0f ? 0.0f : (nume / denom);
			if (use_confidence_weights)
				mi *= neighbors_confidence_sum / (neighbors_confidence_sum + center_reservoir_M);

			if (update_mc)
			{
				// There's one case where we do not need to update 'mc': when the center pixel (that we're currently resampling) is empty: M = 0 / UCW = 0
				// That's because in such cases, the empty reservoir will not be resampled into the final reservoir anyways since it has no contribution
				// Because 'mc' is only used as the MIS weight of the center reservoir, we don't care about 'mc' since the center reservoir is not going
				// to be chosen anyways
				//
				// So we can avoid computing all that stuff

				ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

				float target_function_center_sample_at_neighbor =
					ReSTIR_PT_evaluate_target_function<true>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

				// Because we're using the target function as a PDF here, we need to scale the PDF
				// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

				// Only doing this if we at least have a target function to scale by the jacobian
				if (target_function_center_sample_at_neighbor > 0.0f)
				{
					if (!center_pixel_reservoir_sample.is_envmap_path())
					{
						// If this is an envmap path the jacobian is just 1 so this is not needed

						float jacobian = get_jacobian_determinant_reconnection_shift(
							center_pixel_reservoir_sample.rc_vertex, center_pixel_reservoir_sample.rc_vertex_geometric_normal.unpack(),
							neighbor_pixel_surface.shading_point, center_pixel_surface.shading_point,
							render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());
						if (jacobian == 0.0f)
							// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
							// and has been rejected), the target function is set to 0
							target_function_center_sample_at_neighbor = 0.0f;
						else
							target_function_center_sample_at_neighbor *= jacobian;
					}
				}

				float target_function_center_sample_at_center = center_pixel_reservoir_target_function;

				float nume_mc  = target_function_center_sample_at_center / valid_neighbor_division_term * center_reservoir_M;
				float denom_mc = target_function_center_sample_at_neighbor * neighbors_confidence_sum +
								 target_function_center_sample_at_center / valid_neighbor_division_term * center_reservoir_M;
				float confidence_multiplier = 1.0f;
				if (use_confidence_weights)
					confidence_multiplier = reservoir_resampled_M / (center_reservoir_M + neighbors_confidence_sum);
				if (denom_mc != 0.0f)
					mc += nume_mc / denom_mc * confidence_multiplier;
			}

			if (use_confidence_weights)
				return mi;
			else
				// In the defensive formulation, we want to divide by M, not M-1.
				// (Eq. 7.6 of "A Gentle Introduction to ReSTIR")
				//
				// We also only want that division when not using confidence weights
				return mi / (valid_neighbors_count + 1.0f);
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
			{
				// Returning the weight accumulated so far when resampling the neighbors.
				//
				// !!! This assumes that the center pixel is resampled last (which it is in this ReSTIR implementation) !!!

				if (render_data.render_settings.restir_pt_settings.use_confidence_weights)
					return mc + static_cast<float>(center_pixel_reservoir_M) / static_cast<float>(center_pixel_reservoir_M + valid_neighbors_M_sum);
				else
					// In the defensive formulation, we want to divide by M, not M-1.
					// (Eq. 7.6 of "A Gentle Introduction to ReSTIR") so 'valid_neighbors_count + 1'
					return (1.0f + mc) / (valid_neighbors_count + 1.0f);
			}
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = 0.0f;
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
													  int reservoir_being_resampled_M,
													  float reservoir_being_resampled_target_function,
													  ReSTIRPTReservoirSample& center_pixel_reservoir_sample,
													  int center_pixel_reservoir_M,
													  float center_pixel_reservoir_target_function,

													  ReSTIRSurface& center_pixel_surface,
													  float target_function_neighbor_sample_at_center,
													  int neighbor_pixel_index,
													  int valid_neighbors_count,
													  int valid_neighbors_M_sum,
													  bool update_mc,
													  bool resampling_canonical,
													  Xorshift32Generator& random_number_generator)
	{
		if (!resampling_canonical)
		{
			// Resampling a neighbor

			float target_function_neighbor_sample_at_neighbor = reservoir_being_resampled_target_function;
			float target_function_center_sample_at_center	  = center_pixel_reservoir_target_function;

			bool use_confidence_weights	   = render_data.render_settings.restir_pt_settings.use_confidence_weights;
			float reservoir_resampled_M	   = use_confidence_weights ? reservoir_being_resampled_M : 1;
			float center_reservoir_M	   = use_confidence_weights ? center_pixel_reservoir_M : 1;
			float neighbors_confidence_sum = use_confidence_weights ? valid_neighbors_M_sum : valid_neighbors_count;

			// Eq. 15 of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, 2024] generalized
			// with confidence weights
			float difference_function =
				symmetric_ratio_MIS_weights_difference_function(target_function_neighbor_sample_at_center, target_function_neighbor_sample_at_neighbor,
																render_data.render_settings.restir_pt_settings.symmetric_ratio_mis_weights_beta_exponent);
			float nume_mi  = difference_function * reservoir_resampled_M;
			float denom_mi = center_reservoir_M + neighbors_confidence_sum * difference_function;
			float mi	   = nume_mi / denom_mi;

			if (update_mc)
			{
				ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

				float target_function_center_sample_at_neighbor =
					ReSTIR_PT_evaluate_target_function<true>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

				// Because we're using the target function as a PDF here, we need to scale the PDF
				// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

				// Only doing this if we at least have a target function to scale by the jacobian
				if (target_function_center_sample_at_neighbor > 0.0f)
				{
					// If this is an envmap path the jacobian is just 1 so this is not needed
					if (!center_pixel_reservoir_sample.is_envmap_path())
					{
						float jacobian = get_jacobian_determinant_reconnection_shift(
							center_pixel_reservoir_sample.rc_vertex, center_pixel_reservoir_sample.rc_vertex_geometric_normal.unpack(),
							neighbor_pixel_surface.shading_point, center_pixel_surface.shading_point,
							render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

						if (jacobian == 0.0f)
							// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
							// and has been rejected), the target function is set to 0
							target_function_center_sample_at_neighbor = 0.0f;
						else
							target_function_center_sample_at_neighbor *= jacobian;
					}
				}

				float nume_mc  = center_reservoir_M;
				float denom_mc = center_reservoir_M +
								 neighbors_confidence_sum * symmetric_ratio_MIS_weights_difference_function(
																target_function_center_sample_at_neighbor, target_function_center_sample_at_center,
																render_data.render_settings.restir_pt_settings.symmetric_ratio_mis_weights_beta_exponent);

				float confidence_weights_multiplier;
				if (use_confidence_weights)
				{
					if (neighbors_confidence_sum == 0.0f)
						confidence_weights_multiplier = 0.0f;
					else
						confidence_weights_multiplier = reservoir_resampled_M / neighbors_confidence_sum;
				}
				else
					confidence_weights_multiplier = 1.0f / valid_neighbors_count;

				mc += confidence_weights_multiplier * nume_mc / denom_mc;
			}

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
				return mc;
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = 0.0f;
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
													  int reservoir_being_resampled_M,
													  float reservoir_being_resampled_target_function,
													  ReSTIRPTReservoirSample& center_pixel_reservoir_sample,
													  int center_pixel_reservoir_M,
													  float center_pixel_reservoir_target_function,

													  ReSTIRSurface& center_pixel_surface,
													  float target_function_neighbor_sample_at_center,
													  int neighbor_pixel_index,
													  int valid_neighbors_count,
													  int valid_neighbors_M_sum,
													  bool update_mc,
													  bool resampling_canonical,
													  Xorshift32Generator& random_number_generator)
	{
		if (!resampling_canonical)
		{
			// Resampling a neighbor

			float target_function_neighbor_sample_at_neighbor = reservoir_being_resampled_target_function;
			float target_function_center_sample_at_center	  = center_pixel_reservoir_target_function;

			bool use_confidence_weights	   = render_data.render_settings.restir_pt_settings.use_confidence_weights;
			float reservoir_resampled_M	   = use_confidence_weights ? reservoir_being_resampled_M : 1;
			float center_reservoir_M	   = use_confidence_weights ? center_pixel_reservoir_M : 1;
			float neighbors_confidence_sum = use_confidence_weights ? valid_neighbors_M_sum : valid_neighbors_count;

			// Eq. 16 of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, 2024] generalized
			// with confidence weights
			float difference_function =
				symmetric_ratio_MIS_weights_difference_function(target_function_neighbor_sample_at_center, target_function_neighbor_sample_at_neighbor,
																render_data.render_settings.restir_pt_settings.symmetric_ratio_mis_weights_beta_exponent);
			float nume_mi, denom_mi;

			// Eq. 16 of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, 2024] generalized
			// with confidence weights
			if (target_function_neighbor_sample_at_center <= target_function_neighbor_sample_at_neighbor)
			{
				nume_mi	 = difference_function * reservoir_resampled_M;
				denom_mi = center_reservoir_M + neighbors_confidence_sum * difference_function;
			}
			else
			{
				nume_mi	 = difference_function * reservoir_resampled_M;
				denom_mi = center_reservoir_M + neighbors_confidence_sum;
			}

			float mi = nume_mi / denom_mi;

			if (update_mc)
			{
				ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

				float target_function_center_sample_at_neighbor =
					ReSTIR_PT_evaluate_target_function<true>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

				// Because we're using the target function as a PDF here, we need to scale the PDF
				// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

				// Only doing this if we at least have a target function to scale by the jacobian
				if (target_function_center_sample_at_neighbor > 0.0f)
				{
					// If this is an envmap path the jacobian is just 1 so this is not needed
					if (!center_pixel_reservoir_sample.is_envmap_path())
					{
						float jacobian = get_jacobian_determinant_reconnection_shift(
							center_pixel_reservoir_sample.rc_vertex, center_pixel_reservoir_sample.rc_vertex_geometric_normal.unpack(),
							neighbor_pixel_surface.shading_point, center_pixel_surface.shading_point,
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
					nume_mc	 = difference_function_mc * reservoir_resampled_M;
					denom_mc = center_reservoir_M + neighbors_confidence_sum * difference_function_mc;
				}
				else
				{
					nume_mc	 = difference_function_mc * reservoir_resampled_M;
					denom_mc = center_reservoir_M + neighbors_confidence_sum;
				}

				mc += nume_mc / denom_mc;
			}

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

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight_canonical(const HIPRTRenderData& render_data,

																float reservoir_being_resampled_confidence,
																ReSTIRPTReservoirSample& center_pixel_reservoir_sample,
																int center_pixel_reservoir_confidence,
																float center_pixel_reservoir_target_function,

																ReSTIRSurface& center_pixel_surface,
																int neighbor_pixel_index,
																float neighbors_confidence_sum,

																int reused_neighbors_count,
																float neighbor_selection_probability,

																Xorshift32Generator& random_number_generator)
	{

		if (neighbors_confidence_sum == 0)
			return 1.0f;

		// Resampling the center pixel, we're going to estimate the MIS weight using the stochastic pairwise estimator by selecting N_c (hardcoded to 1 in
		// this implementation) neighbors, according to section 4.2 of "Stochastic Pairwise MIS for Unbiased Large - Kernel Reuse in Real - Time, Hedstrom
		// et al. 2026"

		ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

		float target_function_center_sample_at_neighbor =
			ReSTIR_PT_evaluate_target_function<true>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

		// Because we're using the target function as a PDF here, we need to scale the PDF
		// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

		// Only doing this if we at least have a target function to scale by the jacobian
		if (target_function_center_sample_at_neighbor > 0.0f)
		{
			if (!center_pixel_reservoir_sample.is_envmap_path())
			{
				// If this is an envmap path the jacobian is just 1 so this is not needed

				float jacobian = get_jacobian_determinant_reconnection_shift(center_pixel_reservoir_sample.rc_vertex,
																			 center_pixel_reservoir_sample.rc_vertex_geometric_normal.unpack(),
																			 neighbor_pixel_surface.shading_point, center_pixel_surface.shading_point,
																			 render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());
				if (jacobian == 0.0f)
					// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
					// and has been rejected), the target function is set to 0
					target_function_center_sample_at_neighbor = 0.0f;
				else
					target_function_center_sample_at_neighbor *= jacobian;
			}
		}

		float target_function_center_sample_at_center = center_pixel_reservoir_target_function;

		float nume_mc = target_function_center_sample_at_center * center_pixel_reservoir_confidence;
		float denom_mc =
			target_function_center_sample_at_neighbor * neighbors_confidence_sum + target_function_center_sample_at_center * center_pixel_reservoir_confidence;
		float confidence_multiplier = reservoir_being_resampled_confidence / neighbors_confidence_sum;

		if (denom_mc != 0.0f)
		{
			float spmis_proba = 1.0f;
			if (neighbor_selection_probability > 0.0f)
				// 1.0f / (N_c * P_c(i))
				spmis_proba =
					1.0f / (render_data.render_settings.restir_pt_settings.spmis_settings.canonical_weight_estimation_count * neighbor_selection_probability);

			return spmis_proba * confidence_multiplier * nume_mc / denom_mc;
		}
		else
			return 0.0f;
	}

	HIPRT_HOST_DEVICE float get_resampling_MIS_weight_non_canonical(float neighbor_reservoir_confidence_sum,
																	float target_function_at_neighbor,
																	int center_pixel_reservoir_confidence,

																	float target_function_at_center,
																	float neighbors_confidence_sum,

																	int reused_neighbors_count,
																	float neighbor_selection_probability)
	{
		// Resampling a neighbor

		float nume	= target_function_at_neighbor * neighbor_reservoir_confidence_sum;
		float denom = target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center * center_pixel_reservoir_confidence;
		float mi	= denom == 0.0f ? 0.0f : (nume / denom);

		float spmis_proba = 1.0f / (reused_neighbors_count * neighbor_selection_probability);
		return spmis_proba * mi;
	}
};

template <>
struct ReSTIRPTSpatialResamplingMISWeight<RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight_canonical(const HIPRTRenderData& render_data,

																float reservoir_being_resampled_confidence,
																ReSTIRPTReservoirSample& center_pixel_reservoir_sample,
																int center_pixel_reservoir_confidence,
																float center_pixel_reservoir_target_function,

																ReSTIRSurface& center_pixel_surface,
																int neighbor_pixel_index,
																float neighbors_confidence_sum,

																int reused_neighbors_count,
																float neighbor_selection_probability,

																Xorshift32Generator& random_number_generator)
	{

		if (neighbors_confidence_sum == 0)
			return 1.0f;

		// Resampling the center pixel, we're going to estimate the MIS weight using the stochastic pairwise estimator by selecting N_c (hardcoded to 1 in
		// this implementation) neighbors, according to section 4.2 of "Stochastic Pairwise MIS for Unbiased Large - Kernel Reuse in Real - Time, Hedstrom
		// et al. 2026"

		ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

		float target_function_center_sample_at_neighbor =
			ReSTIR_PT_evaluate_target_function<true>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

		// Because we're using the target function as a PDF here, we need to scale the PDF
		// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

		// Only doing this if we at least have a target function to scale by the jacobian
		if (target_function_center_sample_at_neighbor > 0.0f)
		{
			if (!center_pixel_reservoir_sample.is_envmap_path())
			{
				// If this is an envmap path the jacobian is just 1 so this is not needed

				float jacobian = get_jacobian_determinant_reconnection_shift(center_pixel_reservoir_sample.rc_vertex,
																			 center_pixel_reservoir_sample.rc_vertex_geometric_normal.unpack(),
																			 neighbor_pixel_surface.shading_point, center_pixel_surface.shading_point,
																			 render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());
				if (jacobian == 0.0f)
					// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
					// and has been rejected), the target function is set to 0
					target_function_center_sample_at_neighbor = 0.0f;
				else
					target_function_center_sample_at_neighbor *= jacobian;
			}
		}

		float target_function_center_sample_at_center = center_pixel_reservoir_target_function;

		float nume_mc = target_function_center_sample_at_center * center_pixel_reservoir_confidence;
		float denom_mc =
			target_function_center_sample_at_neighbor * neighbors_confidence_sum + target_function_center_sample_at_center * center_pixel_reservoir_confidence;
		float confidence_multiplier = reservoir_being_resampled_confidence / (neighbors_confidence_sum + center_pixel_reservoir_confidence);

		if (denom_mc != 0.0f)
		{
			float spmis_proba = 1.0f;
			if (neighbor_selection_probability > 0.0f)
				// 1.0f / (N_c * P_c(i))
				spmis_proba =
					1.0f / (render_data.render_settings.restir_pt_settings.spmis_settings.canonical_weight_estimation_count * neighbor_selection_probability);

			return spmis_proba * confidence_multiplier * nume_mc / denom_mc;
		}
		else
			return 0.0f;
	}

	HIPRT_HOST_DEVICE float get_resampling_MIS_weight_non_canonical(float reservoir_being_resampled_confidence,
																	float reservoir_being_resampled_target_function,
																	int center_pixel_reservoir_confidence,

																	float target_function_at_center,
																	float neighbors_confidence_sum,

																	int reused_neighbors_count,
																	float neighbor_selection_probability)
	{
		// Resampling a neighbor

		float target_function_at_neighbor = reservoir_being_resampled_target_function;

		float nume	= target_function_at_neighbor * reservoir_being_resampled_confidence;
		float denom = target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center * center_pixel_reservoir_confidence;
		float mi	= denom == 0.0f ? 0.0f : (nume / denom);

		float spmis_proba	   = 1.0f / (reused_neighbors_count * neighbor_selection_probability);
		float defensive_factor = neighbors_confidence_sum / (neighbors_confidence_sum + (float)center_pixel_reservoir_confidence);
		return defensive_factor * spmis_proba * mi;
	}
};

#endif
