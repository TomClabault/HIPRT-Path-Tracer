/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIAL_MIS_WEIGHT_H
#define DEVICE_RESTIR_DI_SPATIAL_MIS_WEIGHT_H 

#include "Device/includes/ReSTIR/MISWeightsCommon.h"
#include "Device/includes/ReSTIR/DI/TargetFunction.h"
#include "Device/includes/ReSTIR/Utils.h"
#include "Device/includes/ReSTIR/GI/TargetFunction.h"
#include "HostDeviceCommon/ReSTIRSettingsHelper.h"

template <int BiasCorrectionMode, bool IsReSTIRGI>
struct ReSTIRSpatialResamplingMISWeight {};

template <bool IsReSTIRGI>
struct ReSTIRSpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_M, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(int reservoir_being_resampled_M)
	{
		return reservoir_being_resampled_M;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_Z, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(int reservoir_being_resampled_M)
	{
		return reservoir_being_resampled_M;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data, int reservoir_being_resampled_M)
	{
		return ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights ? reservoir_being_resampled_M : 1;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,

		float reservoir_being_resampled_UCW,
		const ReSTIRSampleType<IsReSTIRGI>& reservoir_being_resampled_sample,

		const ReSTIRSurface& center_pixel_surface,
		int current_neighbor_index,
		int2 center_pixel_coords, float2 cos_sin_theta_rotation,
		Xorshift32Generator& random_number_generator)
	{
		if (reservoir_being_resampled_UCW <= 0.0f)
			// Reservoir that doesn't contain any sample, returning 
			// 1.0f MIS weight so that multiplying by that doesn't do anything
			return 1.0f;

		float nume = 0.0f;
		float denom = 0.0f;

		for (int j = 0; j < ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data).reuse_neighbor_count + 1; j++)
		{
			int neighbor_index_j = get_spatial_neighbor_pixel_index<IsReSTIRGI>(render_data, j, center_pixel_coords, cos_sin_theta_rotation);
			if (neighbor_index_j == -1)
				// Invalid neighbor, skipping
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
			if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data, neighbor_index_j, center_pixel_index, center_pixel_surface.shading_point, ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<IsReSTIRGI>(render_data, center_pixel_surface)))
				// Neighbor too dissimilar according to heuristics, skipping
				continue;

			ReSTIRSurface neighbor_surface = get_pixel_surface(render_data, neighbor_index_j, random_number_generator);

			float target_function_at_j;
			if constexpr (IsReSTIRGI)
			{
				// ReSTIR GI target function
				if (j == current_neighbor_index)
					target_function_at_j = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility, /* resampling neighbor */ false>(render_data, reservoir_being_resampled_sample, neighbor_surface, random_number_generator);
				else
					target_function_at_j = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility, /* resampling neighbor */ true>(render_data, reservoir_being_resampled_sample, neighbor_surface, random_number_generator);

				if (!reservoir_being_resampled_sample.is_envmap_path())
					// Applying the jacobian to get "p_hat_from_i"
					target_function_at_j *= hippt::max(0.0f, get_jacobian_determinant_reconnection_shift(reservoir_being_resampled_sample.sample_point, reservoir_being_resampled_sample.sample_point_geometric_normal, center_pixel_surface.shading_point, neighbor_surface.shading_point, render_data.render_settings.restir_gi_settings.get_jacobian_heuristic_threshold()));
			}
			else
				// ReSTIR DI target function
				target_function_at_j = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, reservoir_being_resampled_sample, neighbor_surface, random_number_generator);

			int M = 1;
			if (ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights)
				M = ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(render_data, neighbor_index_j);
			denom += target_function_at_j * M;
			if (j == current_neighbor_index)
				nume = target_function_at_j * M;
		}

		if (denom == 0.0f)
			return 0.0f;
		else
			return nume / denom;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,

		int reservoir_being_resampled_M, float reservoir_being_resampled_target_function, 
		ReSTIRSampleType<IsReSTIRGI>& center_pixel_reservoir_sample, int center_pixel_reservoir_M, float center_pixel_reservoir_target_function,

		ReSTIRSurface& center_pixel_surface, float target_function_at_center,
		int neighbor_pixel_index, int valid_neighbors_count, int valid_neighbors_M_sum,
		bool update_mc, bool resampling_canonical,
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
			float target_function_at_neighbor = reservoir_being_resampled_target_function;
			float target_function_center_reservoir_at_center = center_pixel_reservoir_target_function;

			bool use_confidence_weights = ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights;
			float reservoir_resampled_M = use_confidence_weights ? reservoir_being_resampled_M : 1;
			float center_reservoir_M = use_confidence_weights ? center_pixel_reservoir_M : 1;
			float neighbors_confidence_sum = use_confidence_weights ? valid_neighbors_M_sum : 1;
			// We only want to divide by M-1 if we're not using confidence weights.
			// (Eq. 7.6 and 7.7 of "A Gentle Introduction to ReSTIR")
			float valid_neighbor_division_term = use_confidence_weights ? 1 : valid_neighbors_count;

			float nume = target_function_at_neighbor * reservoir_resampled_M;
			float denom = target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center / valid_neighbor_division_term * center_reservoir_M;
			float mi = denom == 0.0f ? 0.0f : (nume / denom);

			if (update_mc)
			{
				ReSTIRSurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);
				float target_function_center_reservoir_at_neighbor;
				if constexpr (IsReSTIRGI)
				{
					// ReSTIR GI target function
					target_function_center_reservoir_at_neighbor = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);
					
					// Because we're using the target function as a PDF here, we need to scale the PDF
					// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

					// Only doing this if we at least have a target function to scale by the jacobian
					if (target_function_center_reservoir_at_neighbor > 0.0f)
					{
						// If this is an envmap path the jacobian is just 1 so this is not needed
						if (!center_pixel_reservoir_sample.is_envmap_path())
						{
							float jacobian = get_jacobian_determinant_reconnection_shift(center_pixel_reservoir_sample.sample_point, center_pixel_reservoir_sample.sample_point_geometric_normal, center_pixel_surface.shading_point, neighbor_pixel_surface.shading_point, render_data.render_settings.restir_gi_settings.get_jacobian_heuristic_threshold());
							
#if 0
							// TODO below is a test of BSDF ratio jacobian for unbiased ReSTIR GI but this doesn't seem to work
							if (render_data.render_settings.DEBUG_DO_BSDF_RATIO)
							{
								float new_pdf;
								BSDFContext new_pdf_context(hippt::normalize(neighbor_pixel_surface.shading_point - center_pixel_reservoir_sample.sample_point), center_pixel_reservoir_sample.sample_point_shading_normal, center_pixel_reservoir_sample.sample_point_geometric_normal, center_pixel_reservoir_sample.incident_light_direction_at_sample_point, center_pixel_reservoir_sample.incident_light_info_at_sample_point, center_pixel_reservoir_sample.sample_point_volume_state, false, center_pixel_reservoir_sample.sample_point_material.unpack(), 1, 0.0f);
								bsdf_dispatcher_eval(render_data, new_pdf_context, new_pdf, random_number_generator);

								float old_pdf;
								BSDFContext old_pdf_context(hippt::normalize(center_pixel_surface.shading_point - center_pixel_reservoir_sample.sample_point), center_pixel_reservoir_sample.sample_point_shading_normal, center_pixel_reservoir_sample.sample_point_geometric_normal, center_pixel_reservoir_sample.incident_light_direction_at_sample_point, center_pixel_reservoir_sample.incident_light_info_at_sample_point, center_pixel_reservoir_sample.sample_point_volume_state, false, center_pixel_reservoir_sample.sample_point_material.unpack(), 1, 0.0f);
								bsdf_dispatcher_eval(render_data, old_pdf_context, old_pdf, random_number_generator);

								float bsdf_pdf_ratio = new_pdf / old_pdf;
								jacobian *= bsdf_pdf_ratio;
							}
#endif

							if (jacobian == 0.0f)
								// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
								// and has been rejected), the target function is set to 0
								target_function_center_reservoir_at_neighbor = 0.0f;
							else
								target_function_center_reservoir_at_neighbor /= jacobian;
						}
					}
				}
				else
					// ReSTIR DI target function
					target_function_center_reservoir_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

				float nume_mc = target_function_center_reservoir_at_center / valid_neighbor_division_term * center_reservoir_M;
				float denom_mc = target_function_center_reservoir_at_neighbor * neighbors_confidence_sum + target_function_center_reservoir_at_center / valid_neighbor_division_term * center_reservoir_M;

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

template <bool IsReSTIRGI>
struct ReSTIRSpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,

		int reservoir_being_resampled_M, float reservoir_being_resampled_target_function, 
		const ReSTIRSampleType<IsReSTIRGI>& center_pixel_reservoir_sample, int center_pixel_reservoir_M, float center_pixel_reservoir_target_function,

		ReSTIRSurface& center_pixel_surface, float target_function_at_center,
		int neighbor_pixel_index, int valid_neighbors_count, int valid_neighbors_M_sum,
		bool update_mc, bool resampling_canonical,
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

			bool use_confidence_weights = ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights;
			float reservoir_resampled_M = use_confidence_weights ? reservoir_being_resampled_M : 1;
			float center_reservoir_M = use_confidence_weights ? center_pixel_reservoir_M : 1;
			float neighbors_confidence_sum = use_confidence_weights ? valid_neighbors_M_sum : 1;
			// We only want to divide by M-1 if we're not using confidence weights.
			// (Eq. 7.6 and 7.7 of "A Gentle Introduction to ReSTIR")
			float valid_neighbor_division_term = use_confidence_weights ? 1 : valid_neighbors_count;

			float nume = target_function_at_neighbor * reservoir_resampled_M;
			float denom = target_function_at_neighbor * neighbors_confidence_sum + target_function_at_center / valid_neighbor_division_term * center_reservoir_M;
			float mi = 0.0f;
			if (denom != 0.0f)
				mi = nume / denom;
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

				float target_function_center_reservoir_at_neighbor;
				if constexpr (IsReSTIRGI)
				{
					// ReSTIR GI target function
					target_function_center_reservoir_at_neighbor = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

					// Because we're using the target function as a PDF here, we need to scale the PDF
					// by the jacobian. That's p_hat_from_i, Eq. 5.9 of "A Gentle Introduction to ReSTIR"

					// Only doing this if we at least have a target function to scale by the jacobian
					if (target_function_center_reservoir_at_neighbor > 0.0f)
					{
						if (!center_pixel_reservoir_sample.is_envmap_path())
						{
							// If this is an envmap path the jacobian is just 1 so this is not needed

							float jacobian = get_jacobian_determinant_reconnection_shift(center_pixel_reservoir_sample.sample_point, center_pixel_reservoir_sample.sample_point_geometric_normal, center_pixel_surface.shading_point, neighbor_pixel_surface.shading_point, render_data.render_settings.restir_gi_settings.get_jacobian_heuristic_threshold());
							if (jacobian == 0.0f)
								// Clamping at 0.0f so that if the jacobian returned is -1.0f (meaning that the jacobian doesn't match the threshold
								// and has been rejected), the target function is set to 0
								target_function_center_reservoir_at_neighbor = 0.0f;
							else
								target_function_center_reservoir_at_neighbor /= jacobian;
						}
					}
				}
				else
					// ReSTIR DI target function
					target_function_center_reservoir_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, center_pixel_reservoir_sample, neighbor_pixel_surface, random_number_generator);

				float target_function_center_reservoir_at_center = center_pixel_reservoir_target_function;

				float nume_mc = target_function_center_reservoir_at_center / valid_neighbor_division_term * center_reservoir_M;
				float denom_mc = target_function_center_reservoir_at_neighbor * neighbors_confidence_sum + target_function_center_reservoir_at_center / valid_neighbor_division_term * center_reservoir_M;
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

				if (ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights)
					return mc + static_cast<float>(center_pixel_reservoir_M) / static_cast<float>(center_pixel_reservoir_M + valid_neighbors_M_sum);
				else
					// In the defensive formulation, we want to divide by M, not M-1.
					// (Eq. 7.6 of "A Gentle Introduction to ReSTIR") so 'valid_neighbors_count + 1'
					return (1 + mc) / (valid_neighbors_count + 1.0f);
			}
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = 0.0f;
};

#endif
