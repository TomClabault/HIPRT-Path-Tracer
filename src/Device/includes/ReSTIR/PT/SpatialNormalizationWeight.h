/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_SPATIAL_NORMALIZATION_WEIGHT_H
#define DEVICE_RESTIR_PT_SPATIAL_NORMALIZATION_WEIGHT_H

#include "Device/includes/ReSTIR/DI_GI/MISWeightsCommon.h"
#include "Device/includes/ReSTIR/DI_GI/UtilsSpatial.h"
#include "Device/includes/ReSTIR/PT/Utils.h"

#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"

template <int BiasCorrectionMode>
struct ReSTIRPTSpatialNormalizationWeight
{
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
											 float final_reservoir_weight_sum,
											 const ReSTIRSurface& center_pixel_surface,
											 int2_t center_pixel_coords,
											 float& out_normalization_nume,
											 float& out_normalization_denom,
											 Xorshift32Generator& random_number_generator)
	{
		if (final_reservoir_weight_sum <= 0.0f)
		{
			// Invalid reservoir, returning directly
			out_normalization_nume	= 1.0f;
			out_normalization_denom = 1.0f;

			return;
		}

		// 1/M MIS weights are basically confidence weights only i.e. c_i / sum(c_j) with
		// c_i = r_i.M

		out_normalization_nume = 1.0f;
		// We're simply going to divide by the sum of all the M values of all the neighbors we resampled (including the center pixel)
		// so we're only going to set the denominator to that and the numerator isn't going to change
		out_normalization_denom = 0.0f;

		for (int neighbor = 0; neighbor < ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIR_VARIANT_PT>(render_data).reuse_neighbor_count + 1;
			 neighbor++)
		{
			int neighbor_pixel_index = get_spatial_neighbor_pixel_index<ReSTIR_VARIANT_PT>(render_data, neighbor, center_pixel_coords, random_number_generator);
			if (neighbor_pixel_index == -1)
				// Neighbor out of the viewport
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
			if (!check_neighbor_similarity_heuristics<ReSTIR_VARIANT_PT>(
					render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point,
					ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<ReSTIR_VARIANT_PT>(render_data, center_pixel_surface)))
				continue;

			out_normalization_denom +=
				ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_confidence<ReSTIR_VARIANT_PT>(render_data, neighbor_pixel_index);
		}
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
											 const ReSTIRPTReservoirSample& final_reservoir_sample,
											 float final_reservoir_weight_sum,
											 const ReSTIRSurface& center_pixel_surface,
											 int2_t center_pixel_coords,
											 float& out_normalization_nume,
											 float& out_normalization_denom,
											 Xorshift32Generator& random_number_generator)
	{
		if (final_reservoir_weight_sum <= 0)
		{
			// Invalid reservoir, returning directly
			out_normalization_nume	= 1.0f;
			out_normalization_denom = 1.0f;

			return;
		}

		// Checking how many of our neighbors could have produced the sample that we just picked
		// and we're going to divide by the sum of M values of those neighbors
		out_normalization_denom = 0.0f;
		out_normalization_nume	= 1.0f;

		int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
		const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIR_VARIANT_PT>(render_data);

		random_number_generator.m_state.seed = spatial_pass_settings.spatial_neighbors_rng_seed;

		for (int neighbor = 0; neighbor < spatial_pass_settings.reuse_neighbor_count + 1; neighbor++)
		{
			int neighbor_pixel_index = get_spatial_neighbor_pixel_index<ReSTIR_VARIANT_PT>(render_data, neighbor, center_pixel_coords, random_number_generator);
			if (neighbor_pixel_index == -1)
				// Invalid neighbor
				continue;

			if (!check_neighbor_similarity_heuristics<ReSTIR_VARIANT_PT>(
					render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point,
					ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<ReSTIR_VARIANT_PT>(render_data, center_pixel_surface)))
				continue;

			// Getting the surface data at the neighbor
			ReSTIRSurface neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

			float jacobian = 1.0f;
			if (!final_reservoir_sample.is_envmap_path())
				jacobian = get_jacobian_determinant_reconnection_shift(
					final_reservoir_sample.rc_vertex, final_reservoir_sample.rc_vertex_geometric_normal.unpack(), center_pixel_surface.shading_point,
					neighbor_surface.shading_point, render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

			float target_function_at_neighbor =
				jacobian * ReSTIR_PT_evaluate_target_function<true>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);

			if (target_function_at_neighbor > 0.0f)
				// If the neighbor could have produced this sample...
				out_normalization_denom +=
					ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_confidence<ReSTIR_VARIANT_PT>(render_data, neighbor_pixel_index);
		}
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_MIS_LIKE>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
											 const ReSTIRPTReservoirSample& final_reservoir_sample,
											 float final_reservoir_weight_sum,
											 const ReSTIRSurface& center_pixel_surface,
											 int selected_neighbor,
											 int2_t center_pixel_coords,
											 float& out_normalization_nume,
											 float& out_normalization_denom,
											 Xorshift32Generator& random_number_generator)
	{
		if (final_reservoir_weight_sum <= 0)
		{
			// Invalid reservoir, returning directly
			out_normalization_nume	= 1.0f;
			out_normalization_denom = 1.0f;

			return;
		}

		out_normalization_denom = 0.0f;
		out_normalization_nume	= 0.0f;

		random_number_generator.m_state.seed =
			ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIR_VARIANT_PT>(render_data).spatial_neighbors_rng_seed;

		for (int neighbor = 0; neighbor < ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIR_VARIANT_PT>(render_data).reuse_neighbor_count + 1;
			 neighbor++)
		{
			int neighbor_pixel_index = get_spatial_neighbor_pixel_index<ReSTIR_VARIANT_PT>(render_data, neighbor, center_pixel_coords, random_number_generator);
			if (neighbor_pixel_index == -1)
				// Invalid neighbor
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
			if (!check_neighbor_similarity_heuristics<ReSTIR_VARIANT_PT>(
					render_data, neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point,
					ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<ReSTIR_VARIANT_PT>(render_data, center_pixel_surface)))
				continue;

			// Getting the surface data at the neighbor
			ReSTIRSurface neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

			float target_function_at_neighbor =
				ReSTIR_PT_evaluate_target_function<true>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);

			if (!final_reservoir_sample.is_envmap_path())
				// Applying the jacobian to get "p_hat_from_i"
				target_function_at_neighbor *= hippt::max(
					0.0f, get_jacobian_determinant_reconnection_shift(
							  final_reservoir_sample.rc_vertex, final_reservoir_sample.rc_vertex_geometric_normal.unpack(), center_pixel_surface.shading_point,
							  neighbor_surface.shading_point, render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold()));

			if (target_function_at_neighbor > 0.0f)
			{
				int M = ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_confidence<ReSTIR_VARIANT_PT>(render_data, neighbor_pixel_index);

				if (neighbor == selected_neighbor)
					// Not multiplying by M here, this was done already when resampling the sample if we
					// we're using confidence weights
					out_normalization_nume = target_function_at_neighbor;
				out_normalization_denom += target_function_at_neighbor * M;
			};
		}
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_MIS_GBH>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the neighbors with balance heuristic MIS weights in the m_i terms
		out_normalization_nume	= 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled by the MIS weights when resampling the neighbors
		out_normalization_nume	= 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled by the MIS weights when resampling the neighbors
		out_normalization_nume	= 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled by the MIS weights when resampling the neighbors
		out_normalization_nume	= 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled by the MIS weights when resampling the neighbors
		out_normalization_nume	= 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled by the MIS weights when resampling the neighbors
		out_normalization_nume	= 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <>
struct ReSTIRPTSpatialNormalizationWeight<RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled by the MIS weights when resampling the neighbors
		out_normalization_nume	= 1.0f;
		out_normalization_denom = 1.0f;
	}
};

#endif // #ifndef DEVICE_RESTIR_PT_SPATIAL_NORMALIZATION_WEIGHT_H
