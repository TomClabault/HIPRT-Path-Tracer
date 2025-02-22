/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIAL_NORMALIZATION_WEIGHT_H
#define DEVICE_RESTIR_DI_SPATIAL_NORMALIZATION_WEIGHT_H

#include "Device/includes/ReSTIR/MISWeightsCommon.h"
#include "Device/includes/ReSTIR/Utils.h"
#include "Device/includes/ReSTIR/UtilsSpatial.h"

#include "HostDeviceCommon/ReSTIRSettingsHelper.h"

template <int BiasCorrectionMode, bool IsReSTIRGI>
struct ReSTIRSpatialNormalizationWeight {};

template <bool IsReSTIRGI>
struct ReSTIRSpatialNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_M, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		float final_reservoir_weight_sum, const ReSTIRSurface& center_pixel_surface,
		int2 center_pixel_coords,
		float2 cos_sin_theta_rotation, float& out_normalization_nume, float& out_normalization_denom)
	{
		if (final_reservoir_weight_sum <= 0.0f)
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
		out_normalization_denom = 0.0f;

		for (int neighbor = 0; neighbor < ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data).reuse_neighbor_count + 1; neighbor++)
		{
			int neighbor_pixel_index = get_spatial_neighbor_pixel_index<IsReSTIRGI>(render_data, neighbor, center_pixel_coords, cos_sin_theta_rotation);
			if (neighbor_pixel_index == -1)
				// Neighbor out of the viewport
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
			if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data,
				neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
				continue;

			out_normalization_denom += ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(render_data, neighbor_pixel_index);
		}
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_Z, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		const ReSTIRSampleType<IsReSTIRGI>& final_reservoir_sample, float final_reservoir_weight_sum,
		const ReSTIRSurface& center_pixel_surface,
		int2 center_pixel_coords,
		float2 cos_sin_theta_rotation, float& out_normalization_nume, float& out_normalization_denom,
		Xorshift32Generator& random_number_generator)
	{
		if (final_reservoir_weight_sum <= 0)
		{
			// Invalid reservoir, returning directly
			out_normalization_nume = 1.0f;
			out_normalization_denom = 1.0f;

			return;
		}

		// Checking how many of our neighbors could have produced the sample that we just picked
		// and we're going to divide by the sum of M values of those neighbors
		out_normalization_denom = 0.0f;
		out_normalization_nume = 1.0f;

		int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
		const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
		for (int neighbor = 0; neighbor < spatial_pass_settings.reuse_neighbor_count + (render_data.render_settings.DEBUG_DO_ONLY_NEIGHBOR ? 0 : 1); neighbor++)
		{
			int neighbor_pixel_index = get_spatial_neighbor_pixel_index<IsReSTIRGI>(render_data, neighbor, center_pixel_coords, cos_sin_theta_rotation);
			if (neighbor_pixel_index == -1)
				// Invalid neighbor
				continue;

			if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data,
				neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
				continue;

			// Getting the surface data at the neighbor
			ReSTIRSurface neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

			float target_function_at_neighbor;
			if constexpr (IsReSTIRGI)
				// ReSTIR GI target function
				target_function_at_neighbor = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);
			else
				// ReSTIR DI target function
				target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);

			if (target_function_at_neighbor > 0.0f)
				// If the neighbor could have produced this sample...
				out_normalization_denom += ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(render_data, neighbor_pixel_index);
		}
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		const ReSTIRSampleType<IsReSTIRGI>& final_reservoir_sample, float final_reservoir_weight_sum, 
		const ReSTIRSurface& center_pixel_surface,
		int selected_neighbor,
		int2 center_pixel_coords,
		float2 cos_sin_theta_rotation,
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

		out_normalization_denom = 0.0f;
		out_normalization_nume = 0.0f;

		for (int neighbor = 0; neighbor < ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data).reuse_neighbor_count + 1; neighbor++)
		{
			int neighbor_pixel_index = get_spatial_neighbor_pixel_index<IsReSTIRGI>(render_data, neighbor, center_pixel_coords, cos_sin_theta_rotation);
			if (neighbor_pixel_index == -1)
				// Invalid neighbor
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
			if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data,
				neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
				continue;

			// Getting the surface data at the neighbor
			ReSTIRSurface neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

			float target_function_at_neighbor;
			if constexpr (IsReSTIRGI)
				// ReSTIR GI target function
				target_function_at_neighbor = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);
			else
				// ReSTIR DI target function
				target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);

			if (target_function_at_neighbor > 0.0f)
			{
				int M = 1;
				if (ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights)
					M = ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(render_data, neighbor_pixel_index);

				if (neighbor == selected_neighbor)
					// Not multiplying by M here, this was done already when resampling the sample if we
					// we're using confidence weights
					out_normalization_nume += target_function_at_neighbor;
				out_normalization_denom += target_function_at_neighbor * M;
			};
		}
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the neighbors with balance heuristic MIS weights in the m_i terms
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the neighbors
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatialNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the neighbors
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

#endif
