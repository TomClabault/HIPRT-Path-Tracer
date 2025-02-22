/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIOTEMPORAL_NORMALIZATION_WEIGHT_H
#define DEVICE_RESTIR_DI_SPATIOTEMPORAL_NORMALIZATION_WEIGHT_H

#include "Device/includes/ReSTIR/MISWeightsCommon.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/Utils.h"

#define TEMPORAL_NEIGHBOR_ID 0

template <int BiasCorrectionMode, bool IsReSTIRGI>
struct ReSTIRSpatiotemporalNormalizationWeight {};

template <bool IsReSTIRGI>
struct ReSTIRSpatiotemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_M, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		float final_reservoir_weight_sum, int initial_candidates_reservoir_M,
		const ReSTIRSurface& center_pixel_surface,
		int temporal_neighbor_M, int center_pixel_index, int2 temporal_neighbor_coords, 
		float2 cos_sin_theta_rotation, float& out_normalization_nume, float& out_normalization_denom,
		Xorshift32Generator& random_number_generator)
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

		const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
		for (int neighbor = 0; neighbor < spatial_pass_settings.reuse_neighbor_count + 1; neighbor++)
		{
			// The last iteration of the loop is a special case that resamples the initial candidates reservoir
			// and so neighbor_pixel_index is never going to be used so we don't need to set it
			int neighbor_pixel_index;
			if (neighbor != spatial_pass_settings.reuse_neighbor_count)
			{
				neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, spatial_pass_settings, 
					neighbor, temporal_neighbor_coords,
					cos_sin_theta_rotation);

				if (neighbor_pixel_index == -1)
					// Neighbor out of the viewport
					continue;

				if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data,
					neighbor_pixel_index, center_pixel_index, 
					center_pixel_surface.shading_point, center_pixel_surface.shading_normal, render_data.render_settings.use_prev_frame_g_buffer()))
					continue;
			}

			// Getting the surface data at the neighbor
			// 
			// The surface at the center pixel passed in parameters is 
			// the surface in the current frame, that's what we want
			// since we're resampling initial candidates of the current
			// frame in the center pixel. We're not resampling the center
			// pixel from the previous frame so we need the current surface 
			ReSTIRSurface neighbor_surface;
			if (neighbor == spatial_pass_settings.reuse_neighbor_count)
				neighbor_surface = center_pixel_surface;
			else
				neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, render_data.render_settings.use_prev_frame_g_buffer(), random_number_generator);

			if (neighbor == spatial_pass_settings.reuse_neighbor_count)
				out_normalization_denom += initial_candidates_reservoir_M;
			else
				out_normalization_denom += ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(neighbor_pixel_index);
		}

		// The fused spatiotemporal pass also resamples a temporal neighbor so we add the M of that neighbor too
		out_normalization_denom += temporal_neighbor_M;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatiotemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_Z, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		const ReSTIRSampleType<IsReSTIRGI>& final_reservoir_sample, float final_reservoir_weight_sum, 
		ReSTIRSurface& center_pixel_surface, ReSTIRSurface& temporal_neighbor_surface,
		int center_pixel_M, int temporal_neighbor_M, int center_pixel_index, int2 temporal_neighbor_position,
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

		const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
		for (int neighbor = 0; neighbor < spatial_pass_settings.reuse_neighbor_count + 1; neighbor++)
		{
			// The last iteration of the loop is a special case that resamples the initial candidates reservoir
			// and so neighbor_pixel_index is never going to be used so we don't need to set it
			int neighbor_pixel_index;
			if (neighbor != spatial_pass_settings.reuse_neighbor_count)
			{
				neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, spatial_pass_settings,
					neighbor, temporal_neighbor_position, cos_sin_theta_rotation);

				if (neighbor_pixel_index == -1)
					// Invalid neighbor
					continue;

				if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data,
					neighbor_pixel_index, center_pixel_index, 
					center_pixel_surface.shading_point, center_pixel_surface.shading_normal, render_data.render_settings.use_prev_frame_g_buffer()))
					continue;
			}


			// Getting the surface data at the neighbor
			// 
			// The surface at the center pixel passed in parameters is 
			// the surface in the current frame, that's what we want
			// since we're resampling initial candidates of the current
			// frame in the center pixel. We're not resampling the center
			// pixel from the previous frame so we need the current surface 
			ReSTIRSurface neighbor_surface;
			if (neighbor == spatial_pass_settings.reuse_neighbor_count)
				neighbor_surface = center_pixel_surface;
			else
				neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, render_data.render_settings.use_prev_frame_g_buffer(), random_number_generator);

			float target_function_at_neighbor;
			if constexpr (IsReSTIRGI)
				// ReSTIR GI target function
				target_function_at_neighbor = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);
			else
				// ReSTIR DI target function
				target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);

			if (target_function_at_neighbor > 0.0f)
			{
				// If the neighbor could have produced this sample...

				if (neighbor == spatial_pass_settings.reuse_neighbor_count)
					out_normalization_denom += center_pixel_M;
				else
					out_normalization_denom += ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(render_data, neighbor_pixel_index);
			}
		}

		// Also taking the temporal neighbor into account which
		bool target_function_at_neighbor_gt_0;
		if constexpr (IsReSTIRGI)
			// ReSTIR GI target function
			target_function_at_neighbor_gt_0 = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, temporal_neighbor_surface, random_number_generator) > 0.0f;
		else
			// ReSTIR DI target function
			target_function_at_neighbor_gt_0 = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, temporal_neighbor_surface, random_number_generator) > 0.0f;

		if (target_function_at_neighbor_gt_0)
			out_normalization_denom += temporal_neighbor_M;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatiotemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(const HIPRTRenderData& render_data,
		const ReSTIRSampleType<IsReSTIRGI>& final_reservoir_sample, float final_reservoir_weight_sum, 
		ReSTIRSurface& center_pixel_surface, ReSTIRSurface& temporal_neighbor_surface,
		int selected_neighbor,
		int center_pixel_M, int temporal_neighbor_M, int center_pixel_index, int2 temporal_neighbor_coords,
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

		const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
		for (int neighbor = 0; neighbor < spatial_pass_settings.reuse_neighbor_count + 1; neighbor++)
		{
			// The last iteration of the loop is a special case that resamples the initial candidates reservoir
			// and so neighbor_pixel_index is never going to be used so we don't need to set it
			int neighbor_pixel_index;
			if (neighbor != spatial_pass_settings.reuse_neighbor_count)
			{
				neighbor_pixel_index = get_spatial_neighbor_pixel_index(render_data, spatial_pass_settings, neighbor, temporal_neighbor_coords, cos_sin_theta_rotation);

				if (neighbor_pixel_index == -1)
					// Invalid neighbor
					continue;

				if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data,
					neighbor_pixel_index, center_pixel_index, 
					center_pixel_surface.shading_point, center_pixel_surface.shading_normal, render_data.render_settings.use_prev_frame_g_buffer()))
					continue;
			}

			// Getting the surface data at the neighbor
			// 
			// The surface at the center pixel passed in parameters is
			// the surface in the current frame, that's what we want
			// since we're resampling initial candidates of the current
			// frame in the center pixel. We're not resampling the center
			// pixel from the previous frame so we need the current surface
			ReSTIRSurface neighbor_surface;
			if (neighbor == spatial_pass_settings.reuse_neighbor_count)
				neighbor_surface = center_pixel_surface;
			else
				neighbor_surface = get_pixel_surface(render_data, neighbor_pixel_index, random_number_generator);

			float target_function_at_neighbor;
			if constexpr (IsReSTIRGI)
				// ReSTIR GI target function
				target_function_at_neighbor = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);
			else
				// ReSTIR DI target function
				target_function_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, neighbor_surface, random_number_generator);

			if (target_function_at_neighbor > 0.0f)
			{
				// If the neighbor could have produced this sample...

				int M = 1;
				if (ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights)
				{
					if (neighbor == spatial_pass_settings.reuse_neighbor_count)
						M = center_pixel_M;
					else
						M = ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(render_data, neighbor_pixel_index);
				}

				// neighbor + 1 here because 0 is the temporal neighbor, not the first spatial neighbor
				if (neighbor + 1 == selected_neighbor)
					// Not multiplying by M here, this was done already when resampling the sample if we
					// were using confidence weights
					out_normalization_nume += target_function_at_neighbor;
				out_normalization_denom += target_function_at_neighbor * M;
			};
		}

		// Now handling the temporal neighbor
		float target_function_at_temporal_neighbor;
		if constexpr (IsReSTIRGI)
			// ReSTIR GI target function
			target_function_at_temporal_neighbor = ReSTIR_GI_evaluate_target_function<ReSTIR_GI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, temporal_neighbor_surface, random_number_generator);
		else
			// ReSTIR DI target function
			target_function_at_temporal_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisibility>(render_data, final_reservoir_sample, temporal_neighbor_surface, random_number_generator);

		if (selected_neighbor == TEMPORAL_NEIGHBOR_ID)
			out_normalization_nume += target_function_at_temporal_neighbor;

		int temporal_M = ReSTIRSettingsHelper::get_restir_settings<IsReSTIRGI>(render_data).use_confidence_weights ? temporal_neighbor_M : 1;
		out_normalization_denom += target_function_at_temporal_neighbor * temporal_M;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatiotemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the neighbors with balance heuristic MIS weights in the m_i terms
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatiotemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the neighbors
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

template <bool IsReSTIRGI>
struct ReSTIRSpatiotemporalNormalizationWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE, IsReSTIRGI>
{
	HIPRT_HOST_DEVICE void get_normalization(float& out_normalization_nume, float& out_normalization_denom)
	{
		// Nothing more to normalize, everything is already handled when resampling the neighbors
		out_normalization_nume = 1.0f;
		out_normalization_denom = 1.0f;
	}
};

#endif
