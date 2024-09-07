/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SPATIAL_MIS_WEIGHT_H
#define DEVICE_RESTIR_DI_SPATIAL_MIS_WEIGHT_H 

#include "Device/includes/ReSTIR/DI/Utils.h"

template <int BiasCorrectionMode>
struct ReSTIRDISpatialResamplingMISWeight {};

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_M>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& neighbor_reservoir,
		const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int reused_neighbors_count,
		int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation,
		Xorshift32Generator& random_number_generator)
	{
		return neighbor_reservoir.M;
	}
};

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_Z>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& neighbor_reservoir,
		const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int reused_neighbors_count,
		int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation,
		Xorshift32Generator& random_number_generator)
	{
		return neighbor_reservoir.M;
	}
};





template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& neighbor_reservoir,
		const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int reused_neighbors_count,
		int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation,
		Xorshift32Generator& random_number_generator)
	{
		return 1.0f;
	}
}; 

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE_CONFIDENCE_WEIGHTS>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& neighbor_reservoir,
		const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int reused_neighbors_count,
		int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation,
		Xorshift32Generator& random_number_generator)
	{
		return neighbor_reservoir.M;
	}
};





template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& neighbor_reservoir,
		const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int reused_neighbors_count,
		int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation,
		Xorshift32Generator& random_number_generator)
	{
		float nume = 0.0f;
		// We already have the target function at the center pixel, adding it to the denom
		float denom = 0.0f;

		for (int j = 0; j < reused_neighbors_count + 1; j++)
		{
			int neighbor_index_j = get_spatial_neighbor_pixel_index(render_data, j, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));
			if (neighbor_index_j == -1)
				// Invalid neighbor, skipping
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * res.x;
			if (!check_neighbor_similarity_heuristics(render_data, neighbor_index_j, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
				// Neighbor too dissimilar according to heuristics, skipping
				continue;

			ReSTIRDISurface neighbor_surface = get_pixel_surface(render_data, neighbor_index_j);

			float target_function_at_j = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisiblity>(render_data, neighbor_reservoir.sample, neighbor_surface);

			denom += target_function_at_j;
			if (j == current_neighbor)
				nume = target_function_at_j;
		}

		if (denom == 0.0f)
			return 0.0f;
		else
			return nume / denom;
	}
};

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH_CONFIDENCE_WEIGHTS>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& neighbor_reservoir,
		const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor, int reused_neighbors_count,
		int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation,
		Xorshift32Generator& random_number_generator)
	{
		float nume = 0.0f;
		// We already have the target function at the center pixel, adding it to the denom
		float denom = 0.0f;

		for (int j = 0; j < reused_neighbors_count + 1; j++)
		{
			int neighbor_index_j = get_spatial_neighbor_pixel_index(render_data, j, reused_neighbors_count, render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, center_pixel_coords, res, cos_sin_theta_rotation, Xorshift32Generator(render_data.random_seed));
			if (neighbor_index_j == -1)
				// Invalid neighbor, skipping
				continue;

			int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * res.x;
			if (!check_neighbor_similarity_heuristics(render_data, neighbor_index_j, center_pixel_index, center_pixel_surface.shading_point, center_pixel_surface.shading_normal))
				// Neighbor too dissimilar according to heuristics, skipping
				continue;

			ReSTIRDISurface neighbor_surface = get_pixel_surface(render_data, neighbor_index_j);

			float target_function_at_j = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisiblity>(render_data, neighbor_reservoir.sample, neighbor_surface);

			int M = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[neighbor_index_j].M;
			denom += target_function_at_j * M;
			if (j == current_neighbor)
				nume = target_function_at_j * M;
		}

		if (denom == 0.0f)
			return 0.0f;
		else
			return nume / denom;
	}
};

#endif