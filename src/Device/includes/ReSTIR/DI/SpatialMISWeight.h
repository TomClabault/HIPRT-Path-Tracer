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
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(int reservoir_being_resampled_M)
	{
		return reservoir_being_resampled_M;
	}
};

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_1_OVER_Z>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(int reservoir_being_resampled_M)
	{
		return reservoir_being_resampled_M;
	}
};

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_LIKE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data, int reservoir_being_resampled_M)
	{
		return render_data.render_settings.restir_di_settings.use_confidence_weights ? reservoir_being_resampled_M : 1;
	}
}; 

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_MIS_GBH>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled,
		const ReSTIRDISurface& center_pixel_surface,
		int current_neighbor,
		int2 center_pixel_coords, int2 res, float2 cos_sin_theta_rotation)
	{
		if (reservoir_being_resampled.UCW == 0.0f)
			// Reservoir that doesn't contain any sample, returning 
			// 1.0f MIS weight so that multiplying by that doesn't do anything
			return 1.0f;

		float nume = 0.0f;
		float denom = 0.0f;

		int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;
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

			float target_function_at_j = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisiblity>(render_data, reservoir_being_resampled.sample, neighbor_surface);

			int M = 1;
			if (render_data.render_settings.restir_di_settings.use_confidence_weights)
				M = render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[neighbor_index_j].M;
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

template <>
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& center_pixel_reservoir,
		float target_function_at_center,
		int current_neighbor_index, int neighbor_pixel_index, int valid_neighbors_count)
	{
		int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;
		if (current_neighbor_index < reused_neighbors_count)
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
			// so that we know that the visibility is correct and thus we do not run into any issues and we can just$
			// reuse the target function stored in the neighbor's reservoir
			float target_function_at_neighbor = reservoir_being_resampled.sample.target_function;

			float nume = target_function_at_neighbor;
			float denom = target_function_at_neighbor + target_function_at_center / valid_neighbors_count;
			float mi = denom == 0.0f ? 0.0f : (nume / denom);

			ReSTIRDISurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index);
			float target_function_center_reservoir_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisiblity>(render_data, center_pixel_reservoir.sample, neighbor_pixel_surface);
			float target_function_center_reservoir_at_center = center_pixel_reservoir.sample.target_function;

			float nume_mc = target_function_center_reservoir_at_center / valid_neighbors_count;
			float denom_mc = target_function_center_reservoir_at_neighbor + target_function_center_reservoir_at_center / valid_neighbors_count;
			mc += (denom_mc == 0.0f ? 0.0f : (nume_mc / denom_mc)) / valid_neighbors_count;

			return mi / valid_neighbors_count;
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
struct ReSTIRDISpatialResamplingMISWeight<RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE>
{
	HIPRT_HOST_DEVICE float get_resampling_MIS_weight(const HIPRTRenderData& render_data,
		const ReSTIRDIReservoir& reservoir_being_resampled, const ReSTIRDIReservoir& center_pixel_reservoir,
		float target_function_at_center,
		int current_neighbor_index, int neighbor_pixel_index, int valid_neighbors_count)
	{
		int reused_neighbors_count = render_data.render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count;
		if (current_neighbor_index < reused_neighbors_count)
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
			// so that we know that the visibility is correct and thus we do not run into any issues and we can just$
			// reuse the target function stored in the neighbor's reservoir
			float target_function_at_neighbor = reservoir_being_resampled.sample.target_function;

			float nume = target_function_at_neighbor;
			float denom = target_function_at_neighbor + target_function_at_center / valid_neighbors_count;
			float mi = denom == 0.0f ? 0.0f : (nume / denom);

			ReSTIRDISurface neighbor_pixel_surface = get_pixel_surface(render_data, neighbor_pixel_index);
			float target_function_center_reservoir_at_neighbor = ReSTIR_DI_evaluate_target_function<ReSTIR_DI_BiasCorrectionUseVisiblity>(render_data, center_pixel_reservoir.sample, neighbor_pixel_surface);
			float target_function_center_reservoir_at_center = center_pixel_reservoir.sample.target_function;

			float nume_mc = target_function_center_reservoir_at_center / valid_neighbors_count;
			float denom_mc = target_function_center_reservoir_at_neighbor + target_function_center_reservoir_at_center / valid_neighbors_count;
			mc += (denom_mc == 0.0f ? 0.0f : (nume_mc / denom_mc));

			// In the defensive formulation, we want to divide by M, not M-1.
				// (Eq. 7.6 of "A Gentle Introduction to ReSTIR")
			return mi / (valid_neighbors_count + 1);
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

				// In the defensive formulation, we want to divide by M, not M-1.
				// (Eq. 7.6 of "A Gentle Introduction to ReSTIR")
				return (1 + mc) / (valid_neighbors_count + 1);
			}
		}
	}

	// Weight for the canonical sample (center pixel)
	float mc = 0.0f;
};

#endif