/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_UTILS_SPATIAL_SPMIS_H
#define DEVICE_RESTIR_PT_UTILS_SPATIAL_SPMIS_H

#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"

HIPRT_DEVICE float compatibility_guided_cell_selection_weight(const ReSTIRCommonSPMISSettings& spmis_settings,
															  float3_t camera_position,
															  float3_t center_pixel_shading_point,
															  float3_t neighbor_pixel_shading_point,
															  float3_t center_pixel_geometric_normal,
															  float3_t neighbor_pixel_geometric_normal)
{
	float distance_to_camera_2	  = hippt::length2(center_pixel_shading_point - camera_position);
	float shading_points_distance = hippt::length(center_pixel_shading_point - neighbor_pixel_shading_point);
	float s						  = hippt::sqrt(distance_to_camera_2 * spmis_settings.compatibility_guided_cell_selection.solid_angle_omega * hippt::M_INV_PI);
	float heuristic_position	  = hippt::intrin_expf(-(shading_points_distance / s));

	float heuristic_normal = hippt::pow_8(hippt::max(0.0f, hippt::dot(center_pixel_geometric_normal, neighbor_pixel_geometric_normal)));

	return heuristic_position * heuristic_normal;
}

/**
 * Searches for a cell to reuse from around the center pixel, with increasing search radius
 */
HIPRT_DEVICE unsigned int spmis_get_reuse_cell_index(
	const HIPRTRenderData& render_data,
	unsigned int center_pixel_index,
	int2_t center_pixel_coords,
	const ReSTIRSurface& center_pixel_surface,
	int& out_neighbors_confidence_sum,
	unsigned int& out_reuse_cell_pixel_count,
	Xorshift32Generator& rng // Passing the RNG by copy because we don't want the RNG used here to advance our global RNG
)
{
	const ReSTIRCommonSPMISSettings& spmis_settings = ReSTIRSettingsHelper::get_restir_spmis_settings<ReSTIR_VARIANT_PT>(render_data);

	// First, always WRSing the center cell
	unsigned int center_cell_index = spmis_settings.all_pixel_hashes[center_pixel_index];
	if (center_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		// Can happen if hash collision resolution fails
		return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

	unsigned int center_cell_weight = spmis_settings.cell_confidence_sums[center_cell_index];

	// Variables for WRS, starting with the center cell selected
	float weight_sum						  = center_cell_weight;
	unsigned int selected_cell_index		  = center_cell_index;
	unsigned int selected_cell_confidence_sum = center_cell_weight;

	float radius = spmis_settings.initial_search_radius;
	// TODO test more cell taps
	for (int i = 0; i < spmis_settings.neighboring_cell_max_search_iterations; i++, radius *= spmis_settings.neighboring_cell_search_radius_increment)
	{
		int2_t random_offset = make_int2(radius * (rng() * 2.0f - 1.0f), radius * (rng() * 2.0f - 1.0f));
		// This searches in a square for simplicity, not a disk but that's fine
		int2_t neighbor_coords = center_pixel_coords + random_offset;

		// If out of the viewport, mirroring the coordinates on the borders
		if (neighbor_coords.x < 0)
			neighbor_coords.x = -neighbor_coords.x;
		else if (neighbor_coords.x >= render_data.render_settings.render_resolution.x)
			neighbor_coords.x = 2 * render_data.render_settings.render_resolution.x - neighbor_coords.x - 1;

		if (neighbor_coords.y < 0)
			neighbor_coords.y = -neighbor_coords.y;
		else if (neighbor_coords.y >= render_data.render_settings.render_resolution.y)
			neighbor_coords.y = 2 * render_data.render_settings.render_resolution.y - neighbor_coords.y - 1;

		unsigned int neighbor_pixel_index = neighbor_coords.x + neighbor_coords.y * render_data.render_settings.render_resolution.x;
		unsigned int neighbor_cell_index  = spmis_settings.all_pixel_hashes[neighbor_pixel_index];

		if (neighbor_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
			// Hit a pixel that doesn't have a valid cell index, skipping
			continue;
		else if (neighbor_cell_index == center_cell_index)
			// Hit a pixel that has the same cell index as the center pixel, skipping because we already accounted for the center cell at the beginning
			continue;

		float compatibility_weight = 1.0f;
		if (spmis_settings.compatibility_guided_cell_selection.do_compatibility_guided_selection)
		{
			compatibility_weight = compatibility_guided_cell_selection_weight(
				spmis_settings, render_data.current_camera.position, center_pixel_surface.shading_point,
				render_data.g_buffer.primary_hit_position[neighbor_pixel_index], center_pixel_surface.geometric_normal,
				render_data.g_buffer.geometric_normals[neighbor_pixel_index].unpack());
		}
		else if (!check_neighbor_similarity_heuristics<ReSTIR_VARIANT_PT>(render_data, neighbor_pixel_index, center_pixel_index,
																		  center_pixel_surface.shading_point, center_pixel_surface.geometric_normal))
			// Neighbor doesn't pass the similarity heuristics, skipping
			continue;

		unsigned int neighbor_cell_weight = spmis_settings.cell_confidence_sums[neighbor_cell_index];

		float selection_weight = neighbor_cell_weight;
		selection_weight *= compatibility_weight;
		if (!spmis_settings.compatibility_guided_cell_selection.do_compatibility_guided_selection)
			selection_weight *=
				spmis_settings.distance_scaling > 0.0f ? 1.0f / (hippt::length(neighbor_coords - center_pixel_coords) / spmis_settings.distance_scaling) : 1.0f;

		weight_sum += selection_weight;
		if (rng() < selection_weight / weight_sum)
		{
			// Selecting this neighbor cell
			selected_cell_index			 = neighbor_cell_index;
			selected_cell_confidence_sum = neighbor_cell_weight;
		}
	}

	if (selected_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

	if (render_data.render_settings.restir_pt_settings.common_spatial_pass.reuse_neighbor_count > 0)
	{
		out_neighbors_confidence_sum = selected_cell_confidence_sum;
		out_reuse_cell_pixel_count =
			render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings.cell_pixels_counters[selected_cell_index];
	}
	else
	{
		out_neighbors_confidence_sum = 0;
		out_reuse_cell_pixel_count	 = 0;
	}

	return selected_cell_index;
}

HIPRT_DEVICE unsigned int get_spmis_spatial_neighbor_pixel_index(const HIPRTRenderData& render_data,
																 unsigned int neighbor_cell_index,
																 float& out_selection_probability,
																 Xorshift32Generator& rng)
{
	const ReSTIRCommonSPMISSettings& spmis_settings = ReSTIRSettingsHelper::get_restir_spmis_settings<ReSTIR_VARIANT_PT>(render_data);

	unsigned int cell_start_index	= spmis_settings.cell_offsets[neighbor_cell_index];
	unsigned int non_zero_cell_size = spmis_settings.cell_non_zero_reservoir_counters[neighbor_cell_index];
	if (non_zero_cell_size == 0)
	{
		out_selection_probability = 0.0f;

		return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	}

	float weight_sum			   = 0.0f;
	float selected_target_function = 0.0f;
	unsigned int selected_index	   = HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	if (spmis_settings.ris_neighbor_cdf)
	{
		CDFDevice cell_cdf;
		cell_cdf.cdf  = spmis_settings.cell_cdfs + cell_start_index;
		cell_cdf.size = non_zero_cell_size;

		unsigned int random_index		  = cell_cdf.sample(rng);
		unsigned int neighbor_pixel_index = spmis_settings.pixel_indices_sorted[cell_start_index + random_index];

		ReSTIRPTReservoir neighbor_reservoir = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs[neighbor_pixel_index];

		// RIS for selecting the neighbor
		float neighbor_importance = neighbor_reservoir.UCW * neighbor_reservoir.sample.target_function * neighbor_reservoir.M;
		// The total sum weight is stored in cdf[0] by the build cdf kernel
		float neighbor_importance_sum = spmis_settings.cell_cdfs[cell_start_index];
		float source_pdf			  = neighbor_importance / neighbor_importance_sum; // Importance sampling of the pixel in the cell
		float target_function		  = neighbor_importance;
		float mis_weight			  = 1.0f;
		float weight				  = mis_weight * target_function / source_pdf;

		weight_sum += weight;
		selected_index			 = neighbor_pixel_index;
		selected_target_function = target_function;
	}
	else
	{
		for (int i = 0; i < spmis_settings.ris_neighbor_count; i++)
		{
			unsigned int random_index		  = rng.random_index(non_zero_cell_size);
			unsigned int neighbor_pixel_index = spmis_settings.pixel_indices_sorted[cell_start_index + random_index];

			ReSTIRPTReservoir neighbor_reservoir = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs[neighbor_pixel_index];

			// RIS for selecting the neighbor
			float source_pdf	  = 1.0f / non_zero_cell_size; // Uniform sampling of the pixels in the cell
			float target_function = neighbor_reservoir.UCW * neighbor_reservoir.sample.target_function * neighbor_reservoir.M;
			float mis_weight	  = 1.0f / spmis_settings.ris_neighbor_count;
			float weight		  = mis_weight * target_function / source_pdf;

			weight_sum += weight;
			if (rng() < weight / weight_sum)
			{
				selected_index			 = neighbor_pixel_index;
				selected_target_function = target_function;
			}
		}
	}

	if (weight_sum == 0.0f)
	{
		out_selection_probability = 0.0f;

		return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;
	}
	else
	{
		float UCW = weight_sum / selected_target_function;
		// For clarity, the spatial reuse loop expects a PDF not an UCW (inverse PDF) so we're inverting it here
		out_selection_probability = 1.0f / UCW;

		return selected_index;
	}
}

#endif
