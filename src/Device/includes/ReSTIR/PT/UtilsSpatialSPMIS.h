/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_UTILS_SPATIAL_SPMIS_H
#define DEVICE_RESTIR_PT_UTILS_SPATIAL_SPMIS_H

#include "Device/includes/BSDFs/Dispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/LightSampling/LightClamping.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/PT/TargetFunction.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"

HIPRT_DEVICE float compatibility_guided_cell_selection_weight(const ReSTIRPTSPMISSettings& spmis_settings,
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
	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;
	unsigned int cached_reuse_cell_pixel_index	= spmis_settings.all_pixels_reuse_cell_pixel_index[center_pixel_index];
	if (cached_reuse_cell_pixel_index != HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
	{
		// We're going to reuse from the cell at the cached reuse pixel index to avoid running the expensive cell search each frame when we already have a good
		// cell chosen from previous frames
		unsigned int reuse_cell_index = spmis_settings.all_pixel_hashes[cached_reuse_cell_pixel_index];
		out_neighbors_confidence_sum  = spmis_settings.cell_confidence_sums[reuse_cell_index];
		out_reuse_cell_pixel_count	  = spmis_settings.cell_pixels_counters[reuse_cell_index];

		return reuse_cell_index;
	}

	// First, always WRSing the center cell
	unsigned int center_cell_index = spmis_settings.all_pixel_hashes[center_pixel_index];
	if (center_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		// Can happen if hash collision resolution fails
		return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

	unsigned int center_cell_weight = spmis_settings.cell_confidence_sums[center_cell_index];

	// Variables for WRS, starting with the center cell selected
	float weight_sum						  = center_cell_weight;
	unsigned int selected_cell_index		  = center_cell_index;
	unsigned int selected_pixel_index		  = center_pixel_index;
	unsigned int selected_cell_confidence_sum = center_cell_weight;

	constexpr float variance_slider_lower_bound = 0.5f;
	constexpr float variance_slider_upper_bound = 0.725f;
	float cell_variance							= spmis_settings.cell_variance[center_cell_index];
	if (cell_variance == -1.0f || cell_variance == 0.0f)
		// In case we have no variance information, default to maximum variance.
		// Same for 0 variance cells: 0 variance is basically impossible so this probably means that the cell is completely empty (no contributing reservoirs)
		// because the variance is so high (or it's a completely shadowed region)
		cell_variance = 1.0f;
	float variance_slider = hippt::clamp(0.0f, 1.0f, (cell_variance - variance_slider_lower_bound) / (1.0f - variance_slider_upper_bound));

	// Determining the search settings
	float radius;
	unsigned int max_search_iterations;
	float distance_scaling;
	if (spmis_settings.variance_aware_reuse_radius)
	{
		radius				  = hippt::lerp(10.0f, 20.0f, variance_slider);
		max_search_iterations = hippt::lerp(2, 12, variance_slider);
		if (cell_variance > 0.65f)
			// Disabled
			distance_scaling = 0.0f;
		else
			distance_scaling = hippt::lerp(3.0f, 8.0f, 1.0f - variance_slider);
	}
	else
	{
		radius				  = spmis_settings.initial_search_radius;
		max_search_iterations = spmis_settings.neighboring_cell_max_search_iterations;
		distance_scaling	  = spmis_settings.distance_scaling;
	}

	for (int i = 0; i < max_search_iterations; i++, radius *= spmis_settings.neighboring_cell_search_radius_increment)
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
			selection_weight *= distance_scaling > 0.0f ? 1.0f / (hippt::length(neighbor_coords - center_pixel_coords) / distance_scaling) : 1.0f;

		weight_sum += selection_weight;
		if (rng() < selection_weight / weight_sum)
		{
			// Selecting this neighbor cell
			selected_cell_index			 = neighbor_cell_index;
			selected_pixel_index		 = neighbor_pixel_index;
			selected_cell_confidence_sum = neighbor_cell_weight;
		}
	}

	if (selected_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		return HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX;

	if (render_data.render_settings.restir_pt_settings.common_spatial_pass.reuse_neighbor_count > 0)
	{
		out_neighbors_confidence_sum = selected_cell_confidence_sum;
		out_reuse_cell_pixel_count	 = spmis_settings.cell_pixels_counters[selected_cell_index];

		spmis_settings.all_pixels_reuse_cell_pixel_index[center_pixel_index] = selected_pixel_index;
	}
	else
	{
		out_neighbors_confidence_sum = 0;
		out_reuse_cell_pixel_count	 = 0;
	}

	return selected_cell_index;
}

HIPRT_DEVICE unsigned int get_spmis_spatial_neighbor_pixel_index(const HIPRTRenderData& render_data,
																 ReSTIRSurface& center_surface,
																 unsigned int neighbor_cell_index,
																 float& out_selection_probability,
																 Xorshift32Generator& rng)
{
	const ReSTIRPTSPMISSettings& spmis_settings = render_data.render_settings.restir_pt_settings.spmis_settings;

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
		CDFDeviceWithLUT<ReSTIR_PT_SPMISCDFLUTSize, unsigned short int> cell_cdf;
		unsigned int lut_offset = spmis_settings.cell_cdf_lut_offsets[neighbor_cell_index];
		cell_cdf.cdf			= spmis_settings.cell_cdfs + cell_start_index;
		cell_cdf.cdf_lut		= spmis_settings.cell_cdf_luts + lut_offset;
		cell_cdf.size			= non_zero_cell_size;

		for (int neighbor = 0; neighbor < spmis_settings.ris_neighbor_cdf_count; neighbor++)
		{
			unsigned int random_seed		  = rng.m_state.seed;
			unsigned int random_index		  = cell_cdf.sample(rng);
			unsigned int neighbor_pixel_index = spmis_settings.pixel_indices_sorted[cell_start_index + random_index];

			// TODO antithetic sampling?
			//
			// TODO all of that in another kernel pass to have better occupancy, same as House of cards
			// TODO fast approximation to specular lobe rather than going through the full BRDF, we just need something approximate
			// TODO sampling only the best neighbor? Not proportional?
			// TODO under which circumstances is non-canonical scaling good? It's basically when reuse is bad, which happens when? Specular surface in the
			// white room but not the metal bars in Minecraft harbor? What's the consensus? What's the heuristic?
			// TODO 8 seems to be the more efficient but what about when we optimize CDF sampling?
			//
			//
			//
			//
			// The best option so far is a la carte, everything enabled but with an approximation for the sample point BSDF
			ReSTIRPTReservoir neighbor_reservoir   = render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs[neighbor_pixel_index];
			float shift_mapping_jacobian_to_center = 1.0f;
			if (neighbor_reservoir.UCW > 0.0f && !neighbor_reservoir.sample.is_envmap_path())
				shift_mapping_jacobian_to_center = get_jacobian_determinant_reconnection_shift(
					neighbor_reservoir.sample.rc_vertex, neighbor_reservoir.sample.rc_vertex_geometric_normal.unpack(), center_surface.shading_point,
					render_data.g_buffer.primary_hit_position[neighbor_pixel_index],
					render_data.render_settings.restir_pt_settings.get_jacobian_heuristic_threshold());

			// RIS for selecting the neighbor
			float neighbor_importance = neighbor_reservoir.UCW * neighbor_reservoir.sample.target_function * neighbor_reservoir.confidence;
			// The total sum weight is stored in cdf[0] by the build cdf kernel
			float neighbor_importance_sum = spmis_settings.cell_cdfs[cell_start_index];
			float source_pdf			  = neighbor_importance / neighbor_importance_sum; // Importance sampling of the pixel in the cell

			float target_function = 1.0f;
			target_function *= neighbor_reservoir.UCW;
			target_function *= shift_mapping_jacobian_to_center;

			float distance_to_sample_point;
			float3_t incident_light_direction;
			if (neighbor_reservoir.sample.is_envmap_path())
			{
				// For envmap path, the direction is stored in the 'rc_vertex' value
				incident_light_direction = neighbor_reservoir.sample.rc_vertex;
				distance_to_sample_point = 1.0e35f;
			}
			else
			{
				// Not an envmap path, the direction is the difference between the current shading
				// point and the reconnection point
				incident_light_direction = neighbor_reservoir.sample.rc_vertex - center_surface.shading_point;
				distance_to_sample_point = hippt::length(incident_light_direction);
				if (distance_to_sample_point <= 1.0e-6f)
					// To avoid numerical instabilities
					continue;

				incident_light_direction /= distance_to_sample_point;
			}

			if (!neighbor_reservoir.sample.is_envmap_path() && neighbor_reservoir.sample.di_sample &&
				compute_cosine_term_at_light_source(neighbor_reservoir.sample.rc_vertex_geometric_normal.unpack(), -incident_light_direction) <= 0.0f)
				// Backfacing light
				continue;

			float cosine_term = hippt::dot(incident_light_direction, center_surface.shading_normal);
			if (cosine_term <= 0.0f && !bsdf_incident_light_info_transmission_lobe(neighbor_reservoir.sample.incident_light_info_at_visible_point))
				continue;

			target_function *= cosine_term;

			float bsdf_pdf;
			BSDFContext bsdf_context(center_surface.view_direction, center_surface.shading_normal, center_surface.geometric_normal, incident_light_direction,
									 const_cast<BSDFIncidentLightInfo&>(neighbor_reservoir.sample.incident_light_info_at_visible_point),
									 center_surface.ray_volume_state, false, center_surface.material, 0.0f,
									 MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION);

			ColorRGB32F visible_point_throughput = bsdf_dispatcher_eval(render_data, bsdf_context, bsdf_pdf, rng);
			if (visible_point_throughput.luminance() == 0.0f)
				continue;

			ColorRGB32F sample_point_throughput = ColorRGB32F(1.0f);
			if (!neighbor_reservoir.sample.di_sample)
				sample_point_throughput = ColorRGB32F(neighbor_reservoir.sample.bsdf_throughput_luminance_at_sample_point);

			target_function *= (visible_point_throughput * sample_point_throughput * neighbor_reservoir.sample.rc_vertex_incident_radiance).luminance();
			float mis_weight = 1.0f / spmis_settings.ris_neighbor_cdf_count;
			float weight	 = mis_weight * target_function / source_pdf;

			weight_sum += weight;
			if (rng() < weight / weight_sum)
			{
				selected_index			 = neighbor_pixel_index;
				selected_target_function = target_function;
			}
		}
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
			float target_function = neighbor_reservoir.UCW * neighbor_reservoir.sample.target_function * neighbor_reservoir.confidence;
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

#endif // #ifndef DEVICE_RESTIR_PT_UTILS_SPATIAL_SPMIS_H
