/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_UTILS_SPATIAL_H
#define DEVICE_RESTIR_UTILS_SPATIAL_H

#include "Device/includes/PathTracing.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/KernelOptions/ReSTIRGIOptions.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRCommonSettings.h"
#include "HostDeviceCommon/ReSTIRSettingsHelper.h"

template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE void setup_adaptive_directional_spatial_reuse(HIPRTRenderData& render_data, unsigned int center_pixel_index, float2& cos_sin_theta_rotation, Xorshift32Generator& random_number_generator)
{
	ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
	// Generating a unique seed per pixel that will be used to generate the spatial neighbors of that pixel if Hammersley isn't used
	spatial_pass_settings.spatial_neighbors_rng_seed = random_number_generator.xorshift32();
	if (spatial_pass_settings.use_adaptive_directional_spatial_reuse)
	{
		spatial_pass_settings.reuse_radius = spatial_pass_settings.per_pixel_spatial_reuse_radius[center_pixel_index];
		// Storing the direction reuse mask in the 'current_pixel_directions_reuse_mask' field of the spatial
		// reuse settings so that we don't have to carry that parameter around in function calls everywhere...
		//
		// This parameter will be read by later by the function that samples a neighbor based on the allowed directions
		spatial_pass_settings.current_pixel_directions_reuse_mask = ReSTIRSettingsHelper::get_spatial_reuse_direction_mask_ull<IsReSTIRGI>(render_data, center_pixel_index);

		if (spatial_pass_settings.reuse_radius == 0)
			spatial_pass_settings.reuse_neighbor_count = 0;

		// No random rotation if using the adaptive directional spatial reuse so we're setting cos theta to 1.0f
		// and sin theta to 0.0f for no rotation
		cos_sin_theta_rotation = make_float2(1.0f, 0.0f);
	}
}

template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE HIPRT_INLINE bool do_include_visibility_term_or_not(const HIPRTRenderData& render_data, int current_neighbor_index)
{
	const ReSTIRCommonSpatialPassSettings& spatial_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
	bool visibility_only_on_last_pass = spatial_settings.do_visibility_only_last_pass;
	bool is_last_pass = spatial_settings.spatial_pass_index == spatial_settings.number_of_passes - 1;

	// Only using the visibility term on the last pass if so desired
	bool include_target_function_visibility = visibility_only_on_last_pass && is_last_pass;
	// Also allowing visibility if we want it at every pass
	include_target_function_visibility |= !spatial_settings.do_visibility_only_last_pass;

	// Only doing visibility for a few neighbors depending on 'neighbor_visibility_count'
	include_target_function_visibility &= current_neighbor_index < spatial_settings.neighbor_visibility_count;

	// Only doing visibility if we want it at all
	include_target_function_visibility &= (IsReSTIRGI ? ReSTIR_GI_SpatialTargetFunctionVisibility : ReSTIR_DI_SpatialTargetFunctionVisibility);

	// We don't want visibility for the center pixel because we're going to reuse the
	// target function stored in the reservoir anyways
	// Note: the center pixel has index 'spatial_settings.reuse_neighbor_count'
	// while actual *neighbors* have index between [0, spatial_settings.reuse_neighbor_count - 1]
	include_target_function_visibility &= current_neighbor_index != spatial_settings.reuse_neighbor_count;

	return include_target_function_visibility;
}

/**
 * Returns a pair of random numbers that should be used to sample the spatial neighbor disk of the current pixel
 * (i.e. pass the returned float2 to 'sample_in_disk_uv').
 * 
 * This function samples UVs for sampling in a disk such that the point sampled is only sampled in the allowed
 * directions of a pixel (according to its direction reuse masks).
 * 
 * Note that this function will sample the first sector if there are no sectors available around the given pixel
 */
HIPRT_HOST_DEVICE float2 sample_spatial_neighbor_from_allowed_directions(const HIPRTRenderData& render_data, const ReSTIRCommonSpatialPassSettings& spatial_pass_settings, int2 center_pixel_coords, Xorshift32Generator& rng)
{
	unsigned long long int directions_mask = spatial_pass_settings.current_pixel_directions_reuse_mask;
	int number_of_allowed_sectors = hippt::popc(directions_mask);
	unsigned char random_sector_index = rng.random_index(number_of_allowed_sectors);

	// Now that we have our random sector, we need to find what theta rotation corresponds
	// to that sector
	// 
	// So we're counting how many sectors come before our 'random_sector_index' and we're going to
	// multiply that sector count by 2Pi / 32 (or / 64 if using 64 bits)
	unsigned char count_left_to_go = random_sector_index + 1;

	// Counting how many sectors there before we reach our 'random_sector_index'
	int sector_index = 0;

	unsigned char bit_count_so_far = 0;
	if (hippt::popc(directions_mask) == ReSTIR_GI_SpatialDirectionalReuseBitCount)
		// Fast path if all the directions are allowed
		sector_index = random_sector_index;
	else
	{
		// A naive implementation of this would go something like
		// 
		// for (i = 0; i < ReSTIR_GI_SpatialDirectionalReuseBitCount; i++)
		// {
		//     if (directions_mask & (1ull << i))
		//     {
		//         --count_left_to_go;
		//
		//         if (count_left_to_go == 0)
		//             break;
		//     }
		// }
		// sector_index = i;
		// 
		// i.e., counting the bits one by one until we counted the number of bits we needed
		// 
		// 
		// But here we're going to count the sectors 'count_left_to_go' by 'count_left_to_go' to get things
		// a bit faster.
		// 
		// So if we have the directions mask:
		//	- 01110000
		//
		// and we want the 4th valid sector, i.e. random_sector_index == 3, then we can just go ahead and
		// count bits 4 by 4:
		//
		// 11110000 <--- 'directions_mask'
		// &
		// 00001111 <--- 'mask'
		// =
		// 00000000. 
		// --> popc(00000000) = 0 -----> 0 bits found
		//
		// We move the mask to the left by the number of bits we still have to find (which is still 4):
		// 11110000 <--- 'directions_mask'
		// &
		// 11110000 <--- 'mask'
		// =
		// 11110000. 
		// --> popc(11110000) = 4 -----> 4 bits found --> we found all the bits we needed so the sector index
		// is in position '10000000' = 7 here
		while (count_left_to_go > 0)
		{
			unsigned char mask_length = count_left_to_go;
			unsigned long long int mask = ((1ull << mask_length) - 1ull) << bit_count_so_far;
			int count_mask = hippt::popc(directions_mask & mask);

			count_left_to_go -= count_mask;
			bit_count_so_far += mask_length;
		}

		sector_index = --bit_count_so_far;
	}

	float theta_start = sector_index / (float)ReSTIR_GI_SpatialDirectionalReuseBitCount;
	// Generating a random theta in between theta_start and the start of the next sector (which is 1.0f / 32.0f wide)
	// i.e. a random theta inside our disk sector
	float random_theta = theta_start + rng() * (1.0f / (float)ReSTIR_GI_SpatialDirectionalReuseBitCount);

	return make_float2(random_theta, rng());
}

/**
 * Returns the linear index that can be used directly to index a buffer
 * of render_data of the 'neighbor_number'th neighbor that we're going
 * to spatially reuse from
 *
 * 'neighbor_number' is in [0, neighbor_reuse_count]
 * 'neighbor_reuse_count' is in [1, ReSTIRCommonSpatialPassSettings.reuse_neighbor_count]
 * 'neighbor_reuse_radius' is the radius of the disk within which the neighbors are sampled
 * 'center_pixel_coords' is the coordinates of the center pixel that is currently
 *		doing the resampling of its neighbors. Neighbors will be spatially sampled
 *		around that position
 * 'res' is the resolution of the viewport. This is used to check whether the generated
 *		neighbor location is outside of the viewport or not
 * 'cos_sin_theta_rotation' is a pair of float [x, y] with x = cos(random_rotation) and
 *		y = sin(random_rotation). This is used to rotate the points generated by the Hammersley
 *		sampler so that not each pixel on the image resample the exact same neighbors (and so
 *		that a given pixel P resamples different neighbors accros different frame, otherwise
 *		the Hammersley sampler would always generate the exact same points
 * 'rng' is a random generator used for generating spatial neighbor positions if not using a Hammersley
 *		point set. 
 * 
 *		Only used if render_data.render_settings.restir_settings.common_spatial_pass.use_hammersley == false
 */
template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE HIPRT_INLINE int get_spatial_neighbor_pixel_index(const HIPRTRenderData& render_data,
	int neighbor_index,
	int2 center_pixel_coords, float2 cos_sin_theta_rotation, Xorshift32Generator& rng)
{
	const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);

	int neighbor_pixel_index;
	if (neighbor_index == spatial_pass_settings.reuse_neighbor_count)
	{
		// If this is the last neighbor, we set it to ourselves
		// This is why our loop on the neighbors goes up to 'i < NEIGHBOR_REUSE_COUNT + 1'
		// It's so that when i == NEIGHBOR_REUSE_COUNT, we resample ourselves
		neighbor_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
	}
	else
	{
		// +1 and +1 here because we want to skip the first point as it is always (0, 0)
		// which means that we would be resampling ourselves (the center pixel) --> 
		// pointless because we already resample ourselves "manually" (that's why there's that
		// "if (neighbor_index == neighbor_reuse_count)" above, to resample the center pixel)
		float2 uv;
		if (spatial_pass_settings.use_hammersley)
			uv = sample_hammersley_2D(spatial_pass_settings.reuse_neighbor_count + 1, neighbor_index + 1);
		else
		{
			if (spatial_pass_settings.use_adaptive_directional_spatial_reuse)
				uv = sample_spatial_neighbor_from_allowed_directions(render_data, spatial_pass_settings, center_pixel_coords, rng);
			else
				uv = make_float2(rng(), rng());
		}

		float2 neighbor_offset_in_disk = sample_in_disk_uv(spatial_pass_settings.reuse_radius, uv);

		// 2D rotation matrix: https://en.wikipedia.org/wiki/Rotation_matrix
		float cos_theta = cos_sin_theta_rotation.x;
		float sin_theta = cos_sin_theta_rotation.y;
		float2 neighbor_offset_rotated = make_float2(neighbor_offset_in_disk.x * cos_theta - neighbor_offset_in_disk.y * sin_theta, neighbor_offset_in_disk.x * sin_theta + neighbor_offset_in_disk.y * cos_theta);
		int2 neighbor_offset_int = make_int2(static_cast<int>(roundf(neighbor_offset_rotated.x)), static_cast<int>(roundf(neighbor_offset_rotated.y)));

		int2 neighbor_pixel_coords;
		if (spatial_pass_settings.debug_neighbor_location)
		{
			int2 offset;
			if (spatial_pass_settings.debug_neighbor_location_direction == 0)
				// Horizontal
				offset = make_int2(spatial_pass_settings.reuse_radius, 0);
			else if (spatial_pass_settings.debug_neighbor_location_direction == 1)
				// Vertical
				offset = make_int2(0, spatial_pass_settings.reuse_radius);
			else
				// Diagonal
				offset = make_int2(spatial_pass_settings.reuse_radius, spatial_pass_settings.reuse_radius);

			neighbor_pixel_coords = center_pixel_coords + offset;
		}
		else
			neighbor_pixel_coords = center_pixel_coords + neighbor_offset_int;

		if (neighbor_pixel_coords.x < 0 || neighbor_pixel_coords.x >= render_data.render_settings.render_resolution.x ||
			neighbor_pixel_coords.y < 0 || neighbor_pixel_coords.y >= render_data.render_settings.render_resolution.y)
			// Rejecting the sample if it's outside of the viewport
			return -1;

		neighbor_pixel_index = neighbor_pixel_coords.x + neighbor_pixel_coords.y * render_data.render_settings.render_resolution.x;
		if (render_data.render_settings.enable_adaptive_sampling && render_data.render_settings.sample_number >= render_data.render_settings.adaptive_sampling_min_samples)
		{
			// If adaptive sampling is enabled, we only want to reuse a converged neighbor if the user allowed it
			// We also check whether or not we've reached the minimum amount of samples of adaptive sampling because
			// if adaptive sampling hasn't kicked in yet, there's no need to check whether the neighbor has converged or not yet

			if (spatial_pass_settings.allow_converged_neighbors_reuse)
			{
				// If we're allowing the reuse of converged neighbors, only doing so with a certain probability

				Xorshift32Generator rng_converged_neighbor_reuse(render_data.random_number);
				if (rng_converged_neighbor_reuse() > spatial_pass_settings.converged_neighbor_reuse_probability)
				{
					// We didn't pass the probability check, we are not allowed to reuse the neighbor if it
					// has converged

					if (render_data.aux_buffers.pixel_converged_sample_count[neighbor_pixel_index] != -1)
						// The neighbor is indeed converged, returning invalid neighbor with -1
						return -1;
				}
			}
			else if (render_data.aux_buffers.pixel_converged_sample_count[neighbor_pixel_index] != -1)
				// The user doesn't allow reusing converged neighbors and the neighbor is indeed converged
				// Returning -1 for invalid neighbor
				return -1;
		}
	}

	return neighbor_pixel_index;
}

template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE void spatial_neighbor_advance_rng(const HIPRTRenderData& render_data, Xorshift32Generator& rng)
{
	const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);

	if (!spatial_pass_settings.use_hammersley)
	{
		if (spatial_pass_settings.use_adaptive_directional_spatial_reuse)
		{
			// If not using Hammersley, then each point is generated with 3 random numbers
			// 
			// One for the random sector in the disk
			// One for the random theta within that sector
			// One for the random radius
			//
			// See the 'sample_spatial_neighbor_from_allowed_directions' function
			rng();
			rng();
			rng();
		}
		else
		{
			// If not using Hammersley and not using the adaptive directional spatial reuse
			// then we're just sampling with white noise and so this requires two random numbers
			// for sampling a neighbor in the disk
			rng();
			rng();
		}
	}
}

/**
 * Counts how many neighbors are eligible for reuse.
 * This is needed for proper normalization by pairwise MIS weights.
 *
 * A neighbor is not eligible if it is outside of the viewport or if
 * it doesn't satisfy the normal/plane/roughness heuristics
 *
 * 'out_valid_neighbor_M_sum' is the sum of the M values (confidences) of the
 * valid neighbors. Used by confidence-weights pairwise MIS weights
 *
 * The bits of 'out_neighbor_heuristics_cache' are 1 or 0 depending on whether or not
 * the corresponding neighbor was valid or not (can be reused later to avoid having to
 * re-evauate the heuristics). Neighbor 0 is LSB.
 */
template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE HIPRT_INLINE void count_valid_spatial_neighbors(const HIPRTRenderData& render_data,
	const ReSTIRSurface& center_pixel_surface,
	int2 center_pixel_coords, float2 cos_sin_theta_rotation,
	int& out_valid_neighbor_count, int& out_valid_neighbor_M_sum, int& out_neighbor_heuristics_cache)
{
	out_valid_neighbor_count = 0;

	const ReSTIRCommonSpatialPassSettings& spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
	Xorshift32Generator spatial_neighbors_rng(spatial_pass_settings.spatial_neighbors_rng_seed);

	int center_pixel_index = center_pixel_coords.x + center_pixel_coords.y * render_data.render_settings.render_resolution.x;
	int reused_neighbors_count = spatial_pass_settings.reuse_neighbor_count;

	for (int neighbor_index = 0; neighbor_index < reused_neighbors_count; neighbor_index++)
	{
		unsigned long long int* spatial_reuse_hit_rate_hits = nullptr;
		unsigned long long int* spatial_reuse_hit_rate_total = nullptr;

		if (spatial_pass_settings.compute_spatial_reuse_hit_rate)
			hippt::atomic_fetch_add(spatial_pass_settings.spatial_reuse_hit_rate_total, 1ull);

		int neighbor_pixel_index = get_spatial_neighbor_pixel_index<IsReSTIRGI>(render_data, neighbor_index, center_pixel_coords, cos_sin_theta_rotation, spatial_neighbors_rng);
		if (neighbor_pixel_index == -1)
			// Neighbor out of the viewport
			continue;

		if (!check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data,
			neighbor_pixel_index, center_pixel_index, center_pixel_surface.shading_point, ReSTIRSettingsHelper::get_normal_for_rejection_heuristic<IsReSTIRGI>(render_data, center_pixel_surface)))
			continue;

		if (spatial_pass_settings.compute_spatial_reuse_hit_rate)
			hippt::atomic_fetch_add(spatial_pass_settings.spatial_reuse_hit_rate_hits, 1ull);

		out_valid_neighbor_M_sum += ReSTIRSettingsHelper::get_restir_spatial_pass_input_reservoir_M<IsReSTIRGI>(render_data, neighbor_pixel_index);
		out_valid_neighbor_count++;
		out_neighbor_heuristics_cache |= (1 << neighbor_index);
	}
}

#endif
