/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_UTILS_TEMPORAL_H
#define DEVICE_RESTIR_UTILS_TEMPORAL_H

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE int2 apply_permutation_sampling(int2 pixel_position, int random_bits)
{
	int2 offset = make_int2(random_bits & 3, (random_bits >> 2) & 3);
	pixel_position += offset;

	pixel_position.x ^= 3;
	pixel_position.y ^= 3;

	pixel_position -= offset;

	return pixel_position;
}
/**
 * Returns a pair (x, y, z) with
 *	x the linear index that can be used directly to index a buffer
 *	of render_data for getting data of the temporal neighbor. x is -1
 *	if there is no valid temporal neighbor (disoccluion / occlusion / out of viewport)
 *
 *	(y, z) the pixel coordinates of the backproject temporal neighbor position
 *	These two values will always be filled even if the temporal neighbor is invalid
 *	(disoccluion / occlusion / out of viewport)
 */
template <bool IsReSTIRGI>
HIPRT_HOST_DEVICE HIPRT_INLINE int3 find_temporal_neighbor_index(const HIPRTRenderData& render_data,
	const float3& current_shading_point, const float3& current_normal, int center_pixel_index, Xorshift32Generator& random_number_generator)
{
	const ReSTIRCommonTemporalPassSettings& temporal_pass_settings = ReSTIRSettingsHelper::get_restir_temporal_pass_settings<IsReSTIRGI>(render_data);

	float3 previous_screen_space_point_xyz = matrix_X_point(render_data.prev_camera.view_projection, current_shading_point);
	float2 previous_screen_space_point = make_float2(previous_screen_space_point_xyz.x, previous_screen_space_point_xyz.y);

	// Bringing back in [0, 1] from [-1, 1]
	previous_screen_space_point += make_float2(1.0f, 1.0f);
	previous_screen_space_point *= make_float2(0.5f, 0.5f);

	int2 resolution = render_data.render_settings.render_resolution;
	float2 prev_pixel_float = make_float2(previous_screen_space_point.x * resolution.x, previous_screen_space_point.y * resolution.y);
	// Bringing back in the center of the pixel
	prev_pixel_float -= make_float2(0.5f, 0.5f);

	// We're going to randomly look for an acceptable neighbor around the back-projected pixel location to find
	// in a given radius
	int temporal_neighbor_index = -1;
	for (int i = 0; i < temporal_pass_settings.max_neighbor_search_count + 1; i++)
	{
		float2 offset = make_float2(0.0f, 0.0f);
		if (i > 0)
			// Only randomly looking after we've at least checked whether or not the exact temporally reprojected location
			// is valid or not
			offset = make_float2(random_number_generator() - 0.5f, random_number_generator() - 0.5f) * temporal_pass_settings.neighbor_search_radius;

		int2 temporal_neighbor_screen_pixel_pos = make_int2(round(prev_pixel_float.x + offset.x), round(prev_pixel_float.y + offset.y));
		if (temporal_pass_settings.use_permutation_sampling && i == 0)
			// If we're looking at the direct temporal neighbor (without random offset), applying
			// permutation sampling if enabled
			temporal_neighbor_screen_pixel_pos = apply_permutation_sampling(temporal_neighbor_screen_pixel_pos, temporal_pass_settings.permutation_sampling_random_bits);

		if (temporal_neighbor_screen_pixel_pos.x < 0 || temporal_neighbor_screen_pixel_pos.x >= resolution.x || temporal_neighbor_screen_pixel_pos.y < 0 || temporal_neighbor_screen_pixel_pos.y >= resolution.y)
			// Previous pixel is out of the current viewport
			continue;

		temporal_neighbor_index = temporal_neighbor_screen_pixel_pos.x + temporal_neighbor_screen_pixel_pos.y * resolution.x;

		// We always want to read from the previous frame g-buffer for temporal neighbors
		bool use_previous_frame_g_buffer = true;
		// except if we're accumulating because then the camera is not moving --> no motion
		// --> temporal neighbor are on the same surface as the current -> the previous
		// g-buffer is the same as the current frame's --> no need to read from previous
		// frame g-buffer --> the previous frame G-buffer is deallocated to save VRAM
		use_previous_frame_g_buffer &= render_data.render_settings.use_prev_frame_g_buffer();
		if (check_neighbor_similarity_heuristics<IsReSTIRGI>(render_data, 
			temporal_neighbor_index, center_pixel_index, current_shading_point, current_normal, use_previous_frame_g_buffer))
			// We found a good neighbor
			break;

		// We didn't break so we didn't find a good neighbor
		temporal_neighbor_index = -1;
	}

	return make_int3(temporal_neighbor_index, static_cast<int>(round(prev_pixel_float.x)), static_cast<int>(round(prev_pixel_float.y)));
}

#endif
