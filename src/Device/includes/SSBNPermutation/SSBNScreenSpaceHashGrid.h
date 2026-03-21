/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_SSBN_SCREEN_SPACE_HASH_GRID_H
#define DEVICE_INCLUDES_SSBN_SCREEN_SPACE_HASH_GRID_H

#include "Device/includes/HashGridHash.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE void SSBN_update_screen_space_hash_grid(HIPRTRenderData& render_data,
													 unsigned int pixel_x,
													 unsigned int pixel_y,
													 float3_t shading_point,
													 float3_t geometric_normal,
													 HIPRTCamera& current_camera)
{
	if (render_data.render_settings.sample_number != 0)
		// We only do this at the first sample because the camera doesn't move for offline rendering
		return;

	unsigned int sort_key;

	// 10 bits each for x and y
	// +1 to x and y so that we don't get a sort_key == 0 because that's a special case
	unsigned int x_bits = (static_cast<unsigned int>(pixel_x / SSBNPermutationBlockSize + 1) & ((1 << 10) - 1));
	unsigned int y_bits = (static_cast<unsigned int>(pixel_y / SSBNPermutationBlockSize + 1) & ((1 << 10) - 1)) << 10;

	sort_key = x_bits | y_bits;

	if (render_data.ssbn_settings.use_screen_space_hash_grid)
	{
		if (!render_data.ssbn_settings.use_surface_normal)
			geometric_normal = make_float3(0.0f, 0.0f, 0.0f);

		// 6 bits
		unsigned int normal_bits = hash_quantize_normal(geometric_normal, 2) << 20;

		sort_key |= normal_bits;
	}

	render_data.ssbn_settings.screen_space_hash_grid[pixel_x + pixel_y * render_data.render_settings.render_resolution.x] =
							make_uint3(sort_key, pixel_x, pixel_y);
}

#endif
