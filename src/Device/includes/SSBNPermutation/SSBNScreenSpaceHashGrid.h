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

	unsigned int pixel_gbuffer_hash = screen_space_gbuffer_hash(pixel_x, pixel_y, SSBNPermutationBlockSize, shading_point, geometric_normal, current_camera);

	render_data.ssbn_settings.screen_space_hash_grid[pixel_x + pixel_y * render_data.render_settings.render_resolution.x] =
							make_uint3(pixel_gbuffer_hash, pixel_x, pixel_y);
}

#endif
