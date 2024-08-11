/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_H
#define DEVICE_RESTIR_DI_H

#include "Device/includes/RIS.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F sample_light_ReSTIR_DI(const HIPRTRenderData& render_data, const RayPayload& ray_payload, const HitInfo closest_hit_info, const float3& view_direction, Xorshift32Generator& random_number_generator, int2 pixel_coords, int2 resolution)
{
	int pixel_index = pixel_coords.x + pixel_coords.y * resolution.x;
	Reservoir reservoir = render_data.aux_buffers.spatial_reservoirs[pixel_index];

	return evaluate_reservoir_sample(render_data, ray_payload, closest_hit_info.inter_point, closest_hit_info.shading_normal, view_direction, reservoir);
}

#endif
