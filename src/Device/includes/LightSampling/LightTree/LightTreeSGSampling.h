/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/RenderData.h"

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_H

HIPRT_DEVICE LightSampleInformation sample_one_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data,
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 geometric_normal,
	int last_hit_primitive_index, RayPayload& ray_payload,
	Xorshift32Generator& rng)
{
	return LightSampleInformation();
}

HIPRT_DEVICE float pdf_of_emissive_triangle_light_tree_sg(const HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal, int global_emissive_triangle_index)
{
	return 0.0f;
}

#endif
