/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_LIGHT_UTILS_H
#define DEVICE_KERNELS_REGIR_LIGHT_UTILS_H

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE HIPRT_INLINE bool sample_point_on_generic_triangle(int global_triangle_index, const float3* vertices_positions, const int* triangles_indices, Xorshift32Generator& rng,
	float3& out_sample_point, float3& out_sampled_triangle_normal, float& out_triangle_area);

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, unsigned int point_on_light_random_seed, unsigned int emissive_triangle_global_index, float3& out_triangle_normal, float& out_triangle_area)
{
	Xorshift32Generator rng(point_on_light_random_seed);

	float3 sampled_point;
	if (!sample_point_on_generic_triangle(emissive_triangle_global_index, render_data.buffers.vertices_positions, render_data.buffers.triangles_indices, rng,
		sampled_point, out_triangle_normal, out_triangle_area))
		return make_float3(-1.0e35f, -1.0e35f, -1.0e35f);

	return sampled_point;
}

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, unsigned int point_on_light_random_seed, unsigned int emissive_triangle_global_index, float3& out_triangle_normal)
{
	float trash_area;
	return reconstruct_sample_point_on_light(render_data, point_on_light_random_seed, emissive_triangle_global_index, out_triangle_normal, trash_area);
}

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, const ReGIRSample& sample, float3& out_triangle_normal, float& out_triangle_area)
{
	return reconstruct_sample_point_on_light(render_data, sample.point_on_light_random_seed, sample.emissive_triangle_global_index, out_triangle_normal, out_triangle_area);
}

HIPRT_DEVICE float3 reconstruct_sample_point_on_light(const HIPRTRenderData& render_data, const ReGIRSample& sample, float3& out_triangle_normal)
{
	return reconstruct_sample_point_on_light(render_data, sample.point_on_light_random_seed, sample.emissive_triangle_global_index, out_triangle_normal);
}

#endif
