/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_TRIANGLE_UTILS_H
#define DEVICE_TRIANGLE_UTILS_H

#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE float3_t triangle_load_normal_not_normalized(const HIPRTRenderData& render_data, int triangle_index)
{
	int triangle_index_start = triangle_index * 3;

	float3_t vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index_start + 0]];
	float3_t vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index_start + 1]];
	float3_t vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[triangle_index_start + 2]];

	float3_t AB = vertex_B - vertex_A;
	float3_t AC = vertex_C - vertex_A;

	return hippt::cross(AB, AC);
}

HIPRT_DEVICE float triangle_load_area(const HIPRTRenderData& render_data, int triangle_index)
{
	return render_data.buffers.triangles_areas[triangle_index];
}

HIPRT_DEVICE ColorRGB32F triangle_load_emission(const HIPRTRenderData& render_data, int triangle_index)
{
	return render_data.buffers.materials_buffer_soa.get_emission(render_data.buffers.material_indices[triangle_index]);
}

#endif
