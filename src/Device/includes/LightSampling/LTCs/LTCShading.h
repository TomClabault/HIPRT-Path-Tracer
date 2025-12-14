/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTCSHADING_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTCSHADING_H

#include "Device/includes/LightSampling/LTCs/LTCReadParams.h"
#include "Device/includes/LightSampling/LTCs/LTCTransform.h"
#include "Device/includes/LightSampling/TriangleSamplingPolygonClipping.h"

HIPRT_DEVICE float integrate_edge_vector(float3 vertex_1, float3 vertex_2)
{
	vertex_1 = hippt::normalize(vertex_1);
	vertex_2 = hippt::normalize(vertex_2);
	float x = hippt::dot(vertex_1, vertex_2);
	float y = hippt::abs(x);

	float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
	float b = 3.4175940f + (4.1616724f + y) * y;
	float v = a / b;

	float theta_sintheta = (x > 0.0f) ? v : 0.5f * (1.0f / hippt::sqrt(hippt::max(1.0f - x * x, 1e-7f))) - v;

	return hippt::cross(vertex_1, vertex_2).z * theta_sintheta;
}

/**
 * 4 maximum clipped_vertices in the array for a triangle + clipping
 */
HIPRT_DEVICE float integrate_ltc_clipped_triangle(unsigned int vertex_count, float3 clipped_vertices[4])
{
	float result = 0.0;

	result += integrate_edge_vector(clipped_vertices[0], clipped_vertices[1]);
	result += integrate_edge_vector(clipped_vertices[1], clipped_vertices[2]);
	result += integrate_edge_vector(clipped_vertices[2], clipped_vertices[0]);
	if (vertex_count == 4)
		result += integrate_edge_vector(clipped_vertices[3], clipped_vertices[0]);

	return hippt::abs(result);
}

HIPRT_DEVICE float evaluate_ltc(const HIPRTRenderData& render_data,
	float3 vertex_A_world_space, float3 vertex_B_world_space, float3 vertex_C_world_space,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material,
	LTCLobe	ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_params, hippt::dot(view_direction, shading_normal), material, ltc_lobe);

	float3x3 ltc_matrix_inverse = inverse(float3x3(
		ltc_params.r, 0.0f, ltc_params.g,
		0.0f, ltc_params.b, 0.0f,
		ltc_params.a, 0.0f, 1.0f
	));

	float3 T, B;
	build_ONB_XZ_plane(shading_normal, T, B, view_direction);
	float3 vertex_A_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_A_world_space - shading_point);
	float3 vertex_B_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_B_world_space - shading_point);
	float3 vertex_C_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_C_world_space - shading_point);

	// Shading space to cosine space such that we sample the projected
	// solid angle of the triangle but transformed by the LTC
	float NoV = hippt::dot(view_direction, shading_normal);
	vertex_A_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_A_local, material, ltc_lobe);
	vertex_B_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_B_local, material, ltc_lobe);
	vertex_C_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_C_local, material, ltc_lobe);

	float3 vertices_local_space[4] = { vertex_A_local, vertex_C_local, vertex_B_local };
	unsigned int clipped_vertex_count = clip_polygon(3, vertices_local_space);
	if (clipped_vertex_count == 0)
		return 0.0f;

	float ltc_amplitude = read_ltc_amplitude(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_amplitude_data, hippt::dot(view_direction, shading_normal), material, ltc_lobe);

	vertices_local_space[0] = hippt::normalize(vertices_local_space[0]);
	vertices_local_space[1] = hippt::normalize(vertices_local_space[1]);
	vertices_local_space[2] = hippt::normalize(vertices_local_space[2]);
	vertices_local_space[3] = hippt::normalize(vertices_local_space[3]);
	return ltc_amplitude * integrate_ltc_clipped_triangle(clipped_vertex_count, vertices_local_space);
}

#endif
