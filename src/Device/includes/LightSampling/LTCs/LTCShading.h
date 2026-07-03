/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTCSHADING_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTCSHADING_H

#include "Device/includes/BSDFs/Fresnel.h"
#include "Device/includes/LightSampling/LTCs/LTCReadParams.h"
#include "Device/includes/LightSampling/LTCs/LTCTransform.h"
#include "Device/includes/LightSampling/TriangleSamplingPolygonClipping.h"

HIPRT_DEVICE float integrate_edge_vector(float3_t vertex_1, float3_t vertex_2)
{
	vertex_1 = hippt::normalize(vertex_1);
	vertex_2 = hippt::normalize(vertex_2);
	float x	 = hippt::dot(vertex_1, vertex_2);
	float y	 = hippt::abs(x);

	float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
	float b = 3.4175940f + (4.1616724f + y) * y;
	float v = a / b;

	float theta_sintheta = (x > 0.0f) ? v : 0.5f * (1.0f / hippt::sqrt(hippt::max(1.0f - x * x, 1e-7f))) - v;

	return hippt::cross(vertex_1, vertex_2).z * theta_sintheta;
}

/**
 * 4 maximum clipped_vertices in the array for a triangle + clipping
 */
HIPRT_DEVICE float integrate_ltc_clipped_triangle(unsigned int vertex_count, float3_t clipped_vertices[4])
{
	float result = 0.0f;

	result += integrate_edge_vector(clipped_vertices[0], clipped_vertices[1]);
	result += integrate_edge_vector(clipped_vertices[1], clipped_vertices[2]);
	if (vertex_count == 3)
		result += integrate_edge_vector(clipped_vertices[2], clipped_vertices[0]);
	else if (vertex_count == 4)
	{
		result += integrate_edge_vector(clipped_vertices[2], clipped_vertices[3]);
		result += integrate_edge_vector(clipped_vertices[3], clipped_vertices[0]);
	}

	return hippt::abs(result);
}

HIPRT_DEVICE float evaluate_ltc(const HIPRTRenderData& render_data,
								float3_t vertex_A_world_space,
								float3_t vertex_B_world_space,
								float3_t vertex_C_world_space,
								float3_t shading_point,
								float3_t view_direction,
								float3_t shading_normal,
								const DeviceUnpackedEffectiveMaterial& material,
								LTCLobe ltc_lobe)
{
	ColorRGBA32F ltc_params = read_ltc_params(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_params, hippt::dot(view_direction, shading_normal), material,
											  ltc_lobe);

	float3x3 ltc_matrix_inverse = inverse(float3x3(ltc_params.r, 0.0f, ltc_params.g, 0.0f, ltc_params.b, 0.0f, ltc_params.a, 0.0f, 1.0f));

	float3_t T, B;
	build_ONB_XZ_plane(shading_normal, T, B, view_direction);
	float3_t vertex_A_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_A_world_space - shading_point);
	float3_t vertex_B_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_B_world_space - shading_point);
	float3_t vertex_C_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_C_world_space - shading_point);

	// Shading space to cosine space such that we sample the projected
	// solid angle of the triangle but transformed by the LTC
	float NoV	   = hippt::dot(view_direction, shading_normal);
	vertex_A_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_A_local, material, ltc_lobe);
	vertex_B_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_B_local, material, ltc_lobe);
	vertex_C_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_C_local, material, ltc_lobe);

	float3_t vertices_local_space[4]  = { vertex_A_local, vertex_C_local, vertex_B_local };
	unsigned int clipped_vertex_count = clip_polygon(3, vertices_local_space);
	if (clipped_vertex_count == 0)
		return 0.0f;

	float ltc_amplitude;
	if (ltc_lobe == LTCLobe::SPECULAR_LOBE || ltc_lobe == LTCLobe::COAT_LOBE)
	{
		// This here approximates the amplitude of the specular BRDF, fresnel term included,
		// i.e. cook-torrance integral(F * D * G / (4 * NoV * NoL) * cos_theta) even though we
		// only have fitted data for the BRDF without fresnel term: integral(D * G / (4 * NoV * NoL) * cos_theta)
		//
		// Reference: [LTC Fresnel Approximation, Stephen Hill, SIGGRAPH 2016]

		// Assuming coming from air here for simplicity (the true solution is a bit annoying
		// as we'd have to bring a bunch of RayPayload and RayVolumeState state variables in here)
		float R0		= F0_from_eta(ltc_lobe == LTCLobe::SPECULAR_LOBE ? material.ior : material.coat_ior, 1.0f);
		float amplitude = read_ltc_amplitude(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_amplitude_data, hippt::dot(view_direction, shading_normal),
											 material, ltc_lobe);
		float fD = read_ltc_fresnel(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_fresnel_data, hippt::dot(view_direction, shading_normal), material,
									ltc_lobe);

		ltc_amplitude = R0 * amplitude + (1.0f - R0) * fD;
	}
	else
		ltc_amplitude = read_ltc_amplitude(render_data.bsdfs_data.ltcs_data.GGX_conductor_ltc_amplitude_data, hippt::dot(view_direction, shading_normal),
										   material, ltc_lobe);

	vertices_local_space[0] = hippt::normalize(vertices_local_space[0]);
	vertices_local_space[1] = hippt::normalize(vertices_local_space[1]);
	vertices_local_space[2] = hippt::normalize(vertices_local_space[2]);
	vertices_local_space[3] = hippt::normalize(vertices_local_space[3]);

	return ltc_amplitude * integrate_ltc_clipped_triangle(clipped_vertex_count, vertices_local_space);
}

#endif
