/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_SOLID_ANGLE_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_SOLID_ANGLE_H

#include "Device/includes/LightSampling/LTCs/LTCTransform.h"
#include "Device/includes/LightSampling/LTCs/LTCLobeUtils.h"
#include "Device/includes/LightSampling/TriangleSamplingSolidAngleCommon.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE float triangle_solid_angle(float3 vertex_A_worldspace, float3 vertex_B_worldspace, float3 vertex_C_worldspace, float3 shading_point)
{
	float3 vertex_A_local = hippt::normalize(vertex_A_worldspace - shading_point);
	float3 vertex_B_local = hippt::normalize(vertex_B_worldspace - shading_point);
	float3 vertex_C_local = hippt::normalize(vertex_C_worldspace - shading_point);

	float solid_angle = hippt::abs(2 * atan2f(
		hippt::dot(vertex_A_local, hippt::cross(vertex_B_local, vertex_C_local)),
		1 + hippt::dot(vertex_A_local, vertex_B_local) + hippt::dot(vertex_A_local, vertex_C_local) + hippt::dot(vertex_B_local, vertex_C_local)
	));

	return solid_angle;
}

/**
 * Adapted from the implementation given with the paper from Cristoph Peters, 
 * [BRDF Importance Sampling for Polygonal Lights, 2021]
 */

/*! This structure carries intermediate results that only need to be computed
	once per polygon and shading point to take samples proportional to solid
	angle. Sampling is performed by subdividing the convex polygon into
	triangles as triangle fan around vertex 0 and then using our variant of
	Arvo's method.*/
struct solid_angle_triangle_t 
{
	//! The number of vertices that form the polygon
	unsigned int vertex_count;
	//! Normalized direction vectors from the shading point to each vertex
	float3 vertex_dirs[3];
	/*! A few intermediate quantities about the triangle consisting of vertices
		i + 1, 0, and i + 2. If the three vertices are v0, v1, v2, the entries
		are determinant(mat3(v0, v1, v2)), hippt::dot(v0 + v1, v2) and
		1.0f + hippt::dot(v0, v1).*/
	float3 triangle_parameters;
	//! At index i, this array holds the solid angle of the triangle fan formed
	//! by vertices 0 to i + 2.
	float fan_solid_angles;
	//! The total solid angle of the polygon
	float solid_angle;
};

HIPRT_DEVICE float solid_angle_triangle_solid_angle_pdf(const HIPRTRenderData& render_data,
	float triangle_solid_angle, float NoL,
	float3 vertex_A_world_space, float3 vertex_B_world_space, float3 vertex_C_world_space,
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 sampled_dir_shading_space,
	const LTCLobeSampleProbabilities& ltc_lobe_probabilities, const DeviceUnpackedEffectiveMaterial& material)
{
	float out_pdf = 0.0f;

	/*out_pdf += solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		ltc_lobe_probabilities, material,
		LTCLobe::COAT_LOBE);

	out_pdf += solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		ltc_lobe_probabilities, material,
		LTCLobe::METALLIC_LOBE);

	out_pdf += solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		ltc_lobe_probabilities, material,
		LTCLobe::SPECULAR_LOBE);

	out_pdf += solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		ltc_lobe_probabilities, material,
		LTCLobe::DIFFUSE_LOBE);*/

	return out_pdf;
}

HIPRT_DEVICE float solid_angle_triangle_solid_angle_pdf()
{
	return 0.0f;
}

/*! Prepares all intermediate values to sample a triangle fan around vertex 0
	(e.g. a convex polygon) proportional to solid angle using our method.
	\param vertex_count Number of vertices forming the polygon.
	\param vertices List of vertex locations.
	\param shading_point The location of the shading point.
	\return Input for sample_point_on_triangle_solid_angle_peters_2021().*/
HIPRT_DEVICE solid_angle_triangle_t prepare_solid_angle_triangle_sampling(const HIPRTRenderData& render_data,
	float3 vertex_A, float3 vertex_B, float3 vertex_C,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material)
{
	solid_angle_triangle_t polygon;
	polygon.vertex_count = 3;

//#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
//	/**
//	 * With the view direction in the x-z plane + LTC transform
//	 */
//	 // Building a shading space where the shading point is the origin, the shading normal
//	 // is the z axis, and the view direction lies in the x-z plane
//	float3 T, B;
//	build_ONB_XZ_plane(shading_normal, T, B, view_direction);
//	float3 vertex_A_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_A - shading_point);
//	float3 vertex_B_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_B - shading_point);
//	float3 vertex_C_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_C - shading_point);
//
//	// Normalizing vertices helps a bit with float numerical precision
//	vertex_A_local = hippt::normalize(vertex_A_local);
//	vertex_B_local = hippt::normalize(vertex_B_local);
//	vertex_C_local = hippt::normalize(vertex_C_local);
//
//	// Shading space to cosine space such that we sample the
//	// solid angle of the triangle but transformed by the LTC
//	float NoV = hippt::dot(view_direction, shading_normal);
//
//	// TODO UNCOMMENT THIS
//	/*vertex_A_local = ltc_transform_shading_to_cosine(render_data, NoV, material.roughness, vertex_A_local);
//	vertex_B_local = ltc_transform_shading_to_cosine(render_data, NoV, material.roughness, vertex_B_local);
//	vertex_C_local = ltc_transform_shading_to_cosine(render_data, NoV, material.roughness, vertex_C_local);*/
//
//	vertex_A_local = hippt::normalize(vertex_A_local);
//	vertex_B_local = hippt::normalize(vertex_B_local);
//	vertex_C_local = hippt::normalize(vertex_C_local);
//
//	polygon.vertex_dirs[0] = vertex_A_local;
//	polygon.vertex_dirs[1] = vertex_B_local;
//	polygon.vertex_dirs[2] = vertex_C_local;
//#else
	// Normalize vertex directions for better floating point precision during the sampling
	polygon.vertex_dirs[0] = hippt::normalize(vertex_A - shading_point);
	polygon.vertex_dirs[1] = hippt::normalize(vertex_B - shading_point);
	polygon.vertex_dirs[2] = hippt::normalize(vertex_C - shading_point);
//#endif

	// Prepare a Householder transform that maps vertex 0 onto (+/-1, 0, 0). We
	// only store the yz-components of that Householder vector and a factor of
	// 2.0f / sqrt(abs(polygon.vertex_dirs[0].x) + 1.0f) is pulled in there to
	// save on multiplications later. This approach is necessary to avoid
	// numerical instabilities in determinant computation below.
	float householder_sign = (polygon.vertex_dirs[0].x > 0.0f) ? -1.0f : 1.0f;
	float2 householder_yz = make_float2(polygon.vertex_dirs[0].y, polygon.vertex_dirs[0].z) * (1.0f / (hippt::abs(polygon.vertex_dirs[0].x) + 1.0f));
	// Compute solid angles and prepare sampling
	polygon.solid_angle = 0.0f;
	float previous_dot_1_2 = hippt::dot(polygon.vertex_dirs[0], polygon.vertex_dirs[1]);

	// We look at one triangle of the triangle fan at a time
	float3 vertices[3] = {
		polygon.vertex_dirs[1],
		polygon.vertex_dirs[0],
		polygon.vertex_dirs[2] 
	};

	float dot_0_1 = previous_dot_1_2;
	float dot_0_2 = hippt::dot(vertices[0], vertices[2]);
	float dot_1_2 = hippt::dot(vertices[1], vertices[2]);
	previous_dot_1_2 = dot_1_2;
	// Compute the bottom right minor of vertices after application of the
	// Householder transform
	float dot_householder_0 = hippt::fma(-householder_sign, vertices[0].x, dot_0_1);
	float dot_householder_2 = hippt::fma(-householder_sign, vertices[2].x, dot_1_2);
	float2x2 bottom_right_minor = float2x2(
		hippt::fma(make_float2(-dot_householder_0), householder_yz, make_float2(vertices[0].y, vertices[0].z)),
		hippt::fma(make_float2(-dot_householder_2), householder_yz, make_float2(vertices[2].y, vertices[2].z)));
	// The absolute value of the determinant of vertices equals the 2x2
	// determinant because the Householder transform turns the first column
	// into (+/-1, 0, 0)
	float simplex_volume = abs(determinant(bottom_right_minor));
	// Compute the solid angle of the triangle using a formula proposed by:
	// A. Van Oosterom and J. Strackee, 1983, The Solid Angle of a
	// Plane Triangle, IEEE Transactions on Biomedical Engineering 30:2
	// https://doi.org/10.1109/TBME.1983.325207
	float dot_0_2_plus_1_2 = dot_0_2 + dot_1_2;
	float one_plus_dot_0_1 = 1.0f + dot_0_1;
	float tangent = simplex_volume / (one_plus_dot_0_1 + dot_0_2_plus_1_2);
	float triangle_solid_angle = 2.0f * positive_atan(tangent);

	polygon.solid_angle += triangle_solid_angle;
	polygon.fan_solid_angles = polygon.solid_angle;
	// Some intermediate results from above help us with sampling
	polygon.triangle_parameters = make_float3(simplex_volume, dot_0_2_plus_1_2, one_plus_dot_0_1);

	return polygon;
}


/*! An implementation of mix() using two fused-multiply add instructions. Used
	because the native mix() implementation had stability issues in a few
	spots. Credit to Fabian Giessen's blog, see:
	https://fgiesen.wordpress.com/2012/08/15/linear-interpolation-past-present-and-future/
	*/
HIPRT_DEVICE float mix_fma(float x, float y, float a)
{
	return hippt::fma(a, y, hippt::fma(-a, x, x));
}

/*! Given the output of prepare_solid_angle_triangle_sampling(), this function
	maps the given random numbers in the range from 0 to 1 to a normalized
	direction vector providing a sample of the solid angle of the polygon in
	the original space (used for arguments of
	prepare_solid_angle_triangle_sampling()). Samples are distributed in
	proportion to solid angle assuming uniform inputs.*/
HIPRT_DEVICE float3 sample_point_on_triangle_solid_angle_peters_2021(const HIPRTRenderData& render_data,
	float3 vertex_A, float3 vertex_B, float3 vertex_C, float3 triangle_normal, 
	float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material,
	float& out_area_pdf,
	Xorshift32Generator& rng)
{
	solid_angle_triangle_t polygon = prepare_solid_angle_triangle_sampling(render_data, 
		vertex_A, vertex_B, vertex_C, 
		shading_point, view_direction, shading_normal, 
		material);

	// Decide which triangle needs to be sampled
	float rand_1 = rng();
	float target_solid_angle = polygon.solid_angle * rand_1;
	float3 parameters = polygon.triangle_parameters;
	float3 vertices[3] = { polygon.vertex_dirs[1], polygon.vertex_dirs[0], polygon.vertex_dirs[2] };

	// Construct a new vertex 2 on the arc between vertices 0 and 2 such that
	// the resulting triangle has solid angle subtriangle_solid_angle
	float2 cos_sin = make_float2(hippt::intrin_cosf(0.5f * target_solid_angle), hippt::intrin_sinf(0.5f * target_solid_angle));
	float3 offset = vertices[0] * (parameters.x * cos_sin.x - parameters.y * cos_sin.y) + vertices[2] * (parameters.z * cos_sin.y);
	float3 new_vertex_2 = hippt::fma(2.0f * make_float3(hippt::dot(vertices[0], offset) / hippt::dot(offset, offset)), offset, -vertices[0]);

	// Now sample the line between vertex 1 and the newly created vertex 2
	float rand_2 = rng();
	float s2 = hippt::dot(vertices[1], new_vertex_2);
	float s = mix_fma(1.0f, s2, rand_2);
	float denominator = hippt::fma(-s2, s2, 1.0f);
	float t_normed = sqrt(hippt::fma(-s, s, 1.0f) / denominator);
	// s2 may exceed one due to rounding error. random_numbers[1] is the
	// limit of t_normed for s2 -> 1.
	t_normed = (denominator > 0.0f) ? t_normed : rand_2;

	float3 sampled_direction = hippt::normalize(hippt::fma(-t_normed, s2, s) * vertices[1] + t_normed * new_vertex_2);

//#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
//	// From cosine space to shading space
//	float3 sampled_dir_shading_space = hippt::normalize(ltc_transform_cosine_to_shading(render_data, hippt::dot(view_direction, shading_normal), material.roughness, sampled_direction));
//
//	float3 T, B;
//	build_ONB_XZ_plane(shading_normal, T, B, view_direction);
//	float3x3 rotation_matrix = float3x3::from_rows(T, B, shading_normal);
//
//	// Multiplying the vector from the left to effectively
//	// multiply by the transpose of the rotation matrix which is its inverse.
//	//
//	// This brings the direction from shading space to world space.
//	float3 sampled_dir_world_space = hippt::normalize(sampled_dir_shading_space * rotation_matrix);
//
//	float pdf_solid_angle = solid_angle_triangle_solid_angle_pdf(render_data,
//		polygon.projected_solid_angle, sampled_dir.z,
//		vertex_A, vertex_B, vertex_C,
//		shading_point, view_direction, shading_normal, sampled_dir_shading_space,
//		ltc_lobe_probabilities, material);
//#else
	float3 sampled_dir_world_space = sampled_dir_world_space = sampled_direction;
	float pdf_solid_angle = 1.0f / polygon.solid_angle;
//#endif

	return map_direction_to_triangle_point(sampled_dir_world_space, vertex_A, triangle_normal, shading_point, pdf_solid_angle, out_area_pdf);
}

#endif
