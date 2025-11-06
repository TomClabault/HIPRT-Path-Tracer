/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_SOLID_ANGLE_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_SOLID_ANGLE_H

/**
 * Adapted from the implementation given with the paper from Cristoph Peters, 
 * [BRDF Importance Sampling for Polygonal Lights, 2021]
 */

#define MAX_POLYGON_VERTEX_COUNT 3

/*! This structure carries intermediate results that only need to be computed
	once per polygon and shading point to take samples proportional to solid
	angle. Sampling is performed by subdividing the convex polygon into
	triangles as triangle fan around vertex 0 and then using our variant of
	Arvo's method.*/
struct solid_angle_polygon_t 
{
	//! The number of vertices that form the polygon
	unsigned int vertex_count;
	//! Normalized direction vectors from the shading point to each vertex
	float3 vertex_dirs[MAX_POLYGON_VERTEX_COUNT];
	/*! A few intermediate quantities about the triangle consisting of vertices
		i + 1, 0, and i + 2. If the three vertices are v0, v1, v2, the entries
		are determinant(mat3(v0, v1, v2)), hippt::dot(v0 + v1, v2) and
		1.0f + hippt::dot(v0, v1).*/
	float3 triangle_parameters[MAX_POLYGON_VERTEX_COUNT - 2];
	//! At index i, this array holds the solid angle of the triangle fan formed
	//! by vertices 0 to i + 2.
	float fan_solid_angles[MAX_POLYGON_VERTEX_COUNT - 2];
	//! The total solid angle of the polygon
	float solid_angle;
};

/*! A piecewise polynomial approximation to positive_atan(y). The maximal
	absolute error is 1.16e-05f. At least on Turing GPUs, it is faster but also
	significantly less accurate. The proper atan has at most 2 ulps of error
	there.*/
HIPRT_DEVICE float fast_positive_atan(float y)
{
	float rx;
	float ry;
	float rz;
	rx = (abs(y) > 1.0f) ? (1.0f / abs(y)) : abs(y);
	ry = rx * rx;
	rz = hippt::fma(ry, 0.02083509974181652f, -0.08513300120830536);
	rz = hippt::fma(ry, rz, 0.18014100193977356f);
	rz = hippt::fma(ry, rz, -0.3302994966506958f);
	ry = hippt::fma(ry, rz, 0.9998660087585449f);
	rz = hippt::fma(-2.0f * ry, rx, hippt::M_PI_TWO);
	rz = (abs(y) > 1.0f) ? rz : 0.0f;
	rx = hippt::fma(rx, ry, rz);
	return (y < 0.0f) ? (hippt::M_Pi - rx) : rx;
}

/*! Returns an angle between 0 and M_PI such that tan(angle) == tangent. In
	other words, it is a version of atan() that is offset to be non-negative.
	Note that it may be switched to an approximate mode by the
	USE_BIASED_PROJECTED_SOLID_ANGLE_SAMPLING flag.*/
HIPRT_DEVICE float positive_atan(float tangent)
{
#ifdef USE_BIASED_PROJECTED_SOLID_ANGLE_SAMPLING
	return fast_positive_atan(tangent);
#else
	float offset = (tangent < 0.0f) ? hippt::M_Pi : 0.0f;
	return atanf(tangent) + offset;
#endif
}

/*! Prepares all intermediate values to sample a triangle fan around vertex 0
	(e.g. a convex polygon) proportional to solid angle using our method.
	\param vertex_count Number of vertices forming the polygon.
	\param vertices List of vertex locations.
	\param shading_position The location of the shading point.
	\return Input for sample_solid_angle_polygon().*/
HIPRT_DEVICE solid_angle_polygon_t prepare_solid_angle_polygon_sampling(unsigned int vertex_count, float3 vertex_A, float3 vertex_B, float3 vertex_C, float3 shading_position)
{
	solid_angle_polygon_t polygon;
	polygon.vertex_count = vertex_count;

	// Normalize vertex directions
	polygon.vertex_dirs[0] = hippt::normalize(vertex_A - shading_position);
	polygon.vertex_dirs[1] = hippt::normalize(vertex_B - shading_position);
	polygon.vertex_dirs[2] = hippt::normalize(vertex_C - shading_position);

	// Prepare a Householder transform that maps vertex 0 onto (+/-1, 0, 0). We
	// only store the yz-components of that Householder vector and a factor of
	// 2.0f / sqrt(abs(polygon.vertex_dirs[0].x) + 1.0f) is pulled in there to
	// save on multiplications later. This approach is necessary to avoid
	// numerical instabilities in determinant computation below.
	float householder_sign = (polygon.vertex_dirs[0].x > 0.0f) ? -1.0f : 1.0f;
	float2 householder_yz = make_float2(polygon.vertex_dirs[0].y, polygon.vertex_dirs[0].z) * (1.0f / (abs(polygon.vertex_dirs[0].x) + 1.0f));
	// Compute solid angles and prepare sampling
	polygon.solid_angle = 0.0f;
	float previous_dot_1_2 = hippt::dot(polygon.vertex_dirs[0], polygon.vertex_dirs[1]);
	
	for (unsigned int i = 0; i != MAX_POLYGON_VERTEX_COUNT - 2; ++i) 
	{
		if (i >= 1 && i + 2 >= vertex_count) break;
		// We look at one triangle of the triangle fan at a time
		float3 vertices[3] = {
			polygon.vertex_dirs[i + 1],
			polygon.vertex_dirs[0],
			polygon.vertex_dirs[i + 2] };
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
		polygon.fan_solid_angles[i] = polygon.solid_angle;
		// Some intermediate results from above help us with sampling
		polygon.triangle_parameters[i] = make_float3(simplex_volume, dot_0_2_plus_1_2, one_plus_dot_0_1);
	}

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

HIPRT_DEVICE float3 map_direction_to_triangle_point(
	float3 omega,                  // unit direction from shading point
	float3 V0, float3 V1, float3 V2, // triangle vertices in world space
	float3 triangle_normal,             // triangle normal (unit, world space)
	float3 shading_position,
	float   pdf_omega,             // PDF wrt solid-angle for this sample
	float& out_pdf_area)
{
	const float EPS = 1e-8f;
	float3 v0_rel = V0 - shading_position;
	float denom = hippt::dot(omega, triangle_normal);

	if (fabs(denom) < EPS) 
	{
		// nearly parallel: fallback (return V0 and zero pdf). Choose robust policy.
		out_pdf_area = 0.0f;

		return V0;
	}

	float t = hippt::dot(v0_rel, triangle_normal) / denom;
	if (t <= 0.0f) 
	{
		// behind the shading point or numerical; fallback
		out_pdf_area = 0.0f;

		return V0;
	}

	float3 point = shading_position + omega * t;

	// area PDF conversion: p_A = p_omega * (cos_theta / r^2)
	float cos_theta = hippt::max(0.0f, -hippt::dot(triangle_normal, omega));
	out_pdf_area = pdf_omega * (cos_theta / (t * t));

	return point;
}

/*! Given the output of prepare_solid_angle_polygon_sampling(), this function
	maps the given random numbers in the range from 0 to 1 to a normalized
	direction vector providing a sample of the solid angle of the polygon in
	the original space (used for arguments of
	prepare_solid_angle_polygon_sampling()). Samples are distributed in
	proportion to solid angle assuming uniform inputs.*/
HIPRT_DEVICE float3 sample_solid_angle_polygon(solid_angle_polygon_t polygon, float3 vertex_A, float3 vertex_B, float3 vertex_C, float3 shading_point, float3 geometric_normal, float2 random_numbers, float& out_area_pdf)
{
	// Decide which triangle needs to be sampled
	float target_solid_angle = polygon.solid_angle * random_numbers.x;
	float3 parameters = polygon.triangle_parameters[0];
	float3 vertices[3] = { polygon.vertex_dirs[1], polygon.vertex_dirs[0], polygon.vertex_dirs[2] };

	// Construct a new vertex 2 on the arc between vertices 0 and 2 such that
	// the resulting triangle has solid angle subtriangle_solid_angle
	float2 cos_sin = make_float2(hippt::intrin_cosf(0.5f * target_solid_angle), hippt::intrin_sinf(0.5f * target_solid_angle));
	float3 offset = vertices[0] * (parameters.x * cos_sin.x - parameters.y * cos_sin.y) + vertices[2] * (parameters.z * cos_sin.y);
	float3 new_vertex_2 = hippt::fma(2.0f * make_float3(hippt::dot(vertices[0], offset) / hippt::dot(offset, offset)), offset, -vertices[0]);
	// Now sample the line between vertex 1 and the newly created vertex 2
	float s2 = hippt::dot(vertices[1], new_vertex_2);
	float s = mix_fma(1.0f, s2, random_numbers.y);
	float denominator = hippt::fma(-s2, s2, 1.0f);
	float t_normed = sqrt(hippt::fma(-s, s, 1.0f) / denominator);
	// s2 may exceed one due to rounding error. random_numbers[1] is the
	// limit of t_normed for s2 -> 1.
	t_normed = (denominator > 0.0f) ? t_normed : random_numbers.y;

	float3 direction = hippt::normalize(hippt::fma(-t_normed, s2, s) * vertices[1] + t_normed * new_vertex_2);

	return map_direction_to_triangle_point(direction, vertex_B, vertex_A, vertex_C, geometric_normal, shading_point, 1.0f / polygon.solid_angle, out_area_pdf);
}

#endif
