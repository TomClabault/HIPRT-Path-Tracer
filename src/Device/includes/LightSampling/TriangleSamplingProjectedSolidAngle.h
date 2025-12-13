/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_PROJECTED_SOLID_ANGLE_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_PROJECTED_SOLID_ANGLE_H

#include "Device/includes/LightSampling/LTCs/LTCs.h"
#include "Device/includes/LightSampling/TriangleSamplingPolygonClipping.h"
#include "Device/includes/LightSampling/TriangleSamplingSolidAngleCommon.h"
#include "Device/includes/ONB.h"

#include "HostDeviceCommon/Xorshift.h"

/**
 * Adapted from the implementation given with the paper from Cristoph Peters,
 * [BRDF Importance Sampling for Polygonal Lights, 2021]
 */

 /*! This structure carries intermediate results that only need to be computed
	once per polygon and shading point to take samples proportional to
	projected solid angle.*/
struct projected_solid_angle_triangle_t 
{
	//! The number of vertices that form the polygon
	unsigned int vertex_count = 0;
	/*! The x- and y-coordinates of each polygon vertex in a coordinate system
		where the normal is the z-axis. The vertices are sorted
		counterclockwise.*/
	float2 vertices[MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING];
	/*! For each vertex in vertices, this vector describes the ellipse for the
		next edge in counterclockwise direction. The last entry is meaningless,
		except in the central case. For vertex 0, it holds the outer ellipse.
		\see ellipse_from_edge() */
	float2 ellipses[MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING];
	//! The inner ellipse adjacent to vertex 0. If the x-component is positive,
	//! the central case is present.
	float2 inner_ellipse_0 = make_float2(0.0f, 0.0f);
	/*! At index i, this array holds the projected solid angle of the polygon
		in the sector between (sorted) vertices i and (i + 1) % vertex_count
		In the central case, entry vertex_count - 1 is meaningful, otherwise
		not.*/
	float sector_projected_solid_angles[MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING];
	//! The total projected solid angle of the polygon
	float projected_solid_angle = 0.0f;

#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
	// LTC lobe sampled during the preparation of the projected solid angle triangle
	LTCLobe ltc_lobe;
	float ltc_lobe_pdf;
#endif
};


//! Computes a * b - c * d with at most 1.5 ulps of error in the result. See
//! https://pharr.org/matt/blog/2019/11/03/difference-of-floats.html or
//! Claude-Pierre Jeannerod, Nicolas Louvet and Jean-Michel Muller, 2013,
//! Further analysis of Kahan's algorithm for the accurate computation of 2x2
//! determinants, AMS Mathematics of Computation 82:284,
//! https://doi.org/10.1090/S0025-5718-2013-02679-8
HIPRT_DEVICE float kahan(float a, float b, float c, float d) 
{
	// Uncomment the line below to improve efficiency but reduce accuracy
	// return a * b - c * d;
	float cd = c * d;
	float error = hippt::fma(c, d, -cd);
	float result = hippt::fma(a, b, -cd);
	return result - error;
}


//! Implements a cross product using Kahan's algorithm for every single entry,
//! i.e. the error in each output entry is at most 1.5 ulps
HIPRT_DEVICE float3 cross_stable(float3 lhs, float3 rhs) 
{
	return make_float3(
		kahan(lhs.y, rhs.z, lhs.z, rhs.y),
		kahan(lhs.z, rhs.x, lhs.x, rhs.z),
		kahan(lhs.x, rhs.y, lhs.y, rhs.x)
	);
}


//! \return The given vector, rotated 90 degrees counterclockwise around the
//! 		origin
HIPRT_DEVICE float2 rotate_90(float2 input_vector)
{
	return make_float2(-input_vector.y, input_vector.x);
}


//! \return true iff the given ellipse is marked as inner ellipse, i.e. iff the
//!		spherical polygon that is bounded by it is further away from the zenith
//!		than this ellipse.
HIPRT_DEVICE bool is_inner_ellipse(float2 ellipse)
{
	// If the implementation below causes you trouble, e.g. because you want to
	// port to a different language, you may replace it by the commented line
	// below but it will lead to seldom artifacts when ellipse.x == 0.0f.
	// return ellipse.x < 0.0f;

	// Extract the sign bit from ellipse.x (to be able to tell apart +0 and -0)
	return (hippt::float_as_uint(ellipse.x) & 0x80000000) != 0;
}


//! \return true iff the given polygon contains the zenith (also known as
//!		normal vector).
HIPRT_DEVICE bool is_central_case(projected_solid_angle_triangle_t polygon) 
{
	return polygon.inner_ellipse_0.x > 0.0f;
}


/*! Takes the great circle for the plane through the origin and the given two
	points and constructs an ellipse for its projection to the xy-plane.
	\return A vector ellipse such that a point is on the ellipse if and only if
		hippt::dot(ellipse, point) * hippt::dot(ellipse, point) + hippt::dot(point, point) == 1.0f.
		In other words, it is a normal vector of the great circle in half-
		vector space. The sign bit of x encodes whether the edge runs clockwise
		from vertex_0 to vertex_1 (inner ellipse) or not.
	\see is_inner_ellipse() */
HIPRT_DEVICE float2 ellipse_from_edge(float3 vertex_0, float3 vertex_1, bool DEBUG = false) 
{
	float3 normal = cross_stable(vertex_0, vertex_1);
	float scaling = 1.0f / normal.z;
	scaling = is_inner_ellipse(make_float2(normal.x, normal.y)) ? -scaling : scaling;
	float2 ellipse = make_float2(normal.x, normal.y) * scaling;

	// By convention, degenerate ellipses are outer ellipses, i.e. the first
	// component is infinite
	if (normal.z == 0.0f)
		ellipse.x = hippt::Infinity();

	return ellipse;
}


//! Transforms the given point using the matrix that characterizes the given
//! ellipse (as produced by ellipse_from_edge()). To be precise, this matrix is
//! identity + outer_product(ellipse, ellipse).
HIPRT_DEVICE float2 ellipse_transform(float2 ellipse, float2 point) 
{
	return hippt::fma(make_float2(hippt::dot(ellipse, point)), ellipse, point);
}


//! Given an ellipse in the format produced by ellipse_from_edge(), this
//! function returns the determinant of the matrix characterizing this
//! ellipse.
HIPRT_DEVICE float get_ellipse_det(float2 ellipse)
{
	return hippt::fma(ellipse.x, ellipse.x, hippt::fma(ellipse.y, ellipse.y, 1.0f));
}

//! Returns the reciprocal square root of the ellipse determinant produced by
//! get_ellipse_det().
HIPRT_DEVICE float get_ellipse_rsqrt_det(float2 ellipse) 
{
	return hippt::rsqrt(get_ellipse_det(ellipse));
}

//! \return Reciprocal square of get_ellipse_direction_factor(ellipse, dir)
HIPRT_DEVICE float get_ellipse_direction_factor_rsq(float2 ellipse, float2 dir) 
{
	float ellipse_dot_dir = hippt::dot(ellipse, dir);
	float dir_dot_dir = hippt::dot(dir, dir);
	return hippt::fma(ellipse_dot_dir, ellipse_dot_dir, dir_dot_dir);
}

/*! Computes a factor by which a direction vector has to be multiplied to
	obtain a point on the given ellipse.
	\param ellipse An ellipse as produced by ellipse_from_edge().
	\param dir The direction vector to be scaled onto the ellipse.
	\return get_ellipse_direction_factor(ellipse, dir) * dir is a point on
		the ellipse.*/
HIPRT_DEVICE float get_ellipse_direction_factor(float2 ellipse, float2 dir) 
{
	return hippt::rsqrt(get_ellipse_direction_factor_rsq(ellipse, dir));
}

//! Like get_ellipse_direction_factor() but assumes that the given direction is
//! normalized. Faster.
HIPRT_DEVICE float get_ellipse_normalized_direction_factor(float2 ellipse, float2 normalized_dir) 
{
	float ellipse_dot_dir = hippt::dot(ellipse, normalized_dir);
	return hippt::rsqrt(hippt::fma(ellipse_dot_dir, ellipse_dot_dir, 1.0f));
}

//! Helper for get_area_between_ellipses_in_sector() and
//! sample_sector_between_ellipses()
HIPRT_DEVICE float get_area_between_ellipses_in_sector_from_tangents(float inner_rsqrt_det, float inner_tangent, float outer_rsqrt_det, float outer_tangent) 
{
	float inner_area = inner_rsqrt_det * positive_atan(inner_tangent);
	float result = hippt::fma(outer_rsqrt_det, positive_atan(outer_tangent), -inner_area);

	// Sort out NaNs and negative results
	return (result > 0.0f) ? (0.5f * result) : 0.0f;
}

/*! Returns the signed area between the given outer and inner ellipses within
	the sector enclosed by dir_0 and dir_1. Besides ellipses as produced by
	ellipse_from_edge(), you also have to pass output of
	get_ellipse_rsqrt_det(). Faster than calling get_ellipse_area_in_sector()
	twice.*/
HIPRT_DEVICE float get_area_between_ellipses_in_sector(float2 inner_ellipse, float inner_rsqrt_det, float2 outer_ellipse, float outer_rsqrt_det, float2 dir_0, float2 dir_1) 
{
	float det_dirs = hippt::max(+0.0f, hippt::dot(dir_1, rotate_90(dir_0)));

	float inner_dot = inner_rsqrt_det * hippt::dot(dir_0, ellipse_transform(inner_ellipse, dir_1));
	float outer_dot = outer_rsqrt_det * hippt::dot(dir_0, ellipse_transform(outer_ellipse, dir_1));

	return get_area_between_ellipses_in_sector_from_tangents(
		inner_rsqrt_det, det_dirs / inner_dot,
		outer_rsqrt_det, det_dirs / outer_dot);
}

/*! Computes the area for the intersection of the given ellipse and the sector
	between the given two directions (going counterclockwise from dir_0 to
	dir_1 for at most 180 degrees). The scaling of the directions is
	irrelevant.
	\see ellipse_from_edge() */
HIPRT_DEVICE float get_ellipse_area_in_sector(float2 ellipse, float2 dir_0, float2 dir_1) 
{
	float ellipse_rsqrt_det = get_ellipse_rsqrt_det(ellipse);
	float det_dirs = hippt::max(+0.0f, hippt::dot(dir_1, rotate_90(dir_0)));
	float ellipse_dot = ellipse_rsqrt_det * hippt::dot(dir_0, ellipse_transform(ellipse, dir_1));
	float area = 0.5f * ellipse_rsqrt_det * positive_atan(det_dirs / ellipse_dot);

	// For degenerate ellipses, the result may be NaN but must be 0.0f
	return (ellipse_rsqrt_det > 0.0f) ? area : 0.0f;
}


/*! Swaps vertices lhs and rhs (along with corresponding ellipses) of the given
	polygon if the shorter path from lhs to rhs is clockwise. If the vertices
	have identical directions in the xy-plane, vertices with degenerate
	ellipses come first.
	\note To avoid costly register spilling, lhs and rhs must be compile time
		constants.*/
HIPRT_DEVICE void compare_and_swap(projected_solid_angle_triangle_t& polygon, unsigned int lhs, unsigned int rhs) 
{
	float2 lhs_copy = polygon.vertices[lhs];
	// This line is designed to agree with the implementation of cross_stable
	// for the z-coordinate, which determines if ellipses are inner or outer
	float normal_z = kahan(lhs_copy.x, -polygon.vertices[rhs].y, lhs_copy.y, -polygon.vertices[rhs].x);
	// Tie breaker: If both vertices are at the same angle (i.e. on a common
	// great circle through the zenith), the one with the degenerate ellipse
	// comes first

	bool swap = (normal_z == 0.0f) ? (hippt::is_inf(polygon.ellipses[rhs].x)) : (normal_z > 0.0f);

	polygon.vertices[lhs] = swap ? polygon.vertices[rhs] : lhs_copy;
	polygon.vertices[rhs] = swap ? lhs_copy : polygon.vertices[rhs];
	lhs_copy = polygon.ellipses[lhs];
	polygon.ellipses[lhs] = swap ? polygon.ellipses[rhs] : lhs_copy;
	polygon.ellipses[rhs] = swap ? lhs_copy : polygon.ellipses[rhs];
}


//! Sorts the vertices of the given convex polygon counterclockwise using a
//! special sorting network. For non-convex polygons, the method may fail.
HIPRT_DEVICE void sort_convex_polygon_vertices(projected_solid_angle_triangle_t& polygon) 
{
	if (polygon.vertex_count == 3) {
		compare_and_swap(polygon, 1, 2);
	}
#if MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING >= 4
	else if (polygon.vertex_count == 4) {
		compare_and_swap(polygon, 1, 3);
	}
#endif
#if MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING >= 5
	else if (polygon.vertex_count == 5) {
		compare_and_swap(polygon, 2, 4);
		compare_and_swap(polygon, 1, 3);
		compare_and_swap(polygon, 1, 2);
		compare_and_swap(polygon, 0, 3);
		compare_and_swap(polygon, 3, 4);
	}
#endif
#if MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING >= 6
	else if (polygon.vertex_count == 6) {
		compare_and_swap(polygon, 3, 5);
		compare_and_swap(polygon, 2, 4);
		compare_and_swap(polygon, 1, 5);
		compare_and_swap(polygon, 0, 4);
		compare_and_swap(polygon, 4, 5);
		compare_and_swap(polygon, 1, 3);
	}
#endif
#if MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING >= 7
	else if (polygon.vertex_count == 7) {
		compare_and_swap(polygon, 2, 5);
		compare_and_swap(polygon, 1, 6);
		compare_and_swap(polygon, 5, 6);
		compare_and_swap(polygon, 3, 4);
		compare_and_swap(polygon, 0, 4);
		compare_and_swap(polygon, 4, 6);
		compare_and_swap(polygon, 1, 3);
		compare_and_swap(polygon, 3, 5);
		compare_and_swap(polygon, 4, 5);
	}
#endif
#if MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING >= 8
	else if (polygon.vertex_count == 8) {
		compare_and_swap(polygon, 2, 6);
		compare_and_swap(polygon, 3, 7);
		compare_and_swap(polygon, 1, 5);
		compare_and_swap(polygon, 0, 4);
		compare_and_swap(polygon, 4, 6);
		compare_and_swap(polygon, 5, 7);
		compare_and_swap(polygon, 6, 7);
		compare_and_swap(polygon, 4, 5);
		compare_and_swap(polygon, 1, 3);
	}
#endif
	// This comparison is shared by all sorting networks
	compare_and_swap(polygon, 0, 2);
#if MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING >= 4
	if (polygon.vertex_count >= 4) {
		// This comparison is shared by all sorting networks except the one for
		// triangles
		compare_and_swap(polygon, 2, 3);
	}
#endif
	// This comparison is shared by all sorting networks
	compare_and_swap(polygon, 0, 1);
}

/*! Prepares all intermediate values to sample a convex polygon proportional to
	projected solid angle.
	\param vertex_count Number of vertices forming the polygon (at least 3).
	\param vertices List of vertex locations in a coordinate system where the
		shading position is the origin and the normal is the z-axis. The
		polygon should be already clipped against the plane z=0. If
		vertex_count < MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING, the first vertex has to be
		repeated at vertex_count. They need not be normalized but if you
		encounter issues with under- or overflow (e.g. NaN or INF outputs),
		normalization may help. The polygon must be convex, and the winding of
		the vertices as seen from the origin must be clockwise. No three
		vertices should be collinear.
	\return Intermediate values for sampling.*/
HIPRT_DEVICE projected_solid_angle_triangle_t prepare_projected_solid_angle_triangle_sampling(unsigned int vertex_count, float3 vertices_clockwise_order[MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING])
{
	projected_solid_angle_triangle_t polygon;
	// Copy vertices and assign ellipses
	polygon.vertex_count = vertex_count;
	if (vertex_count == 0)
		return polygon;

	polygon.inner_ellipse_0 = make_float2(1.0f, 0.0f);
	polygon.vertices[0] = make_float2(vertices_clockwise_order[0].x, vertices_clockwise_order[0].y); 
	polygon.ellipses[0] = ellipse_from_edge(vertices_clockwise_order[0], vertices_clockwise_order[1]);

	float2 previous_ellipse = polygon.ellipses[0];

UNROLL_LOOP
	for (unsigned int i = 1; i != MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING; ++i)
	{
		polygon.vertices[i] = make_float2(vertices_clockwise_order[i].x, vertices_clockwise_order[i].y);
		if (i > 2 && i == polygon.vertex_count) break;
		float2 ellipse = ellipse_from_edge(vertices_clockwise_order[i], vertices_clockwise_order[(i + 1) % MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING]);
		bool ellipse_inner = is_inner_ellipse(ellipse);
		// If the edge is an inner edge, the order is going to flip
		polygon.ellipses[i] = ellipse_inner ? previous_ellipse : ellipse;
		// In doing so, we drop one ellipse, unless we store it explicitly
		polygon.inner_ellipse_0 = (is_inner_ellipse(previous_ellipse) && !ellipse_inner) ? previous_ellipse : polygon.inner_ellipse_0;
		previous_ellipse = ellipse;
	}

	// Same thing for the first vertex (i.e. here we close the loop)
	float2 ellipse = polygon.ellipses[0];
	bool ellipse_inner = is_inner_ellipse(ellipse);

	polygon.ellipses[0] = ellipse_inner ? previous_ellipse : ellipse;	
	polygon.inner_ellipse_0 = (is_inner_ellipse(previous_ellipse) && !ellipse_inner) ? previous_ellipse : polygon.inner_ellipse_0;
	// Compute projected solid angles per sector and in total
	polygon.projected_solid_angle = 0.0f;

	if (is_central_case(polygon)) 
	{
		// In the central case, we have polygon.vertex_count sectors, each
		// bounded by a single ellipse
UNROLL_LOOP
		for (unsigned int i = 0; i != MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING; ++i) 
		{
			if (i > 2 && i == polygon.vertex_count) break;
			polygon.sector_projected_solid_angles[i] = get_ellipse_area_in_sector(polygon.ellipses[i], polygon.vertices[i], polygon.vertices[(i + 1) % MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING]);
			polygon.projected_solid_angle += polygon.sector_projected_solid_angles[i];
		}
	}
	else 
	{
		// Sort vertices counter clockwise
		sort_convex_polygon_vertices(polygon);

		// There are polygon.vertex_count - 1 sectors, each bounded by an inner
		// and an outer ellipse
		float2 inner_ellipse = polygon.inner_ellipse_0;
		float inner_rsqrt_det = get_ellipse_rsqrt_det(inner_ellipse);
		float2 outer_ellipse = make_float2(0.0f, 0.0f);
		float outer_rsqrt_det = 0.0f;

UNROLL_LOOP
		for (unsigned int i = 0; i != MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING - 1; ++i)
		{
			if (i > 1 && i + 1 == polygon.vertex_count) break;

			float2 vertex_ellipse = polygon.ellipses[i];
			bool vertex_inner = is_inner_ellipse(vertex_ellipse);
			float vertex_rsqrt_det = get_ellipse_rsqrt_det(vertex_ellipse);

			if (i == 0) 
			{
				outer_ellipse = vertex_ellipse;
				outer_rsqrt_det = vertex_rsqrt_det;
			}
			else 
			{
				inner_ellipse = vertex_inner ? vertex_ellipse : inner_ellipse;
				inner_rsqrt_det = vertex_inner ? vertex_rsqrt_det : inner_rsqrt_det;
				outer_ellipse = vertex_inner ? outer_ellipse : vertex_ellipse;
				outer_rsqrt_det = vertex_inner ? outer_rsqrt_det : vertex_rsqrt_det;
			}

			polygon.sector_projected_solid_angles[i] = get_area_between_ellipses_in_sector(
				inner_ellipse, inner_rsqrt_det, outer_ellipse, outer_rsqrt_det, polygon.vertices[i], polygon.vertices[i + 1]);
			polygon.projected_solid_angle += polygon.sector_projected_solid_angles[i];
		}
	}

	return polygon;
}

HIPRT_DEVICE projected_solid_angle_triangle_t prepare_projected_solid_angle_triangle_sampling_from_world_space_internal(const HIPRTRenderData& render_data,
	float3 vertex_A_world_space, float3 vertex_B_world_space, float3 vertex_C_world_space,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe, float ltc_lobe_pdf)
{
#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
	/**
	 * With the view direction in the x-z plane + LTC transform
	 */
	 // Building a shading space where the shading point is the origin, the shading normal
	 // is the z axis, and the view direction lies in the x-z plane
	float3 T, B;
	build_ONB_XZ_plane(shading_normal, T, B, view_direction);
	float3 vertex_A_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_A_world_space - shading_point);
	float3 vertex_B_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_B_world_space - shading_point);
	float3 vertex_C_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_C_world_space - shading_point);

	// Normalizing vertices helps a bit with float numerical precision
	vertex_A_local = hippt::normalize(vertex_A_local);
	vertex_B_local = hippt::normalize(vertex_B_local);
	vertex_C_local = hippt::normalize(vertex_C_local);

	// Shading space to cosine space such that we sample the projected
	// solid angle of the triangle but transformed by the LTC
	float NoV = hippt::dot(view_direction, shading_normal);
	vertex_A_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_A_local, material, ltc_lobe);
	vertex_B_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_B_local, material, ltc_lobe);
	vertex_C_local = ltc_transform_shading_to_cosine(render_data, NoV, vertex_C_local, material, ltc_lobe);
#else
	float3 T, B;
	build_ONB(shading_normal, T, B);

	float3 vertex_A_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_A_world_space - shading_point);
	float3 vertex_B_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_B_world_space - shading_point);
	float3 vertex_C_local = world_to_local_frame_non_normalized(T, B, shading_normal, vertex_C_world_space - shading_point);
#endif

	// The vertices array reorganizes the vertices in clockwise order
	float3 vertices_local_space[MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING] = { vertex_A_local, vertex_C_local, vertex_B_local };
	unsigned int clipped_vertex_count = clip_polygon(3, vertices_local_space);

	vertices_local_space[0] = hippt::normalize(vertices_local_space[0]);
	vertices_local_space[1] = hippt::normalize(vertices_local_space[1]);
	vertices_local_space[2] = hippt::normalize(vertices_local_space[2]);
	vertices_local_space[3] = hippt::normalize(vertices_local_space[3]);

	//// Normalizing the vertices for better fp32 precision
	//float min_len = hippt::Infinity(), max_len = 0.0f;
	//for (unsigned int i = 0; i < clipped_vertex_count; ++i)
	//{
	//	float l = hippt::length(vertices_local_space[i]);

	//	min_len = hippt::min(min_len, l);
	//	max_len = hippt::max(max_len, l);
	//}

	//if (min_len == 0.0f || max_len / hippt::max(min_len, 1e-30f) > 1e3f)
	//	// Scale range too large or a zero-length vertex --> normalize
	//	for (unsigned int i = 0; i < clipped_vertex_count; ++i)
	//		vertices_local_space[i] = hippt::normalize(vertices_local_space[i]);

	projected_solid_angle_triangle_t prepared_triangle = prepare_projected_solid_angle_triangle_sampling(clipped_vertex_count, vertices_local_space);
#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
	prepared_triangle.ltc_lobe = ltc_lobe;
	prepared_triangle.ltc_lobe_pdf = ltc_lobe_pdf;
#endif

	return prepared_triangle;
}

HIPRT_DEVICE projected_solid_angle_triangle_t prepare_projected_solid_angle_triangle_sampling_from_world_space(const HIPRTRenderData& render_data,
	float3 vertex_A_world_space, float3 vertex_B_world_space, float3 vertex_C_world_space,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material, Xorshift32Generator& rng)
{
#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
	float ltc_lobe_pdf;
	LTCLobe ltc_lobe = ltc_lobe_sample(material, rng, ltc_lobe_pdf);

	return prepare_projected_solid_angle_triangle_sampling_from_world_space_internal(
		render_data,
		vertex_A_world_space, vertex_B_world_space, vertex_C_world_space,
		shading_point, view_direction, shading_normal,
		material, ltc_lobe, ltc_lobe_pdf);
#else
	return prepare_projected_solid_angle_triangle_sampling_from_world_space_internal(
		render_data,
		vertex_A_world_space, vertex_B_world_space, vertex_C_world_space,
		shading_point, view_direction, shading_normal,
		material, 
		// Not using LTCs, we don't care about these 2 parameters, just setting some defaults
		LTCLobe::DIFFUSE_LOBE, 1.0f);
#endif
}

HIPRT_DEVICE float projected_solid_angle_triangle_solid_angle_pdf_internal(const HIPRTRenderData& render_data,
	float triangle_projected_solid_angle, float NoL,
	float3 view_direction, float3 shading_normal, float3 sampled_dir_shading_space,
	const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
	float ltc_lobe_pdf = ltc_lobe_eval_pdf(material, ltc_lobe);
	if (ltc_lobe_pdf == 0.0f)
		return 0.0f;

	float pdf_solid_angle;
	if (NoL < 1.0e-8f)
		pdf_solid_angle = 0.0f;
	else
	{
		pdf_solid_angle = NoL / triangle_projected_solid_angle;
		pdf_solid_angle *= ltc_jacobian(render_data, hippt::dot(view_direction, shading_normal), sampled_dir_shading_space, material, ltc_lobe);
		pdf_solid_angle *= ltc_lobe_pdf;
	}
#else
	// TODO does the dot product match here with sampled_dir_shading_space.z?
	float pdf_solid_angle = hippt::max(0.0f, NoL) / triangle_projected_solid_angle;
	// float pdf_solid_angle = hippt::max(0.0f, hippt::dot(shading_normal, to_light_direction)) / projected_solid_angle_triangle.projected_solid_angle;
#endif

	return pdf_solid_angle;
}

HIPRT_DEVICE float projected_solid_angle_triangle_solid_angle_pdf_internal(const HIPRTRenderData& render_data,
	float3 vertex_A_world_space, float3 vertex_B_world_space, float3 vertex_C_world_space,
	float3 shading_point, float3 view_direction, float3 shading_normal,
	float3 point_on_light,
	const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	float ltc_lobe_pdf = ltc_lobe_eval_pdf(material, ltc_lobe);
	if (ltc_lobe_pdf == 0.0f)
		return 0.0f;

	projected_solid_angle_triangle_t projected_solid_angle_triangle = prepare_projected_solid_angle_triangle_sampling_from_world_space_internal(
		render_data,
		vertex_A_world_space, vertex_B_world_space, vertex_C_world_space,
		shading_point, view_direction, shading_normal,
		material, ltc_lobe, ltc_lobe_pdf);

	if (projected_solid_angle_triangle.vertex_count == 0 || projected_solid_angle_triangle.projected_solid_angle == 0.0f)
		// The whole polygon is below the hemisphere, clipping returned 0 vertices
		return 0.0f;

	float3 to_light_direction = hippt::normalize(point_on_light - shading_point);

	float3 sampled_dir_shading_space;
	float3 sampled_dir_cosine_space;
#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
	// Jacobian of the LTC transform
	float3 T, B;
	build_ONB_XZ_plane(shading_normal, T, B, view_direction);
	sampled_dir_shading_space = world_to_local_frame(T, B, shading_normal, to_light_direction);
	sampled_dir_cosine_space = hippt::normalize(ltc_transform_shading_to_cosine(render_data, hippt::dot(view_direction, shading_normal), sampled_dir_shading_space, material, ltc_lobe));

	float pdf_solid_angle = projected_solid_angle_triangle_solid_angle_pdf_internal(
		render_data,
		projected_solid_angle_triangle.projected_solid_angle,
		sampled_dir_cosine_space.z,
		view_direction, shading_normal, sampled_dir_shading_space,
		material, ltc_lobe);
#else
	float pdf_solid_angle = projected_solid_angle_triangle_solid_angle_pdf_internal(render_data,
		projected_solid_angle_triangle.projected_solid_angle, hippt::dot(shading_normal, to_light_direction),
		view_direction, shading_normal, make_float3(0.0f, 0.0f, 0.0f),
		material, ltc_lobe);
#endif

	return pdf_solid_angle;
}

HIPRT_DEVICE float projected_solid_angle_triangle_solid_angle_pdf(const HIPRTRenderData& render_data,
	float triangle_projected_solid_angle, float NoL,
	float3 view_direction, float3 shading_normal, float3 sampled_dir_shading_space,
	const DeviceUnpackedEffectiveMaterial& material)
{
	float out_pdf = 0.0f;

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		material, LTCLobe::COAT_LOBE);

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		material, LTCLobe::METALLIC_LOBE);

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		material, LTCLobe::SPECULAR_LOBE);

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(render_data,
		triangle_projected_solid_angle, NoL,
		view_direction, shading_normal, sampled_dir_shading_space,
		material, LTCLobe::DIFFUSE_LOBE);

	return out_pdf;
}

HIPRT_DEVICE float projected_solid_angle_triangle_solid_angle_pdf(const HIPRTRenderData& render_data, 
	float3 vertex_A_world_space, float3 vertex_B_world_space, float3 vertex_C_world_space,
	float3 shading_point, float3 view_direction, float3 shading_normal, float3 point_on_light,
	const DeviceUnpackedEffectiveMaterial& material)
{
	float out_pdf = 0.0f;

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(
		render_data,
		vertex_A_world_space, vertex_B_world_space, vertex_C_world_space,
		shading_point, view_direction, shading_normal, point_on_light,
		material, LTCLobe::COAT_LOBE);

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(
		render_data,
		vertex_A_world_space, vertex_B_world_space, vertex_C_world_space,
		shading_point, view_direction, shading_normal, point_on_light,
		material, LTCLobe::METALLIC_LOBE);

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(
		render_data,
		vertex_A_world_space, vertex_B_world_space, vertex_C_world_space,
		shading_point, view_direction, shading_normal, point_on_light,
		material, LTCLobe::SPECULAR_LOBE);

	out_pdf += projected_solid_angle_triangle_solid_angle_pdf_internal(
		render_data,
		vertex_A_world_space, vertex_B_world_space, vertex_C_world_space,
		shading_point, view_direction, shading_normal, point_on_light,
		material, LTCLobe::DIFFUSE_LOBE);

	return out_pdf;
}

/*! \return A scalar multiple of rhs that is not too far from being normalized.
		For the result, length() returns something between sqrt(2.0f) and 8.0f.
		The sign gets flipped such that the dot product of semi_circle and the
		result is non-negative.
	\note Introduces less latency than normalize() and does not use special
		functions. Useful to avoid under- and overflow when working with
		homogeneous coordinates. The result is undefined if rhs is zero.*/
HIPRT_DEVICE float2 normalize_approx_and_flip(float2 rhs, float2 semi_circle) 
{
	float scaling = hippt::abs(rhs.x) + hippt::abs(rhs.y);
	// By flipping each bit on the exponent E, we turn it into 1 - E, which is
	// close enough to a reciprocal.
	scaling = hippt::uint_as_float(hippt::float_as_uint(scaling) ^ 0x7F800000u);
	// If the line above causes you any sort of trouble (e.g. because you want
	// to port the code to another language or you are doing differentiable
	// rendering), just use this one instead:
	// scaling = 1.0f / scaling;
	// Flip the sign as needed
	scaling = (hippt::dot(rhs, semi_circle) >= 0.0f) ? scaling : -scaling;

	return scaling * rhs;
}

/*! Returns a solution to the given homogeneous quadratic equation, i.e. a
	non-zero vector root such that hippt::dot(root, quadratic * root) == 0.0f. The
	returned root depends continuously on quadratic. Pass -quadratic if you
	want the other root.
	\note The implementation is as proposed by Blinn, except that we do not
	have a special case for quadratic[0][1] + quadratic[1][0] == 0.0f. Unlike
	the standard quadratic formula, it allows us to postpone a division and is
	stable in all cases.
	James F. Blinn 2006, How to Solve a Quadratic Equation, Part 2, IEEE
	Computer Graphics and Applications 26:2 https://doi.org/10.1109/MCG.2006.35
*/
HIPRT_DEVICE float2 solve_homogeneous_quadratic(float2x2 quadratic) 
{
	float coeff_xy = 0.5f * (quadratic.m[0][1] + quadratic.m[1][0]);
	float sqrt_discriminant = hippt::sqrt(hippt::max(0.0f, coeff_xy * coeff_xy - quadratic.m[0][0] * quadratic.m[1][1]));
	float scaled_root = hippt::abs(coeff_xy) + sqrt_discriminant;
	return (coeff_xy >= 0.0f) ? make_float2(scaled_root, -quadratic.m[0][0]) : make_float2(quadratic.m[1][1], scaled_root);
}

/*! Generates a sample between two ellipses and in a specified sector. The
	sample is distributed uniformly with respect to the area measure.
	\param random_numbers A pair of independent uniform random numbers on [0,1]
	\param target_area rand_1 multiplied by the projected solid
		angle of the area to be sampled.
	\param inner_ellipse, outer_ellipse The inner and outer ellipse, as
		 produced by ellipse_from_edge().
	\param dir_0, dir_1 Two direction vectors bounding the sector. They need
		not be normalized.
	\param iteration_count The number of iterations to perform. Lower values
		trade speed for bias. Two iterations give practically no bias.
	\return The sample in Cartesian coordinates.*/
HIPRT_DEVICE float2 sample_sector_between_ellipses(float2 random_numbers, float target_area, float2 inner_ellipse, float2 outer_ellipse, float2 dir_0, float2 dir_1, unsigned int iteration_count) 
{
	// For the initialization, split the sector in half
	float2 quad_dirs[3];
	quad_dirs[0] = hippt::normalize(dir_0);
	quad_dirs[2] = hippt::normalize(dir_1);
	quad_dirs[1] = quad_dirs[0] + quad_dirs[2];
	// Compute where these lines intersect the ellipses. The six intersection
	// points define two adjacent quads.
	float normalization_factor[2][3] = 
	{
		{
			get_ellipse_normalized_direction_factor(inner_ellipse, quad_dirs[0]),
			get_ellipse_direction_factor(inner_ellipse, quad_dirs[1]),
			get_ellipse_normalized_direction_factor(inner_ellipse, quad_dirs[2])
		},
		{
			get_ellipse_normalized_direction_factor(outer_ellipse, quad_dirs[0]),
			get_ellipse_direction_factor(outer_ellipse, quad_dirs[1]),
			get_ellipse_normalized_direction_factor(outer_ellipse, quad_dirs[2])
		}
	};

	// Compute the relative size of the areas inside these quads
	float sector_areas[2] = 
	{
		normalization_factor[1][0] * normalization_factor[1][1] - normalization_factor[0][0] * normalization_factor[0][1],
		normalization_factor[1][1] * normalization_factor[1][2] - normalization_factor[0][1] * normalization_factor[0][2]
	};

	// Now pick which of the two quads should be sampled for the
	// initialization. If it is not the second, we move data such that the
	// relevant array indices are 1 and 2 anyway.
	float target_quad_area = mix_fma(-sector_areas[0], sector_areas[1], random_numbers.x);
	quad_dirs[2] = (target_quad_area <= 0.0f) ? quad_dirs[0] : quad_dirs[2];
	normalization_factor[0][2] = (target_quad_area <= 0.0f) ? normalization_factor[0][0] : normalization_factor[0][2];
	normalization_factor[1][2] = (target_quad_area <= 0.0f) ? normalization_factor[1][0] : normalization_factor[1][2];
	target_quad_area += (target_quad_area <= 0.0f) ? sector_areas[0] : -sector_areas[1];
	// We have been a bit lazy about area computation before but now we need
	// all the factors (except for a factor of 0.5 that cancels with a 2 later)
	target_quad_area *= hippt::abs(determinant(float2x2(quad_dirs[1], quad_dirs[2])));
	// Construct normal vectors for the inner and outer edge of the selected
	// quad. We construct the normal like a half vector (i.e. by addition)
	// because it is less prone to cancellation than an approach using the edge
	// direction (i.e. subtraction of sometimes nearly identical vectors)
	float2 quad_normals[2] = 
	{
		quad_dirs[1] * normalization_factor[0][1] + quad_dirs[2] * normalization_factor[0][2],
		quad_dirs[1] * normalization_factor[1][1] + quad_dirs[2] * normalization_factor[1][2]
	};

	quad_normals[0] = ellipse_transform(inner_ellipse, quad_normals[0]);
	quad_normals[1] = ellipse_transform(outer_ellipse, quad_normals[1]);
	// Construct complete line equations
	float quad_offsets[2] = 
	{
		hippt::dot(quad_normals[0], quad_dirs[1]) * normalization_factor[0][1],
		hippt::dot(quad_normals[1], quad_dirs[1]) * normalization_factor[1][1]
	};

	// Now sample the direction within the selected quad by constructing a
	// quadratic equation. This is the initialization for the iteration.
	float2x2 quadratic = outer_product((quad_offsets[1] * normalization_factor[1][2]) * rotate_90(quad_dirs[2]), quad_normals[0]);
	quadratic = quadratic - outer_product((quad_offsets[0] * normalization_factor[0][2]) * rotate_90(quad_dirs[2]) + target_quad_area * quad_normals[0], quad_normals[1]);
	float2 current_dir = solve_homogeneous_quadratic(quadratic);

#ifndef USE_BIASED_PROJECTED_SOLID_ANGLE_SAMPLING
	// For boundary values, the initialization is perfect but the iteration may
	// be unstable, so we disable it
	float acceptable_error = 1.0e-5f;
	iteration_count = (hippt::abs(random_numbers.x - 0.5f) <= 0.5f - acceptable_error) ? iteration_count : 0;

	// Now refine this initialization iteratively
	float inner_rsqrt_det = get_ellipse_rsqrt_det(inner_ellipse);
	float outer_rsqrt_det = get_ellipse_rsqrt_det(outer_ellipse);

UNROLL_LOOP
	for (unsigned int i = 0; i < iteration_count; i++)
	{
		// Avoid under- or overflow and flip the sign so that the clamping to
		// zero below makes sense
		current_dir = normalize_approx_and_flip(current_dir, quad_dirs[1]);

		// Transform current_dir using both ellipses
		float2 inner_dir = ellipse_transform(inner_ellipse, current_dir);
		float2 outer_dir = ellipse_transform(outer_ellipse, current_dir);

		// Evaluate the objective function (reusing inner_dir and outer_dir)
		float det_dirs = hippt::max(+0.0f, hippt::dot(current_dir, rotate_90(quad_dirs[0])));
		float error = target_area - get_area_between_ellipses_in_sector_from_tangents(
			inner_rsqrt_det, det_dirs / (inner_rsqrt_det * hippt::dot(quad_dirs[0], inner_dir)),
			outer_rsqrt_det, det_dirs / (outer_rsqrt_det * hippt::dot(quad_dirs[0], outer_dir)));

		// Construct a homogeneous quadratic whose solutions include the next
		// step of the iteration
		quadratic = outer_product(inner_dir - outer_dir, rotate_90(current_dir)) - outer_product((2.0f * error) * inner_dir, outer_dir);
		current_dir = solve_homogeneous_quadratic(quadratic);
	}
#endif

	// The halved sector is at most 90 degrees large, so the dot product with
	// the half vector has to be positive
	current_dir = (hippt::dot(current_dir, quad_dirs[1]) >= 0.0f) ? current_dir : -current_dir;
	// Sample a squared radius uniformly between the two ellipses
	float inner_factor = 1.0f / get_ellipse_direction_factor_rsq(inner_ellipse, current_dir);
	float outer_factor = 1.0f / get_ellipse_direction_factor_rsq(outer_ellipse, current_dir);
	current_dir *= hippt::sqrt(mix_fma(inner_factor, outer_factor, random_numbers.y));

	return current_dir;
}

/*! Produces a sample in the solid angle of the given polygon. If the random
	numbers are uniform in [0,1]^2, the sample is uniform in the projected
	solid angle of the polygon.
	\param polygon Output of prepare_projected_solid_angle_triangle_sampling().
	\param random_numbers A uniform point in [0,1]^2.
	\return A sample on the upper hemisphere (i.e. z>=0) in Cartesian
		coordinates.*/
HIPRT_DEVICE float3 sample_point_on_triangle_projected_solid_angle_peters_2021(const HIPRTRenderData& render_data,
	float3 vertex_A, float3 vertex_B, float3 vertex_C, float3 triangle_normal, 
	float3 shading_point, float3 view_direction, float3 shading_normal,
	const DeviceUnpackedEffectiveMaterial& material,
	float& out_area_pdf,
	Xorshift32Generator& rng)
{
	projected_solid_angle_triangle_t polygon = prepare_projected_solid_angle_triangle_sampling_from_world_space(render_data,
		vertex_A, vertex_B, vertex_C, 
		shading_point, view_direction, shading_normal,
		material, rng);

	if (polygon.vertex_count == 0 || polygon.projected_solid_angle == 0.0f)
	{
		out_area_pdf = 0.0f;

		return make_float3(0.0f, 0.0f, 0.0f);
	}

	float rand_1 = rng();
	float rand_2 = rng();

	float target_projected_solid_angle = rand_1 * polygon.projected_solid_angle;
	// Distinguish between the central case
	float3 sampled_dir = make_float3(0.0f, 0.0f, 0.0f);
	float2 outer_ellipse = make_float2(0.0f, 0.0f);
	float2 dir_0 = make_float2(0.0f, 0.0f);

	if (is_central_case(polygon)) 
	{
		// Select a sector and copy the relevant attributes
UNROLL_LOOP
		for (unsigned int i = 0; i != MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING; ++i) 
		{
			if (i > 0)
				target_projected_solid_angle -= polygon.sector_projected_solid_angles[i - 1];

			outer_ellipse = polygon.ellipses[i];
			dir_0 = polygon.vertices[i];
			if ((i >= 2 && i + 1 == polygon.vertex_count) || target_projected_solid_angle < polygon.sector_projected_solid_angles[i])
				break;
		}

		// Sample a direction within the sector
		float sqrt_det = hippt::sqrt(get_ellipse_det(outer_ellipse));
		float angle = 2.0f * target_projected_solid_angle * sqrt_det;
		float2 dir_xy = (hippt::intrin_cosf(angle) * sqrt_det) * dir_0 + hippt::intrin_sinf(angle) * rotate_90(ellipse_transform(outer_ellipse, dir_0));

		sampled_dir.x = dir_xy.x;
		sampled_dir.y = dir_xy.y;

		// Sample a squared radius uniformly within the ellipse
		float sampled = hippt::sqrt(rand_2 / get_ellipse_direction_factor_rsq(outer_ellipse, make_float2(sampled_dir.x, sampled_dir.y)));
		sampled_dir.x *= sampled;
		sampled_dir.y *= sampled;
	}
	// And the decentral case
	else 
	{
		// Select a sector and copy the relevant attributes
		float sector_projected_solid_angle = 0.0f;
		float2 inner_ellipse = polygon.inner_ellipse_0;
		float2 dir_1 = make_float2(0.0f, 0.0f);

UNROLL_LOOP
		for (unsigned int i = 0; i < MAX_POLYGON_VERTEX_COUNT_PROJECTED_SOLID_ANGLE_SAMPLING - 1; i++) 
		{
			float2 vertex_ellipse = polygon.ellipses[i];

			if (i == 0)
				outer_ellipse = vertex_ellipse;
			else 
			{
				target_projected_solid_angle -= polygon.sector_projected_solid_angles[i - 1];
				bool vertex_inner = is_inner_ellipse(vertex_ellipse);
				inner_ellipse = vertex_inner ? vertex_ellipse : inner_ellipse;
				outer_ellipse = vertex_inner ? outer_ellipse : vertex_ellipse;
			}

			dir_0 = polygon.vertices[i];
			dir_1 = polygon.vertices[i + 1];
			sector_projected_solid_angle = polygon.sector_projected_solid_angles[i];

			if ((i >= 1 && i + 2 == polygon.vertex_count) || target_projected_solid_angle < sector_projected_solid_angle)
				break;
		}

		// Sample it
		rand_1 = target_projected_solid_angle / sector_projected_solid_angle;

		float2 sector = sample_sector_between_ellipses(make_float2(rand_1, rand_2), target_projected_solid_angle, inner_ellipse, outer_ellipse, dir_0, dir_1, 2);
		sampled_dir.x = sector.x;
		sampled_dir.y = sector.y;
	}

	// Construct the sample
	sampled_dir.z = hippt::sqrt(hippt::max(0.0f, hippt::fma(-sampled_dir.x, sampled_dir.x, hippt::fma(-sampled_dir.y, sampled_dir.y, 1.0f))));

	// Transform the sample back to world space
	
#if TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE
	/**
	 * View direction lies in the x-z plane + LTC.
	 */

	// From cosine space to shading space
	float ltc_lobe_pdf;
	float3 sampled_dir_shading_space = hippt::normalize(ltc_transform_cosine_to_shading(render_data, hippt::dot(view_direction, shading_normal), sampled_dir, material, polygon.ltc_lobe));

	float3 T, B;
	build_ONB_XZ_plane(shading_normal, T, B, view_direction);
	float3x3 rotation_matrix = float3x3::from_rows(T, B, shading_normal);
	// Multiplying the vector from the left to effectively
	// multiply by the transpose of the rotation matrix which is its inverse.
	//
	// This brings the direction from shading space to world space.
	float3 sampled_dir_world_space = hippt::normalize(sampled_dir_shading_space * rotation_matrix);

	float pdf_solid_angle = hippt::max(0.0f, sampled_dir.z) / polygon.projected_solid_angle;
	pdf_solid_angle *= ltc_jacobian(render_data, hippt::dot(view_direction, shading_normal), sampled_dir_shading_space, material, polygon.ltc_lobe);
	pdf_solid_angle *= polygon.ltc_lobe_pdf;

	pdf_solid_angle = projected_solid_angle_triangle_solid_angle_pdf(render_data,
		polygon.projected_solid_angle, sampled_dir.z,
		view_direction, shading_normal, sampled_dir_shading_space,
		material);
#else
	/**
	 * Simply sampling projected solid angle.
	 */
	float3 sampled_dir_world_space = hippt::normalize(local_to_world_frame(shading_normal, sampled_dir));
	float pdf_solid_angle = hippt::max(0.0f, hippt::dot(shading_normal, sampled_dir_world_space)) / polygon.projected_solid_angle;
#endif

	return map_direction_to_triangle_point(sampled_dir_world_space, vertex_A, triangle_normal, shading_point, pdf_solid_angle, out_area_pdf);
}

#endif
