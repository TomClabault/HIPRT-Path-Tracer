/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "HostDeviceCommon/HitInfo.h"
#include <hiprt/hiprt_types.h> // for hiprtRay

struct Triangle
{
	Triangle() {}
	Triangle(const float3& a, const float3& b, const float3& c) : m_a(a), m_b(b), m_c(c) {}

	float3 bbox_centroid() const;

	//From https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	inline bool intersect(const hiprtRay& ray, HitInfo& hit_info) const
	{
		const float EPSILON = 0.0000001f;
		float3 edge1, edge2, h, s, q;
		float a, f, u, v;
		edge1 = m_b - m_a;
		edge2 = m_c - m_a;

		h = hippt::cross(ray.direction, edge2);
		a = hippt::dot(edge1, h);

		if (a > -EPSILON && a < EPSILON)
			return false;    // This ray is parallel to this triangle.

		f = 1.0f / a;
		s = ray.origin - m_a;
		u = f * hippt::dot(s, h);

		if (u < 0.0f || u > 1.0f)
			return false;

		q = hippt::cross(s, edge1);
		v = f * hippt::dot(ray.direction, q);

		if (v < 0.0f || u + v > 1.0f)
			return false;

		// At this stage we can compute t to find out where the intersection point is on the line.
		float t = f * hippt::dot(edge2, q);

		if (t > EPSILON) // ray intersection
		{
			hit_info.inter_point = ray.origin + ray.direction * t;
			hit_info.geometric_normal = hippt::normalize(hippt::cross(edge1, edge2));

			hit_info.t = t;

			hit_info.uv = make_float2(u, v);

			return true;
		}
		else // This means that there is a line intersection but not a ray intersection.
			return false;
	}

    float area() const;

	float3& operator[] (int index);
	const float3& operator[] (int index) const;

	float3 m_a, m_b, m_c;
};

#endif
