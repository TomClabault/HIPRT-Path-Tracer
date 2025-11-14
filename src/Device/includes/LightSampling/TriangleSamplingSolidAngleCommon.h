#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_SOLID_ANGLE_COMMON_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_SOLID_ANGLE_COMMON_H

#include "HostDeviceCommon/Maths/Math.h"

// #define USE_BIASED_PROJECTED_SOLID_ANGLE_SAMPLING

/*! A piecewise polynomial approximation to positive_atan(y). The maximal
	absolute error is 1.16e-05f. At least on Turing GPUs, it is faster but also
	significantly less accurate. The proper atan has at most 2 ulps of error
	there.*/
HIPRT_DEVICE float fast_positive_atan(float y)
{
	float rx;
	float ry;
	float rz;
	rx = (hippt::abs(y) > 1.0f) ? (1.0f / hippt::abs(y)) : hippt::abs(y);
	ry = rx * rx;
	rz = hippt::fma(ry, 0.02083509974181652f, -0.08513300120830536);
	rz = hippt::fma(ry, rz, 0.18014100193977356f);
	rz = hippt::fma(ry, rz, -0.3302994966506958f);
	ry = hippt::fma(ry, rz, 0.9998660087585449f);
	rz = hippt::fma(-2.0f * ry, rx, hippt::M_PI_TWO);
	rz = (hippt::abs(y) > 1.0f) ? rz : 0.0f;
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

/**
 * Intersects the sampled triangle with the ray given by the sampled direction to find the point on the triangle
 * that is sampled from the direction.
 *
 * This is needed because this renderer works from sampled points on triangles, not directions.
 */
HIPRT_DEVICE float3 map_direction_to_triangle_point(float3 sampled_solid_angle_direction, float3 vertex_A, float3 triangle_normal, float3 shading_point,
	float pdf_solid_angle, float& out_pdf_area)
{
	float3 v0_rel = vertex_A - shading_point;
	float denom = hippt::dot(sampled_solid_angle_direction, triangle_normal);

	if (hippt::abs(denom) < 1e-8f)
	{
		// Nearly parallel
		out_pdf_area = 0.0f;

		return vertex_A;
	}

	float t = hippt::dot(v0_rel, triangle_normal) / denom;

	float3 point = shading_point + sampled_solid_angle_direction * t;

	// Conversion of the PDF to area measure
	float cos_theta = compute_cosine_term_at_light_source(triangle_normal, -sampled_solid_angle_direction);
	out_pdf_area = pdf_solid_angle * (cos_theta / (t * t));

	return point;
}

#endif
