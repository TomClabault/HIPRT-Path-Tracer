/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDF_MICROFACET_COMMON_H
#define DEVICE_INCLUDES_BSDF_MICROFACET_COMMON_H

#include "HostDeviceCommon/Maths/Math.h"

// Clamping value for dot products when evaluating the GGX distribution
// This helps with fireflies due to numerical imprecisions
//
// 1.0e-4f seems indistinguishable from 1.0e-8f (which is closer to
// "ground truth" since we're not clamping as hard) except that 1.0e-8f
// has a bunch of fireflies / is not very stable at all.
//
// So even though 1.0e-4f may seem a bit harsh, it's actually fine
#define GGX_DOT_PRODUCTS_CLAMP 1.0e-4f

/**
 * Evaluates the GGX anisotropic normal distribution function
 */
HIPRT_DEVICE static float GGX_anisotropic(float alpha_x, float alpha_y, const float3& local_microfacet_normal)
{
	float denom = (local_microfacet_normal.x * local_microfacet_normal.x) / (alpha_x * alpha_x) +
				  (local_microfacet_normal.y * local_microfacet_normal.y) / (alpha_y * alpha_y) + (local_microfacet_normal.z * local_microfacet_normal.z);

	if (denom * local_microfacet_normal.z <= 0.0f)
		return 0.0f;

	return 1.0f / (hippt::M_Pi * alpha_x * alpha_y * denom * denom);
}

/**
 * Evaluates the visible normal distribution function with GGX as
 * the normal disitrbution function
 *
 * Reference: [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * Equation 3
 */
HIPRT_DEVICE static float GGX_anisotropic_vndf(float D, float G1V, const float3& local_view_direction, const float3& local_microfacet_normal)
{
	float HoV = hippt::max(GGX_DOT_PRODUCTS_CLAMP, hippt::dot(local_view_direction, local_microfacet_normal));
	return G1V * D * hippt::abs(HoV) / hippt::abs(local_view_direction.z);
}

/**
 * Lambda function for the denominator of the G1 Smith masking/shadowing functions
 */
HIPRT_DEVICE static float G1_Smith_lambda(float alpha_x, float alpha_y, const float3& local_direction)
{
	float ax = local_direction.x * alpha_x;
	float ay = local_direction.y * alpha_y;

	return (-1.0f + hippt::sqrt(1.0f + (ax * ax + ay * ay) / (local_direction.z * local_direction.z))) * 0.5f;
}

/**
 * G1 Smith masking/shadowing (depending on whether local_direction is wo or wi) function
 *
 * Reference: [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs, Heitz, 2014]
 * Equation 43
 */
HIPRT_DEVICE static float G1_Smith(float alpha_x, float alpha_y, const float3& local_direction)
{
	float lambda = G1_Smith_lambda(alpha_x, alpha_y, local_direction);

	return 1.0f / (1.0f + lambda);
}

#endif
