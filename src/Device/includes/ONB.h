/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ONB_H
#define DEVICE_INCLUDES_ONB_H

#include "HostDeviceCommon/Maths/Math.h"

/*
 * This uses the technique from "Improved accuracy when building an orthonormal basis" by Nelson Max,
 * https://jcgt.org/published/0006/01/02.
 *
 * Taken from https://github.com/nvpro-samples/nvpro_core/blob/master/nvvkhl/shaders/func.h
 * and optimised a little bit by @tigrazone
 */
HIPRT_DEVICE static void build_ONB(const float3_t& N, float3_t& T, float3_t& B)
{
	if (N.z < -0.99998796f) // Handle the singularity
	{
		T = make_float3(0.0f, -1.0f, 0.0f);
		B = make_float3(-1.0f, 0.0f, 0.0f);
		return;
	}

	float nxa = -N.x / (1.0f + N.z);
	T		  = make_float3(1.0f + N.x * nxa, nxa * N.y, -N.x);
	B		  = make_float3(T.y, 1.0f - N.y * N.y / (1.0f + N.z), -N.y);
}

HIPRT_DEVICE static float3_t rotate_vector(float3_t vector, float3_t rotate_around, float theta)
{
	return vector * hippt::intrin_cosf(theta) + hippt::cross(rotate_around, vector) * hippt::intrin_sinf(theta) +
		   rotate_around * hippt::dot(rotate_around, vector) * (1.0f - hippt::intrin_cosf(theta));
}

/*
 * Rotation of the basis around the normal by 'basis_rotation' radians
 */
HIPRT_DEVICE static void build_rotated_ONB(const float3_t& N, float3_t& T, float3_t& B, float basis_rotation)
{
	float3_t up = hippt::abs(N.z) < 0.9999999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
	T			= hippt::normalize(hippt::cross(up, N));

	// Rodrigues' rotation
	T = rotate_vector(T, N, basis_rotation);
	B = hippt::cross(N, T);
}

/**
 * Build an ONB with the given 'N' axis as the Z axis (up) and also such that
 * vec lies perfectly in the X/Z plane
 */
HIPRT_DEVICE static void build_ONB_XZ_plane(const float3_t& N, float3_t& T, float3_t& B, const float3_t& vec_xz)
{
	if (hippt::abs(hippt::dot(vec_xz, N)) > 0.99998796f)
		// TODO this test looks wrong, need to check
		T = N.x > 0.99998796f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
	else
		T = hippt::normalize(vec_xz - hippt::dot(vec_xz, N) * N);
	B = hippt::cross(N, T);
}

/*
 * Transforms V from its local space to the space around the normal
 */
HIPRT_DEVICE static float3_t local_to_world_frame(const float3_t& N, const float3_t& V)
{
	float3_t T, B;
	build_ONB(N, T, B);

	return hippt::normalize(V.x * T + V.y * B + V.z * N);
}

HIPRT_DEVICE static float3_t local_to_world_frame(const float3_t& T, const float3_t& B, const float3_t& N, const float3_t& V)
{
	return hippt::normalize(V.x * T + V.y * B + V.z * N);
}

HIPRT_DEVICE static float3_t local_to_world_frame_non_normalized(const float3_t& T, const float3_t& B, const float3_t& N, const float3_t& V)
{
	return V.x * T + V.y * B + V.z * N;
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
HIPRT_DEVICE static float3_t world_to_local_frame(const float3_t& N, const float3_t& V)
{
	float3_t T, B;
	build_ONB(N, T, B);

	return hippt::normalize(make_float3(hippt::dot(V, T), hippt::dot(V, B), hippt::dot(V, N)));
}

HIPRT_DEVICE static float3_t world_to_local_frame(const float3_t& T, const float3_t& B, const float3_t& N, const float3_t& V)
{
	return hippt::normalize(make_float3(hippt::dot(V, T), hippt::dot(V, B), hippt::dot(V, N)));
}

HIPRT_DEVICE static float3_t world_to_local_frame_non_normalized(const float3_t& T, const float3_t& B, const float3_t& N, const float3_t& V)
{
	return make_float3(hippt::dot(V, T), hippt::dot(V, B), hippt::dot(V, N));
}

#endif // #ifndef DEVICE_INCLUDES_ONB_H
