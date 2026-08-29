/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_VMF_H
#define DEVICE_INCLUDES_VMF_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Maths/VecTypes.h"
#include "HostDeviceCommon/Xorshift.h"

/**
 * Reference: [Numerically Stable Implementation of the von Mises–Fisher Distribution on S2, Tokuyoshi, 2024]
 */
struct VMF
{
	static constexpr float INVALID_SHARPNESS = -42.0f;

	float3_t axis	= make_float3(0.0f, 0.0f, 0.0f);
	float sharpness = INVALID_SHARPNESS;

	HIPRT_DEVICE float density_evaluation(float3_t direction) const
	{
		float3_t d = direction - axis;

		if (sharpness < 1.0e-4f)
			return hippt::M_INV_TWO_PI;

		return hippt::max(1.0e-30f, hippt::intrin_expf(-0.5f * sharpness * hippt::dot(d, d)) * hippt::x_over_expm1(-2.0f * sharpness) / (4.0f * hippt::M_Pi));
	}

	HIPRT_DEVICE float3_t sample(Xorshift32Generator& random_number_generator) const
	{
		float rand_1 = random_number_generator();
		float rand_2 = random_number_generator();

		constexpr float THRESHOLD = hippt::FLOAT_EPSILON / 4.0f;
		float phi				  = 2.0f * hippt::M_Pi * rand_1;
		float r					  = sharpness > THRESHOLD ? hippt::intrin_log1pf(rand_2 * hippt::intrin_expm1f(-2.0f * sharpness)) / sharpness : -2.0f * rand_2;

		float cos_theta = 1.0f + r;
		float sin_theta = hippt::sqrt(-hippt::fma(r, r, 2.0f * r));
		float3_t dir	= { hippt::intrin_cosf(phi) * sin_theta, hippt::intrin_sinf(phi) * sin_theta, cos_theta };

		return local_to_world_frame(axis, dir);
	}
};

#endif // #ifndef DEVICE_INCLUDES_VMF_H
