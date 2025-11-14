/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_FLOAT2X2_H
#define HOST_DEVICE_COMMON_FLOAT2X2_H

struct float2x2
{
	HIPRT_DEVICE float2x2() {}
	HIPRT_DEVICE float2x2(float m00, float m01, float m10, float m11)
	{
		m[0][0] = m00; m[0][1] = m01;
		m[1][0] = m10; m[1][1] = m11;
	}

	/**
	 * Construct from 2 columns
	 */
	HIPRT_DEVICE float2x2(float2 col0, float2 col1)
	{
		m[0][0] = col0.x; m[0][1] = col1.x;
		m[1][0] = col0.y; m[1][1] = col1.y;
	}

	float m[2][2];
};

HIPRT_DEVICE static float2x2 operator+(const float2x2& a, const float2x2& b)
{
	float2x2 result;

	result.m[0][0] = a.m[0][0] + b.m[0][0];
	result.m[0][1] = a.m[0][1] + b.m[0][1];
	result.m[1][0] = a.m[1][0] + b.m[1][0];
	result.m[1][1] = a.m[1][1] + b.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator-(const float2x2& a, const float2x2& b)
{
	float2x2 result;

	result.m[0][0] = a.m[0][0] - b.m[0][0];
	result.m[0][1] = a.m[0][1] - b.m[0][1];
	result.m[1][0] = a.m[1][0] - b.m[1][0];
	result.m[1][1] = a.m[1][1] - b.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator*(const float2x2& a, const float2x2& b)
{
	float2x2 result;

	result.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0];
	result.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1];
	result.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0];
	result.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator*(const float k, const float2x2& a)
{
	float2x2 result;

	result.m[0][0] = k * a.m[0][0];
	result.m[0][1] = k * a.m[0][1];
	result.m[1][0] = k * a.m[1][0];
	result.m[1][1] = k * a.m[1][1];

	return result;
}

HIPRT_DEVICE static float2x2 operator*(const float2x2& a, const float k)
{
	return k * a;
}

HIPRT_DEVICE static float2 operator*(const float2x2& a, const float2& v)
{
	float2 result;

	result.x = a.m[0][0] * v.x + a.m[0][1] * v.y;
	result.y = a.m[1][0] * v.x + a.m[1][1] * v.y;

	return result;
}

HIPRT_DEVICE static float2x2 operator/(const float2x2& a, const float k)
{
	float inv_k = 1.0f / k;
	float2x2 result;

	result.m[0][0] = a.m[0][0] * inv_k;
	result.m[0][1] = a.m[0][1] * inv_k;
	result.m[1][0] = a.m[1][0] * inv_k;
	result.m[1][1] = a.m[1][1] * inv_k;

	return result;
}

HIPRT_DEVICE static float2x2 transpose(const float2x2& m)
{
	float2x2 result;

	result.m[0][0] = m.m[0][0];
	result.m[0][1] = m.m[1][0];
	result.m[1][0] = m.m[0][1];
	result.m[1][1] = m.m[1][1];

	return result;
}

HIPRT_DEVICE static float determinant(const float2x2& m)
{
	return m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0];
}

HIPRT_DEVICE static float2x2 outer_product(const float2& a, const float2& b)
{
	float2x2 out;

	// row 0
	out.m[0][0] = a.x * b.x;
	out.m[0][1] = a.x * b.y;
	// row 1
	out.m[1][0] = a.y * b.x;
	out.m[1][1] = a.y * b.y;

	return out;
}

#endif
