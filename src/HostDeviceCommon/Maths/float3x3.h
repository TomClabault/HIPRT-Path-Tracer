/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_FLOAT3X3_H
#define HOST_DEVICE_COMMON_FLOAT3X3_H

#include "HostDeviceCommon/Maths/VecTypes.h"

struct float3x3
{
	HIPRT_DEVICE float3x3() {}
	HIPRT_DEVICE float3x3(
		float m00, float m01, float m02,
		float m10, float m11, float m12,
		float m20, float m21, float m22)
	{
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
	}

	float3x3(const float3& col0, const float3& col1, const float3& col2)
	{
		m[0][0] = col0.x; m[0][1] = col1.x; m[0][2] = col2.x;
		m[1][0] = col0.y; m[1][1] = col1.y; m[1][2] = col2.y;
		m[2][0] = col0.z; m[2][1] = col1.z; m[2][2] = col2.z;
	}

	/**
	 * Construct from 3 columns
	 */
	HIPRT_DEVICE static float3x3 from_cols(float3 col0, float3 col1, float3 col2)
	{
		float3x3 result;

		result.m[0][0] = col0.x; result.m[0][1] = col1.x; result.m[0][2] = col2.x;
		result.m[1][0] = col0.y; result.m[1][1] = col1.y; result.m[1][2] = col2.y;
		result.m[2][0] = col0.z; result.m[2][1] = col1.z; result.m[2][2] = col2.z;

		return result;
	}

	HIPRT_DEVICE static float3x3 from_rows(float3 row0, float3 row1, float3 row2)
	{
		float3x3 result;

		result.m[0][0] = row0.x; result.m[0][1] = row0.y; result.m[0][2] = row0.z;
		result.m[1][0] = row1.x; result.m[1][1] = row1.y; result.m[1][2] = row1.z;
		result.m[2][0] = row2.x; result.m[2][1] = row2.y; result.m[2][2] = row2.z;

		return result;
	}

	float m[3][3] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };
};

HIPRT_DEVICE static float3 matrix_X_vec(const float3x3& m, const float3& u)
{
	float x = u.x;
	float y = u.y;
	float z = u.z;

	// Assuming w = 0.0f for the vector u
	float xt = m.m[0][0] * x + m.m[1][0] * y + m.m[2][0] * z;
	float yt = m.m[0][1] * x + m.m[1][1] * y + m.m[2][1] * z;
	float zt = m.m[0][2] * x + m.m[1][2] * y + m.m[2][2] * z;

	return make_float3(xt, yt, zt);
}

HIPRT_DEVICE static float3x3 operator-(const float3x3& a)
{
	float3x3 result;

	result.m[0][0] = -a.m[0][0];
	result.m[0][1] = -a.m[0][1];
	result.m[0][2] = -a.m[0][2];
	result.m[1][0] = -a.m[1][0];
	result.m[1][1] = -a.m[1][1];
	result.m[1][2] = -a.m[1][2];
	result.m[2][0] = -a.m[2][0];
	result.m[2][1] = -a.m[2][1];
	result.m[2][2] = -a.m[2][2];

	return result;
}

HIPRT_DEVICE static float3x3 operator*(const float3x3& a, const float3x3& b)
{
	float3x3 result;

	// Row 0
	result.m[0][0] = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[1][0] + a.m[0][2] * b.m[2][0];
	result.m[0][1] = a.m[0][0] * b.m[0][1] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[2][1];
	result.m[0][2] = a.m[0][0] * b.m[0][2] + a.m[0][1] * b.m[1][2] + a.m[0][2] * b.m[2][2];

	// Row 1
	result.m[1][0] = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[1][0] + a.m[1][2] * b.m[2][0];
	result.m[1][1] = a.m[1][0] * b.m[0][1] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[2][1];
	result.m[1][2] = a.m[1][0] * b.m[0][2] + a.m[1][1] * b.m[1][2] + a.m[1][2] * b.m[2][2];

	// Row 2
	result.m[2][0] = a.m[2][0] * b.m[0][0] + a.m[2][1] * b.m[1][0] + a.m[2][2] * b.m[2][0];
	result.m[2][1] = a.m[2][0] * b.m[0][1] + a.m[2][1] * b.m[1][1] + a.m[2][2] * b.m[2][1];
	result.m[2][2] = a.m[2][0] * b.m[0][2] + a.m[2][1] * b.m[1][2] + a.m[2][2] * b.m[2][2];

	return result;
}

/**
 * Multiplies the matrix on the left of the column vector v
 * The rows of the matrix should be the basis vectors (if that matrix is a change of basis)
 */
HIPRT_DEVICE static float3 operator*(const float3x3& a, const float3& v)
{
	float3 result;

	result.x = a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z;
	result.y = a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z;
	result.z = a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z;

	return result;
}

/**
 * Multiplies the matrix on the right of the row vector v
 * The columns of the matrix should be the basis vectors (if that matrix is a change of basis)
 * 
 * Equivalent to transpose(a) * v
 */
HIPRT_DEVICE static float3 operator*(const float3& v, const float3x3& a)
{
	float3 result;

	result.x = a.m[0][0] * v.x + a.m[1][0] * v.y + a.m[2][0] * v.z;
	result.y = a.m[0][1] * v.x + a.m[1][1] * v.y + a.m[2][1] * v.z;
	result.z = a.m[0][2] * v.x + a.m[1][2] * v.y + a.m[2][2] * v.z;

	return result;
}

HIPRT_DEVICE static float3x3 operator*(const float3x3& a, const float x)
{
	float3x3 result;

	result.m[0][0] = a.m[0][0] * x;
	result.m[0][1] = a.m[0][1] * x;
	result.m[0][2] = a.m[0][2] * x;
	result.m[1][0] = a.m[1][0] * x;
	result.m[1][1] = a.m[1][1] * x;
	result.m[1][2] = a.m[1][2] * x;
	result.m[2][0] = a.m[2][0] * x;
	result.m[2][1] = a.m[2][1] * x;
	result.m[2][2] = a.m[2][2] * x;

	return result;
}

HIPRT_DEVICE static float3x3 operator*(const float x, const float3x3& a)
{
	return a * x;
}

HIPRT_DEVICE static float determinant(const float3x3& m)
{
	return m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
		m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
		m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
}

HIPRT_DEVICE static float3x3 transpose(const float3x3& m)
{
	float3x3 result;

	result.m[0][0] = m.m[0][0];
	result.m[0][1] = m.m[1][0];
	result.m[0][2] = m.m[2][0];
	result.m[1][0] = m.m[0][1];
	result.m[1][1] = m.m[1][1];
	result.m[1][2] = m.m[2][1];
	result.m[2][0] = m.m[0][2];
	result.m[2][1] = m.m[1][2];
	result.m[2][2] = m.m[2][2];

	return result;
}

HIPRT_DEVICE static float3x3 inverse(const float3x3& m)
{
	float det = determinant(m);
	float inv_det = 1.0f / det;

	float3x3 result;

	// 1.0 / det * adjoint(m)
	result.m[0][0] = inv_det * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]);
	result.m[0][1] = inv_det * (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]);
	result.m[0][2] = inv_det * (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]);
	result.m[1][0] = inv_det * (m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]);
	result.m[1][1] = inv_det * (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]);
	result.m[1][2] = inv_det * (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]);
	result.m[2][0] = inv_det * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
	result.m[2][1] = inv_det * (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]);
	result.m[2][2] = inv_det * (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]);

	return result;
}

#endif
