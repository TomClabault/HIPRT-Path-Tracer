/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_FLOAT4X4_H
#define HOST_DEVICE_COMMON_FLOAT4X4_H

struct float4x4
{
	float m[4][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
};

HIPRT_DEVICE static float3_t matrix_X_point(const float4x4& m, const float3_t& p)
{
	float x = p.x;
	float y = p.y;
	float z = p.z;

	// Assuming w = 1.0f for the point p
	float xt = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
	float yt = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
	float zt = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
	float wt = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];

	float inv_w = 1.0f;
	if (abs(wt) > 1.0e-10f)
		inv_w = 1.0f / wt;

	return make_float3(xt * inv_w, yt * inv_w, zt * inv_w);
}

HIPRT_DEVICE static float3_t matrix_X_vec(const float4x4& m, const float3_t& u)
{
	float x = u.x;
	float y = u.y;
	float z = u.z;

	// Assuming w = 0.0f for the vector u
	float xt = m.m[0][0] * x + m.m[1][0] * y + m.m[2][0] * z;
	float yt = m.m[0][1] * x + m.m[1][1] * y + m.m[2][1] * z;
	float zt = m.m[0][2] * x + m.m[1][2] * y + m.m[2][2] * z;
	float wt = m.m[0][3] * x + m.m[1][3] * y + m.m[2][3] * z;

	float inv_w = 1.0f;
	if (abs(wt) > 1.0e-10f)
		inv_w = 1.0f / wt;

	return make_float3(xt * inv_w, yt * inv_w, zt * inv_w);
}

#endif // #ifndef HOST_DEVICE_COMMON_FLOAT4X4_H
