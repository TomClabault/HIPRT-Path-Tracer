/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SURFACE_NORMAL_FACE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_SURFACE_NORMAL_FACE_H

#include "HostDeviceCommon/Maths/VecTypes.h"

enum SurfaceNormalFace : unsigned char
{
	SurfaceNormalFace_PositiveX = 0,
	SurfaceNormalFace_NegativeX = 1,
	SurfaceNormalFace_PositiveY = 2,
	SurfaceNormalFace_NegativeY = 3,
	SurfaceNormalFace_PositiveZ = 4,
	SurfaceNormalFace_NegativeZ = 5,

	SurfaceNormalFace_Count = 6
};

HIPRT_HOST_DEVICE inline unsigned int illumination_aware_kd_tree_classify_surface_normal_face(const float3_t& shading_normal)
{
	float absolute_x = hippt::abs(shading_normal.x);
	float absolute_y = hippt::abs(shading_normal.y);
	float absolute_z = hippt::abs(shading_normal.z);

	if (absolute_x >= absolute_y && absolute_x >= absolute_z)
	{
		return shading_normal.x >= 0.0f ? SurfaceNormalFace_PositiveX : SurfaceNormalFace_NegativeX;
	}

	if (absolute_y >= absolute_z)
	{
		return shading_normal.y >= 0.0f ? SurfaceNormalFace_PositiveY : SurfaceNormalFace_NegativeY;
	}

	return shading_normal.z >= 0.0f ? SurfaceNormalFace_PositiveZ : SurfaceNormalFace_NegativeZ;
}

#endif
