/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_HIT_INFO_H
#define HOST_DEVICE_COMMON_HIT_INFO_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Maths/Math.h"

struct HitInfo
{
	float3_t inter_point	= { 0, 0, 0 };
	float3_t shading_normal = { 0, 0, 0 };
	// This geometric_normal member is flipped to be facing the ray direction if the geometry is backfacing
	float3_t geometric_normal = { 0, 0, 0 };
	// TODO is texcoords useful? This may actually be returned by the intersection function and used only for reading textures but then we don't need it anymore
	// when evaluating the BSDF and computing the main path tracing stuff so let's save some registers
	float2_t texcoords = { 0, 0 };

	// Distance along ray
	float t = -1.0f;

	int primitive_index = -1;

	// If true, the geometry that was hit had normals facing the opposite direction our ray came from. This means that the geometric normal and shading normal
	// members of this structure have been flipped (the convention is that in HitInfo, those are always facing the ray such that dot(ray.direction,
	// geometric_normal) < 0.0f
	bool geometry_backfacing = false;

	HIPRT_DEVICE float3_t original_geometric_normal() const
	{
		return geometry_backfacing ? -geometric_normal : geometric_normal;
	}
};

#endif
