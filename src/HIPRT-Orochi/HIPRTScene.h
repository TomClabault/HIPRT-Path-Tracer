/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_SCENE_H
#define HIPRT_SCENE_H

#include "hiprt/hiprt.h"
#include "Orochi/Orochi.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

struct HIPRTScene
{
	HIPRTScene() : hiprt_ctx(nullptr) 
	{
		mesh.vertices = nullptr;
		mesh.triangleIndices = nullptr;
		geometry = nullptr;

		normals_present = nullptr;
		vertex_normals = nullptr;

		material_indices = nullptr;
		materials_buffer = nullptr;

		emissive_triangles_count = 0;
		emissive_triangles_indices = nullptr;
	}

	HIPRTScene(hiprtContext ctx) : HIPRTScene()
	{
		hiprt_ctx = ctx;
	}

	~HIPRTScene()
	{
		if (mesh.triangleIndices)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(mesh.triangleIndices)));

		if (mesh.vertices)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(mesh.vertices)));

		if (geometry)
			HIPRT_CHECK_ERROR(hiprtDestroyGeometry(hiprt_ctx, geometry));

		if (material_indices)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(material_indices)));

		if (materials_buffer)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(materials_buffer)));

		if (emissive_triangles_indices)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(emissive_triangles_indices)));
	}

	hiprtContext hiprt_ctx = nullptr;
	hiprtTriangleMeshPrimitive mesh;
	hiprtGeometry geometry = nullptr;

	hiprtDevicePtr normals_present;
	hiprtDevicePtr vertex_normals;

	hiprtDevicePtr material_indices;
	hiprtDevicePtr materials_buffer;

	int emissive_triangles_count;
	hiprtDevicePtr emissive_triangles_indices;
};

#endif
