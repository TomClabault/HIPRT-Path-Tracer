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
	HIPRTScene() : mesh({ nullptr }) {}

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

		if (material_textures)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(material_textures)));

		if (texture_is_srgb)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(texture_is_srgb)));

		if (texcoords_buffer)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(texcoords_buffer)));
	}

	hiprtContext hiprt_ctx = nullptr;
	hiprtTriangleMeshPrimitive mesh;
	hiprtGeometry geometry = nullptr;

	hiprtDevicePtr has_vertex_normals = nullptr;
	hiprtDevicePtr vertex_normals = nullptr;

	hiprtDevicePtr material_indices = nullptr;
	hiprtDevicePtr materials_buffer = nullptr;

	int emissive_triangles_count = 0;
	hiprtDevicePtr emissive_triangles_indices = nullptr;

	hiprtDevicePtr material_textures = nullptr;
	hiprtDevicePtr texture_is_srgb = nullptr;
	hiprtDevicePtr texcoords_buffer = nullptr;
};

#endif
