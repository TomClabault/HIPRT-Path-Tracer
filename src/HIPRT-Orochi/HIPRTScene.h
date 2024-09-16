/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_SCENE_H
#define HIPRT_SCENE_H

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "HostDeviceCommon/Material.h"

#include "hiprt/hiprt.h"
#include "Orochi/Orochi.h"

struct HIPRTGeometry
{
	HIPRTGeometry() : m_hiprt_ctx(nullptr) {}
	HIPRTGeometry(hiprtContext ctx) : m_hiprt_ctx(ctx) {}

	~HIPRTGeometry()
	{
		if (m_mesh.triangleIndices)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_mesh.triangleIndices)));

		if (m_mesh.vertices)
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_mesh.vertices)));

		if (m_geometry)
			HIPRT_CHECK_ERROR(hiprtDestroyGeometry(m_hiprt_ctx, m_geometry));
	}

	void upload_indices(const std::vector<int>& triangles_indices)
	{
		int triangle_count = triangles_indices.size() / 3;
		// Allocating and initializing the indices buffer
		m_mesh.triangleCount = triangle_count;
		m_mesh.triangleStride = sizeof(int3);
		OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_mesh.triangleIndices), triangle_count * sizeof(int3)));
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_mesh.triangleIndices), triangles_indices.data(), triangle_count * sizeof(int3), oroMemcpyHostToDevice));
	}

	void upload_vertices(const std::vector<float3>& vertices_positions)
	{
		// Allocating and initializing the vertices positions buiffer
		m_mesh.vertexCount = vertices_positions.size();
		m_mesh.vertexStride = sizeof(float3);
		OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_mesh.vertices), m_mesh.vertexCount * sizeof(float3)));
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_mesh.vertices), vertices_positions.data(), m_mesh.vertexCount * sizeof(float3), oroMemcpyHostToDevice));
	}

	void log_bvh_building(hiprtBuildFlags build_flags)
	{
		std::cout << "Compiling BVH building kernels & building scene ";
		if (build_flags == 0)
			// This is hiprtBuildFlagBitPreferFastBuild
			std::cout << "LBVH";
		else if (build_flags & hiprtBuildFlagBitPreferBalancedBuild)
			std::cout << "PLOC BVH";
		else if (build_flags & hiprtBuildFlagBitPreferHighQualityBuild)
			std::cout << "SBVH";
	}

	void build_bvh()
	{
		auto start = std::chrono::high_resolution_clock::now();

		hiprtBuildOptions build_options;
		hiprtGeometryBuildInput geometry_build_input;
		size_t geometry_temp_size;
		hiprtDevicePtr geometry_temp;

		build_options.buildFlags = hiprtBuildFlagBitPreferHighQualityBuild;
		geometry_build_input.type = hiprtPrimitiveTypeTriangleMesh;
		geometry_build_input.primitive.triangleMesh = m_mesh;

		log_bvh_building(build_options.buildFlags);
		// Getting the buffer sizes for the construction of the BVH
		HIPRT_CHECK_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(m_hiprt_ctx, geometry_build_input, build_options, geometry_temp_size));
		OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&geometry_temp), geometry_temp_size));

		HIPRT_CHECK_ERROR(hiprtCreateGeometry(m_hiprt_ctx, geometry_build_input, build_options, m_geometry));
		HIPRT_CHECK_ERROR(hiprtBuildGeometry(m_hiprt_ctx, hiprtBuildOperationBuild, geometry_build_input, build_options, geometry_temp, /* stream */ 0, m_geometry));

		OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(geometry_temp)));

		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "BVH built in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
	}

	hiprtContext m_hiprt_ctx = nullptr;
	hiprtTriangleMeshPrimitive m_mesh = { nullptr };
	hiprtGeometry m_geometry = nullptr;
};

struct HIPRTScene
{
	HIPRTGeometry geometry;

	OrochiBuffer<bool> has_vertex_normals;
	OrochiBuffer<float3> vertex_normals;
	OrochiBuffer<int> material_indices;
	OrochiBuffer<RendererMaterial> materials_buffer;

	int emissive_triangles_count = 0;
	OrochiBuffer<int> emissive_triangles_indices;

	OrochiBuffer<oroTextureObject_t> materials_textures;
	OrochiBuffer<int2> textures_dims;
	OrochiBuffer<float2> texcoords_buffer;
};

#endif
