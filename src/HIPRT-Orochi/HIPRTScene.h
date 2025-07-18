/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_SCENE_H
#define HIPRT_SCENE_H

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "HIPRT-Orochi/OrochiTexture.h"
#include "Renderer/GPUDataStructures/MaterialPackedSoAGPUData.h"
#include "Renderer/CPUGPUCommonDataStructures/PrecomputedEmissiveTrianglesDataSoAHost.h"
#include "UI/ImGui/ImGuiLogger.h"

#include "hiprt/hiprt.h"
#include "Orochi/Orochi.h"

extern ImGuiLogger g_imgui_logger;

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

	void upload_triangle_indices(const std::vector<int>& triangles_indices)
	{
		int triangle_count = triangles_indices.size() / 3;
		// Allocating and initializing the indices buffer
		m_mesh.triangleCount = triangle_count;
		m_mesh.triangleStride = sizeof(int3);
		OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_mesh.triangleIndices), triangle_count * sizeof(int3)));
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_mesh.triangleIndices), triangles_indices.data(), triangle_count * sizeof(int3), oroMemcpyHostToDevice));
	}

	std::vector<int> download_triangle_indices()
	{
		if (m_mesh.vertices != nullptr)
			return OrochiBuffer<int>::download_data(reinterpret_cast<int*>(m_mesh.triangleIndices), m_mesh.triangleCount * 3);

		return std::vector<int>();
	}

	void upload_vertices_positions(const std::vector<float3>& vertices_positions)
	{
		// Allocating and initializing the vertices positions buiffer
		m_mesh.vertexCount = vertices_positions.size();
		m_mesh.vertexStride = sizeof(float3);
		OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_mesh.vertices), m_mesh.vertexCount * sizeof(float3)));
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_mesh.vertices), vertices_positions.data(), m_mesh.vertexCount * sizeof(float3), oroMemcpyHostToDevice));
	}

	std::vector<float3> download_vertices_positions()
	{
		if (m_mesh.vertices != nullptr)
			return OrochiBuffer<float3>::download_data(reinterpret_cast<float3*>(m_mesh.vertices), m_mesh.vertexCount);

		return std::vector<float3>();
	}

	void log_bvh_building(hiprtBuildFlags build_flags)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Compiling BVH building kernels & building scene BVH...");
	}

	void build_bvh(hiprtBuildFlags build_flags, bool do_compaction, bool disable_spatial_splits_on_OOM, oroStream_t build_stream)
	{
		auto start = std::chrono::high_resolution_clock::now();

		if (m_geometry != nullptr)
		{
			HIPRT_CHECK_ERROR(hiprtDestroyGeometry(m_hiprt_ctx, m_geometry));

			m_geometry = nullptr;
		}

		if (m_mesh.vertexCount == 0 || m_mesh.triangleCount == 0)
			// No BVH to build
			return;

		hiprtBuildOptions build_options;
		hiprtGeometryBuildInput geometry_build_input;
		size_t geometry_temp_size;
		hiprtDevicePtr geometry_temp;

		build_options.buildFlags = build_flags;

		geometry_build_input.type = hiprtPrimitiveTypeTriangleMesh;
		geometry_build_input.primitive.triangleMesh = m_mesh;
		// Geom type 0 here 
		geometry_build_input.geomType = 0;

		log_bvh_building(build_options.buildFlags);
		// Getting the buffer sizes for the construction of the BVH
		HIPRT_CHECK_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(m_hiprt_ctx, geometry_build_input, build_options, geometry_temp_size));

		oroError_t error = oroMalloc(reinterpret_cast<oroDeviceptr*>(&geometry_temp), geometry_temp_size);
		if (error != oroSuccess && error == oroErrorOutOfMemory && disable_spatial_splits_on_OOM)
		{
			if (error == oroErrorOutOfMemory && disable_spatial_splits_on_OOM)
			{
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING, "Out of memory while trying to build the BVH... Retrying without spatial splits. Tracing performance may suffer...");

				build_options.buildFlags |= hiprtBuildFlagBitDisableSpatialSplits;

				HIPRT_CHECK_ERROR(hiprtGetGeometryBuildTemporaryBufferSize(m_hiprt_ctx, geometry_build_input, build_options, geometry_temp_size));
				error = oroMalloc(reinterpret_cast<oroDeviceptr*>(&geometry_temp), geometry_temp_size);

				if (error != oroSuccess)
				{
					g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING, "Error while trying to build the BVH even without spatial splits... Aborting...");

					OROCHI_CHECK_ERROR(error);
				}
			}
		}
		else
			OROCHI_CHECK_ERROR(error);

		HIPRT_CHECK_ERROR(hiprtCreateGeometry(m_hiprt_ctx, geometry_build_input, build_options, m_geometry));
		HIPRT_CHECK_ERROR(hiprtBuildGeometry(m_hiprt_ctx, hiprtBuildOperationBuild, geometry_build_input, build_options, geometry_temp, build_stream, m_geometry));
		OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(geometry_temp)));

		if (do_compaction)
			HIPRT_CHECK_ERROR(hiprtCompactGeometry(m_hiprt_ctx, 0, m_geometry, m_geometry));

		auto stop = std::chrono::high_resolution_clock::now();
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "BVH built in %ldms", std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
	}

	hiprtContext m_hiprt_ctx = nullptr;
	hiprtTriangleMeshPrimitive m_mesh = { nullptr };
	// One geometry for the whole scene for now
	hiprtGeometry m_geometry = nullptr;
};

struct HIPRTScene
{
	void print_statistics(std::ostream& stream)
	{
		stream << "Scene statistics: " << std::endl;
		stream << "\t" << geometry.m_mesh.vertexCount << " vertices" << std::endl;
		stream << "\t" << geometry.m_mesh.triangleCount << " triangles" << std::endl;
		stream << "\t" << emissive_triangles_indices.size() << " emissive triangles" << std::endl;
		stream << "\t" << materials_buffer.m_element_count << " materials" << std::endl;
		stream << "\t" << orochi_materials_textures.size() << " textures" << std::endl;
	}

	HIPRTGeometry geometry;

	OrochiBuffer<float> triangle_areas;
	OrochiBuffer<unsigned char> has_vertex_normals;
	OrochiBuffer<float3> vertex_normals;
	OrochiBuffer<int> material_indices;
	DevicePackedTexturedMaterialSoAGPUData materials_buffer;

	// This vector contains true for a material that has a fully opaque base color texture.
	// Otherwise, the texture has some alpha transparency in it
	//
	// This vector isn't used on the GPU, it's only used by the CPU to basically remember which 
	// materials had textures with some alpha in it
	std::vector<bool> material_has_opaque_base_color_texture;
	OrochiBuffer<unsigned char> material_opaque;

	int emissive_triangles_count = 0;
	OrochiBuffer<int> emissive_triangles_indices;
	OrochiBuffer<float> emissive_power_alias_table_probas;
	OrochiBuffer<int> emissive_power_alias_table_alias;
	// This is a remnant of some tests and it was actually not worth it
	PrecomputedEmissiveTrianglesDataSoAHost<OrochiBuffer> precomputed_emissive_triangles_data;

	// Vector to keep the textures data alive otherwise the OrochiTexture objects would
	// be destroyed which means that the underlying textures would be destroyed
	std::vector<OrochiTexture> orochi_materials_textures;
	OrochiBuffer<oroTextureObject_t> gpu_materials_textures;
	OrochiBuffer<float2> texcoords_buffer;
};

#endif
