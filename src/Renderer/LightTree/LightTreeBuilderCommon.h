/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_BUILDER_COMMON_H
#define RENDERER_LIGHT_TREE_BUILDER_COMMON_H

#include "HostDeviceCommon/Material/MaterialCPU.h"

#include <vector>

struct LightTreeBuilderTrianglesData
{
	LightTreeBuilderTrianglesData(const std::vector<int>& emissive_triangles_primitive_indices,
								  const std::vector<int>& triangle_vertex_indices,
								  const std::vector<float3_t>& vertices_positions)
		: emissive_triangles_primitive_indices(emissive_triangles_primitive_indices), triangle_vertex_indices(triangle_vertex_indices),
		  vertices_positions(vertices_positions)
	{
	}

	const std::vector<int>& emissive_triangles_primitive_indices;
	const std::vector<int>& triangle_vertex_indices;
	const std::vector<float3_t>& vertices_positions;
};

#endif // #ifndef RENDERER_LIGHT_TREE_BUILDER_COMMON_H
