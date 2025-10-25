/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_H

#include "Renderer/LightTree/LightTreeATSBuilder.h"
#include "Renderer/LightTree/LightTreeBuilderCommon.h"
#include "Renderer/LightTree/LightTreeSGBuilderDeviceData.h"
#include "Renderer/LightTree/LightTreeSGNode.h"

class LightTreeSGBuilder
{
public:
	void build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices, const std::vector<int>& triangle_indices, const std::vector<float3>& vertices_positions, const std::vector<int>& material_indices, const std::vector<CPUMaterial>& materials);

	void compute_node_spherical_gaussian(unsigned int node_index, const LightTreeBuilderTrianglesData& triangle_data);

private:
	LightTreeATSBuilder m_light_tree_ats_builder;

	std::vector<LightTreeSGNode> m_nodes;
};

#endif
