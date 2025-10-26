/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_SAMPLING_DATA_STRUCTURE_H
#define RENDERER_LIGHT_TREE_SG_SAMPLING_DATA_STRUCTURE_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/LightTree/LightTreeSGBuilder.h"
#include "Scene/SceneParser.h"

class GPURenderer;

class LightTreeSGSamplingDataStructure
{
public:
	LightTreeSGSamplingDataStructure() : m_renderer(nullptr) {}
	LightTreeSGSamplingDataStructure(GPURenderer* renderer) : m_renderer(renderer) {}

	void compute_from_scene(const Scene& scene);
	void compute(
		const std::vector<int>& emissive_triangle_indices,
		const std::vector<float3>& vertices_positions,
		const std::vector<int>& triangles_vertex_indices,
		const std::vector<int>& material_indices,
		const std::vector<CPUMaterial>& materials);

	void recompute_if_needed(bool skip_if_already_computed = false);
	void free();

	bool is_needed(unsigned int emissive_count);

private:
	LightTreeSGBuilder m_light_tree_builder_sg;
	LightTreeSGBuilderDeviceData<OrochiBuffer> m_light_tree_sg_device_data;

	GPURenderer* m_renderer = nullptr;
};

#endif
