/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SAMPLING_DATA_STRUCTURE_H
#define RENDERER_LIGHT_TREE_SAMPLING_DATA_STRUCTURE_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/LightTreeBuilder.h"
#include "Scene/SceneParser.h"

class GPURenderer;

class LightTreeSamplingDataStructure
{
public:
	LightTreeSamplingDataStructure() : m_renderer(nullptr) {}
	LightTreeSamplingDataStructure(GPURenderer* renderer) : m_renderer(renderer) {}

	void compute_from_scene(const Scene& scene);
	void compute(
		const std::vector<int>& emissive_triangle_indices,
		const std::vector<float3>& vertices_positions,
		const std::vector<int>& triangles_vertex_indices,
		const std::vector<int>& material_indices,
		const std::vector<CPUMaterial>& materials);

	void recompute();
	void free();

	bool is_needed(unsigned int emissive_count);

	LightTreeBuilderOptions& get_builder_options();

private:
	LightTreeBuilder m_light_tree_builder;
	LightTreeBuilderDeviceData<OrochiBuffer> m_light_tree_device_data;

	GPURenderer* m_renderer = nullptr;
};

#endif
