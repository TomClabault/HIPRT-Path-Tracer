/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_SAMPLING_DATA_STRUCTURE_H
#define RENDERER_LIGHT_TREE_SG_SAMPLING_DATA_STRUCTURE_H

#include "Compiler/GPUKernelCompilerOptions.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/LightTree/LightTreeSGBuilder.h"
#include "Scene/SceneParser.h"

class GPURenderer;

class LightTreeSGSamplingDataStructure
{
public:
	LightTreeSGSamplingDataStructure() : m_renderer(nullptr) {}
	LightTreeSGSamplingDataStructure(GPURenderer* renderer) : m_renderer(renderer) {}

	void compute_from_scene(const Scene& scene, std::shared_ptr<GPUKernelCompilerOptions> compiler_options);
	void compute(std::shared_ptr<GPUKernelCompilerOptions> compiler_options,
				 const std::vector<int>& emissive_triangle_indices,
				 const std::vector<float>& triangles_average_emissive_power_luminance,
				 const std::vector<float3_t>& vertices_positions,
				 const std::vector<int>& triangles_vertex_indices);

	void recompute_if_needed_or_free(std::shared_ptr<GPUKernelCompilerOptions> compiler_options, bool skip_if_already_computed = false);
	void free();

	bool is_needed(unsigned int emissive_count, std::shared_ptr<GPUKernelCompilerOptions> compiler_option);

	size_t get_VRAM_usage_bytes() const;

	LightTreeSGBuilderOptions& get_builder_options();
	LightTreeSGBuilder& get_builder();
	int get_spatial_lobe_count() const;
	void set_spatial_lobe_count(int spatial_lobe_count);
	int get_tree_cut_size() const;
	void set_tree_cut_size(int tree_cut_size);
	int get_tree_cut_size_neural_many_lights() const;
	void set_tree_cut_size_neural_many_lights(int tree_cut_size_neural_many_lights);
	int get_second_tree_cut_size() const;
	void set_second_tree_cut_size(int second_tree_cut_size);
	const LightTreeSGBuildResult<OrochiBuffer>& get_build_result() const;

private:
	LightTreeSGBuilder m_light_tree_builder_sg;
	LightTreeSGBuildResult<OrochiBuffer> m_light_tree_sg_build_result;

	GPURenderer* m_renderer = nullptr;
};

#endif // #ifndef RENDERER_LIGHT_TREE_SG_SAMPLING_DATA_STRUCTURE_H
