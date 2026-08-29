/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_POWER_SAMPLING_DATA_STRUCTURE_H
#define RENDERER_POWER_SAMPLING_DATA_STRUCTURE_H

#include "Device/includes/AliasTable.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Scene/SceneParser.h"

class GPURenderer;

class PowerSamplingDataStructure
{
public:
	PowerSamplingDataStructure() : PowerSamplingDataStructure(nullptr) {};
	PowerSamplingDataStructure(GPURenderer* renderer);

	void compute_from_scene(const Scene& scene, std::shared_ptr<GPUKernelCompilerOptions> compiler_options);
	void compute(std::shared_ptr<GPUKernelCompilerOptions> compiler_options,
				 const std::vector<int>& emissive_triangle_indices,
				 const std::vector<float>& triangles_average_emissive_power_luminance,
				 AliasTableDevice& power_alias_table);

	void recompute_if_needed_or_free(std::shared_ptr<GPUKernelCompilerOptions> compiler_options, bool skip_if_already_computed = false);
	void free();

	bool is_needed(unsigned int emissive_count, std::shared_ptr<GPUKernelCompilerOptions> compiler_options);

private:
	OrochiBuffer<float> m_alias_table_probas;
	OrochiBuffer<int> m_alias_table_aliases;

	GPURenderer* m_renderer = nullptr;
};

#endif // #ifndef RENDERER_POWER_SAMPLING_DATA_STRUCTURE_H
