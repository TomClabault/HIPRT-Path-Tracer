/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_COMMON_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_SAMPLING_COMMON_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

struct SGSpecularImportanceData;

HIPRT_DEVICE float light_tree_sg_node_importance(const LightTreeSGNodeDevice& node,
												 const SGSpecularImportanceData& spec_data,
												 float3_t shading_point,
												 float3_t view_direction,
												 float3_t shading_normal,
												 float specular,
												 float alpha_x,
												 float alpha_y);

struct LightSubtreeSample
{
	unsigned int light_leaf_index	   = 0;
	float conditional_leaf_probability = 0.0f;
};

HIPRT_DEVICE LightSubtreeSample sample_light_tree_subtree(const LightTreeSGNodeDevice* nodes,
														  unsigned int subtree_root,
														  float3_t shading_point,
														  float3_t view_direction,
														  float3_t shading_normal,
														  const SGSpecularImportanceData& spec_data,
														  float specular,
														  float alpha_x,
														  float alpha_y,
														  Xorshift32Generator& random_number_generator)
{
	float conditional_probability = 1.0f;
	unsigned int node_index		  = subtree_root;

	while (nodes[node_index].triangle_count == 0)
	{
		unsigned int left_child_index  = nodes[node_index].left_child_index_or_first_triangle_index;
		unsigned int right_child_index = left_child_index + 1;

		float left_importance =
			light_tree_sg_node_importance(nodes[left_child_index], spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);
		float right_importance =
			light_tree_sg_node_importance(nodes[right_child_index], spec_data, shading_point, view_direction, shading_normal, specular, alpha_x, alpha_y);

		float importance_sum = left_importance + right_importance;
		if (!(importance_sum > 0.0f))
			return LightSubtreeSample{};

		float left_probability = left_importance / importance_sum;
		if (random_number_generator() < left_probability)
		{
			conditional_probability *= left_probability;
			node_index = left_child_index;
		}
		else
		{
			conditional_probability *= 1.0f - left_probability;
			node_index = right_child_index;
		}
	}

	return LightSubtreeSample{ node_index, conditional_probability };
}

#endif
