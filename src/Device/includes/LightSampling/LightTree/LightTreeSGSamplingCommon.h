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

struct IlluminationAwareKDTreeConditionalLightTreeSample
{
	unsigned int light_leaf_index	   = 0;
	float conditional_leaf_probability = 0.0f;
};

HIPRT_DEVICE IlluminationAwareKDTreeConditionalLightTreeSample sample_light_tree_subtree(const LightTreeSGNodeDevice* nodes,
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
			return IlluminationAwareKDTreeConditionalLightTreeSample{};

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

	return IlluminationAwareKDTreeConditionalLightTreeSample{ node_index, conditional_probability };
}

HIPRT_DEVICE LightSampleInformation sample_light_inside_nis_cluster(const HIPRTRenderData& render_data,
																	unsigned int cluster_node_index,
																	float3_t shading_point,
																	float3_t view_direction,
																	float3_t shading_normal,
																	const SGSpecularImportanceData& spec_data,
																	float specular,
																	float alpha_x,
																	float alpha_y,
																	Xorshift32Generator& random_number_generator)
{
	const LightTreeSGNodeDevice* nodes								  = render_data.light_tree_sg.nodes;
	IlluminationAwareKDTreeConditionalLightTreeSample sampled_subtree = sample_light_tree_subtree(
		nodes, cluster_node_index, shading_point, view_direction, shading_normal, spec_data, specular, alpha_x, alpha_y, random_number_generator);

	if (!(sampled_subtree.conditional_leaf_probability > 0.0f))
		return LightSampleInformation();

	const LightTreeSGNodeDevice& sampled_leaf = nodes[sampled_subtree.light_leaf_index];
	if (sampled_leaf.triangle_count == 0)
		return LightSampleInformation();

	int index		   = sampled_leaf.left_child_index_or_first_triangle_index + random_number_generator.random_index(sampled_leaf.triangle_count);
	int triangle_index = render_data.light_tree_sg.indices_array[index];

	LightSampleInformation light_sample;
	light_sample.emissive_triangle_global_index = render_data.buffers.emissive_triangles_primitive_indices[triangle_index];
	light_sample.pdf							= sampled_subtree.conditional_leaf_probability / sampled_leaf.triangle_count;

	return light_sample;
}

#endif
