/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeCoreDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterDevice.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNISMLDevice.h"

struct IlluminationAwareKDTreeDevice
{
	HIPRT_DEVICE unsigned int resolve_lightcut(const IlluminationAwareKDTreeSGShadingContext& context,
											   unsigned int mesh_id,
											   unsigned int* out_guiding_node_index = nullptr,
											   unsigned int* out_normal_face		= nullptr,
											   unsigned int* out_set_index			= nullptr) const
	{
		if (out_guiding_node_index != nullptr)
			*out_guiding_node_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
		if (out_normal_face != nullptr)
			*out_normal_face = 0u;
		if (out_set_index != nullptr)
			*out_set_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

		if (core.nodes == nullptr || core.node_capacity == 0 || learning_to_cluster.normal_lightcut_sets == nullptr ||
			learning_to_cluster.normal_lightcut_set_capacity == 0)
			return IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

		unsigned int guiding_node_index = core.find_guiding_cell(context.position);
		if (guiding_node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX || guiding_node_index >= core.node_capacity)
			return IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
		if (out_guiding_node_index != nullptr)
			*out_guiding_node_index = guiding_node_index;

		unsigned int normal_face = illumination_aware_kd_tree_classify_surface_normal_face(context.shading_normal);
		if (out_normal_face != nullptr)
			*out_normal_face = normal_face;

		unsigned int set_index = core.nodes[guiding_node_index].lightcut_normal_set_index;
		if (set_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX || set_index >= learning_to_cluster.normal_lightcut_set_capacity)
			return IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;
		if (out_set_index != nullptr)
			*out_set_index = set_index;

		unsigned int lightcut_index = learning_to_cluster.resolve_lightcut_for_normal_face(set_index, normal_face, mesh_id);
		if (lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX || lightcut_index >= learning_to_cluster.lightcut_capacity)
			return IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

		return lightcut_index;
	}

	IlluminationAwareKDTreeCoreDevice core;
	IlluminationAwareKDTreeLearningToClusterDevice learning_to_cluster;
	IlluminationAwareKDTreeNISMLDevice nisml;

	unsigned char* any_cell_needs_split = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H
