/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NODE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NODE_DEVICE_H

#include "HostDeviceCommon/Maths/VecTypes.h"

enum IlluminationAwareKDTreeNodeFlags : unsigned char
{
	IlluminationAwareKDTreeNodeFlag_None		= 0,
	IlluminationAwareKDTreeNodeFlag_Guiding		= 1u << 0,
	IlluminationAwareKDTreeNodeFlag_Lookahead	= 1u << 1,
	IlluminationAwareKDTreeNodeFlag_HasChildren = 1u << 2
};

struct IlluminationAwareKDTreeNode
{
	static constexpr unsigned int INVALID_NODE_INDEX			 = 0xFFFFFFFFu;
	static constexpr unsigned int INVALID_LIGHT_CLUSTERING_INDEX = 0xFFFFFFFFu;
	static constexpr unsigned int INVALID_CREATION_TAG			 = 0xFFFFFFFFu;
	static constexpr unsigned char INVALID_SPLIT_AXIS			 = 255;

	// Index of the left child.
	//
	// The right child is always left_child_index + 1.
	// INVALID_NODE_INDEX means that this node currently has no children.
	unsigned int left_child_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	// Index of the per-cell adaptive light clustering.
	//
	// Only an active guiding cell owns a valid light clustering.
	unsigned int light_clustering_normal_set_index = INVALID_LIGHT_CLUSTERING_INDEX;

	// Identifies the lookahead-allocation pass that created this node.
	//
	// It is used when replaying the current SPP's samples into only
	// newly created nodes.
	unsigned int creation_tag = IlluminationAwareKDTreeNode::INVALID_CREATION_TAG;

	// The split plane belongs to this parent node, not to its children.
	float split_position = 0.0f;

	unsigned char split_axis = IlluminationAwareKDTreeNode::INVALID_SPLIT_AXIS;
	unsigned char flags		 = 0;
};

struct IlluminationAwareKDTreeNodeBounds
{
	float3_t minimum;
	float3_t maximum;
};

#endif
