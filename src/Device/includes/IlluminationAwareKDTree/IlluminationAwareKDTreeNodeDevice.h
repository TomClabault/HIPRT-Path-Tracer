/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NODE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NODE_DEVICE_H

#include "HostDeviceCommon/Maths/VecTypes.h"

#include <cstdint>

enum IlluminationAwareKDTreeNodeFlags : uint8_t
{
	IlluminationAwareKDTreeNodeFlag_None		= 0,
	IlluminationAwareKDTreeNodeFlag_Guiding		= 1u << 0,
	IlluminationAwareKDTreeNodeFlag_Lookahead	= 1u << 1,
	IlluminationAwareKDTreeNodeFlag_HasChildren = 1u << 2
};

struct IlluminationAwareKDTreeNode
{
	static constexpr uint32_t INVALID_NODE_INDEX   = 0xFFFFFFFFu;
	static constexpr uint32_t INVALID_CREATION_TAG = 0xFFFFFFFFu;
	static constexpr uint8_t INVALID_SPLIT_AXIS	   = 255;

	// Index of the left child.
	//
	// The right child is always left_child_index + 1.
	// INVALID_NODE_INDEX means that this node currently has no children.
	uint32_t left_child_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

	// Index of the per-cell adaptive light clustering.
	//
	// Only an active guiding cell owns a valid light clustering.
	static constexpr unsigned int INVALID_LIGHTCUT_INDEX = 0xFFFFFFFFu;
	unsigned int lightcut_normal_set_index				 = INVALID_LIGHTCUT_INDEX;

	// Identifies the lookahead-allocation pass that created this node.
	//
	// It is used when replaying the current SPP's samples into only
	// newly created nodes.
	uint32_t creation_tag = 0xFFFFFFFFu;

	// The split plane belongs to this parent node, not to its children.
	float split_position = 0.0f;

	uint8_t split_axis = IlluminationAwareKDTreeNode::INVALID_SPLIT_AXIS;
	uint8_t flags	   = 0;
	uint16_t padding;
};

struct IlluminationAwareKDTreeNodeBounds
{
	float3_t minimum;
	float3_t maximum;
};

#endif
