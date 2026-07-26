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
	IlluminationAwareKDTreeNodeFlag_HasChildren = 1u << 1
};

struct IlluminationAwareKDTreeNode
{
	static constexpr uint32_t INVALID_NODE_INDEX   = 0xFFFFFFFFu;
	static constexpr uint32_t INVALID_GUIDING_SLOT = 0xFFFFFFFFu;
	static constexpr uint32_t INVALID_CREATION_TAG = 0xFFFFFFFFu;

	// Index of the left child.
	//
	// The right child is always left_child_index + 1.
	// INVALID_NODE_INDEX means that this node currently has no children.
	uint32_t left_child_index;

	// Index of the NEE distribution used by this guiding cell.
	//
	// Real inner nodes and lookahead-only nodes use INVALID_GUIDING_SLOT.
	uint32_t guiding_distribution_index;

	// Identifies the lookahead-allocation pass that created this node.
	//
	// It is used when replaying the current SPP's samples into only
	// newly created nodes.
	uint32_t creation_tag;

	// The split plane belongs to this parent node, not to its children.
	float split_position;

	uint8_t split_axis;
	uint8_t flags;
	uint16_t padding;
};

struct IlluminationAwareKDTreeNodeBounds
{
	float3_t minimum;
	float3_t maximum;
};

#endif
