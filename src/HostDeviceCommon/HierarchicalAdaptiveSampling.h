/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_HIERARCHICAL_ADAPTIVE_SAMPLING_H
#define HOST_DEVICE_COMMON_HIERARCHICAL_ADAPTIVE_SAMPLING_H

enum class HierarchicalAdaptiveSamplingBuildCommand : unsigned int
{
	INITIALIZE	  = 0xffffffffu,
	PREPARE_LEVEL = 0xfffffffeu,
	FINALIZE	  = 0xfffffffdu
};

enum class HierarchicalAdaptiveSamplingNodeState : unsigned int
{
	ACTIVE	 = 0,
	COMPLETE = 1,
	SPLIT_X	 = 2,
	SPLIT_Y	 = 3
};

struct HierarchicalAdaptiveSamplingNode
{
	float minimum_x		 = 0.0f;
	float minimum_y		 = 0.0f;
	float maximum_x		 = 0.0f;
	float maximum_y		 = 0.0f;
	float split_position = 0.0f;

	unsigned int left_child						= 0;
	unsigned int right_child					= 0;
	unsigned int depth							= 0;
	HierarchicalAdaptiveSamplingNodeState state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;
};

#endif // #ifndef HOST_DEVICE_COMMON_HIERARCHICAL_ADAPTIVE_SAMPLING_H
