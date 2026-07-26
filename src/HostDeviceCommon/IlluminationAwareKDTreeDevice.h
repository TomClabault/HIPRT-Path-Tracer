/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_DEVICE_H
#define HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_DEVICE_H

#include "HostDeviceCommon/IlluminationAwareKDTreeNodeDevice.h"

#include <cstdint>

enum class IlluminationAwareKDTreeSubdivisionMode
{
	Disabled,
	RecordSamplesOnly,
	MeanRadianceOnly,
	Full
};

struct IlluminationTreeDebugCounters
{
	uint32_t training_sample_count;
	uint32_t invalid_training_sample_count;
	uint32_t training_buffer_overflow_count;

	uint32_t physical_node_count;
	uint32_t active_guiding_cell_count;
	uint32_t lookahead_node_count;

	uint32_t created_lookahead_count;
	uint32_t real_split_count;

	uint32_t mean_radiance_split_count;
	uint32_t mean_direction_split_count;

	uint32_t invalid_tree_traversal_count;
	uint32_t node_capacity_overflow_count;
};

struct IlluminationAwareKDTreeDevice
{
	IlluminationAwareKDTreeSubdivisionMode subdivision_mode = IlluminationAwareKDTreeSubdivisionMode::Disabled;
	IlluminationTreeDebugCounters debug_counters			= {};

	IlluminationAwareKDTreeNode* nodes			   = nullptr;
	IlluminationAwareKDTreeNodeBounds* node_bounds = nullptr;

	uint32_t* node_count   = nullptr;
	uint32_t node_capacity = 0;
};

#endif
