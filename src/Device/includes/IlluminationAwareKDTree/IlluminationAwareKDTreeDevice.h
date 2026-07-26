/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDirectIlluminationTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/KDTreeIlluminationSignature.h"
#include "Device/includes/IlluminationAwareKDTree/KDTreeSpatialSampleMoments.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

#include <cstdint>

enum class IlluminationAwareKDTreeSubdivisionMode
{
	DISABLED,
	RECORD_SAMPLES_ONLY,
	MEAN_RADIANCE_ONLY,
	FULL
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
	HIPRT_DEVICE uint32_t find_guiding_cell(const float3_t position) const
	{
		uint32_t node_index = 0;

		while (true)
		{
			const IlluminationAwareKDTreeNode& node = nodes[node_index];

			if (node.flags & IlluminationAwareKDTreeNodeFlag_Guiding)
				// We stop at the first guiding cell, even if it may have lookahead cells, we only want guiding cells from this function
				return node_index;

			const uint32_t left_child_index	 = node.left_child_index;
			const uint32_t right_child_index = left_child_index + 1;
			const float* position_components = &position.x;

			if (position_components[node.split_axis] < node.split_position)
				node_index = left_child_index;
			else
				node_index = right_child_index;
		}
	}

	HIPRT_DEVICE void append_direct_illumination_training_sample(const IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample)
	{
		// Invalid samples must not consume buffer space or affect b0.
		if (!sample.valid)
			return;

		const uint32_t sample_index = hippt::atomic_fetch_add(training_sample_count, 1u);

		// The counter may exceed capacity, but memory must never be written
		// outside the allocated buffer.
		if (sample_index >= training_sample_capacity)
			return;

		training_samples[sample_index] = sample;
	}

	IlluminationAwareKDTreeSubdivisionMode subdivision_mode = IlluminationAwareKDTreeSubdivisionMode::DISABLED;
	IlluminationTreeDebugCounters debug_counters			= {};

	IlluminationAwareKDTreeNode* nodes			   = nullptr;
	IlluminationAwareKDTreeNodeBounds* node_bounds = nullptr;

	uint32_t* node_count   = nullptr;
	uint32_t node_capacity = 0;

	uint32_t* active_guiding_nodes		= nullptr;
	uint32_t* active_guiding_node_count = nullptr;

	IlluminationAwareKDTreeDirectIlluminationTrainingSample* training_samples = nullptr;
	AtomicType<uint32_t>* training_sample_count								  = nullptr;
	uint32_t training_sample_capacity										  = 0;
};

#endif
