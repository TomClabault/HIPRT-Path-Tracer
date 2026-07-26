/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_DEVICE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDirectIlluminationTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeNodeDevice.h"

#include "Device/includes/IlluminationAwareKDTree/KDTreeIlluminationSignature.h"
#include "Device/includes/IlluminationAwareKDTree/KDTreeSpatialSampleMoments.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeOptions.h"

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

	HIPRT_DEVICE void atomic_add_illumination_signature(IlluminationAwareKDTreeIlluminationSignature* signatures,
														const uint32_t node_index,
														const IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample)
	{
		const float radiance_weight = sample.radiance_weight;

		// b0 counts all valid samples, including samples with L == 0.
		hippt::atomic_fetch_add_gpu(&signatures[node_index].valid_observation_count, 1u);

		hippt::atomic_fetch_add_gpu(&signatures[node_index].scalar_radiance_sum, radiance_weight);
		hippt::atomic_fetch_add_gpu(&signatures[node_index].squared_scalar_radiance_sum, radiance_weight * radiance_weight);

		hippt::atomic_fetch_add_gpu(&signatures[node_index].weighted_direction_sum.x, radiance_weight * sample.incoming_direction.x);
		hippt::atomic_fetch_add_gpu(&signatures[node_index].weighted_direction_sum.y, radiance_weight * sample.incoming_direction.y);
		hippt::atomic_fetch_add_gpu(&signatures[node_index].weighted_direction_sum.z, radiance_weight * sample.incoming_direction.z);
	}

	HIPRT_DEVICE void atomic_add_spatial_moments(IlluminationAwareKDTreeSpatialSampleMoments* moments, const uint32_t node_index, const float3_t position)
	{
		hippt::atomic_fetch_add_gpu(&moments[node_index].positive_radiance_sample_count, 1u);

		hippt::atomic_fetch_add_gpu(&moments[node_index].position_sum.x, position.x);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_sum.y, position.y);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_sum.z, position.z);

		hippt::atomic_fetch_add_gpu(&moments[node_index].position_squared_sum.x, position.x * position.x);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_squared_sum.y, position.y * position.y);
		hippt::atomic_fetch_add_gpu(&moments[node_index].position_squared_sum.z, position.z * position.z);
	}

	HIPRT_DEVICE void accumulate_sample_into_existing_tree(const IlluminationAwareKDTreeDirectIlluminationTrainingSample& sample)
	{
		// First find the active guiding cell used at this position.
		uint32_t node_index = find_guiding_cell(sample.position);

		// Invalid traversal indicates a broken topology.
		if (node_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			return;

		// Level zero is the guiding cell.
		//
		// Levels one through six are the lookahead cells along the sample's
		// unique spatial path.
		for (uint32_t level = 0; level <= IlluminationAwareKDTreeMaximumLookaheadDepth; level++)
		{
			atomic_add_illumination_signature(batch_signatures, node_index, sample);

			// Candidate k-d split placement uses only non-zero samples.
			//
			// Zero-radiance samples still contribute to the illumination
			// signature above, but not to the spatial mean and variance.
			if (sample.radiance_weight > 0.0f)
				atomic_add_spatial_moments(batch_spatial_moments, node_index, sample.position);

			// Level six is the deepest lookahead level.
			if (level == IlluminationAwareKDTreeMaximumLookaheadDepth)
				break;

			IlluminationAwareKDTreeNode& node = nodes[node_index];

			// A missing child means that this lookahead path has not yet been
			// constructed deeply enough. Accumulation stops here.
			if (!(node.flags & IlluminationAwareKDTreeNodeFlag_HasChildren))
				break;

			uint32_t left_child_index		 = node.left_child_index;
			uint32_t right_child_index		 = left_child_index + 1;
			const float* position_components = &sample.position.x;

			// Follow exactly one child because the sample position belongs to
			// exactly one k-d cell at this level.
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

		uint32_t sample_index = hippt::atomic_fetch_add(training_sample_count, 1u);

		// The counter may exceed capacity, but memory must never be written
		// outside the allocated buffer.
		if (sample_index >= training_sample_capacity)
			return;

		training_samples[sample_index] = sample;
	}

	HIPRT_DEVICE void compute_split_axis_and_position(const IlluminationAwareKDTreeSpatialSampleMoments& moments,
													  uint8_t& out_split_axis,
													  float& out_split_position) const
	{
		float countf = static_cast<float>(moments.positive_radiance_sample_count);
		if (countf <= 0.0f)
		{
			out_split_axis	   = 255;
			out_split_position = 0.0f;

			return;
		}

		float3_t mean_position = moments.position_sum / countf;
		float3_t variance	   = (moments.position_squared_sum / countf) - (mean_position * mean_position);

		if (variance.x >= variance.y && variance.x >= variance.z)
		{
			out_split_axis	   = 0;
			out_split_position = mean_position.x;
		}
		else if (variance.y >= variance.z)
		{
			out_split_axis	   = 1;
			out_split_position = mean_position.y;
		}
		else
		{
			out_split_axis	   = 2;
			out_split_position = mean_position.z;
		}
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

	IlluminationAwareKDTreeIlluminationSignature* batch_signatures	 = nullptr;
	IlluminationAwareKDTreeIlluminationSignature* history_signatures = nullptr;

	IlluminationAwareKDTreeSpatialSampleMoments* batch_spatial_moments	 = nullptr;
	IlluminationAwareKDTreeSpatialSampleMoments* history_spatial_moments = nullptr;
};

#endif
