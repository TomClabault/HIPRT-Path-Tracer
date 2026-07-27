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

struct IlluminationAwareKDTreeDevice
{
	HIPRT_DEVICE unsigned int find_guiding_cell(const float3_t position) const
	{
		unsigned int node_index = 0;

		while (true)
		{
			const IlluminationAwareKDTreeNode& node = nodes[node_index];

			if (node.flags & IlluminationAwareKDTreeNodeFlag_Guiding)
				// We stop at the first guiding cell, even if it may have lookahead cells, we only want guiding cells from this function
				return node_index;

			const unsigned int left_child_index	 = node.left_child_index;
			const unsigned int right_child_index = left_child_index + 1;
			const float* position_components	 = &position.x;

			if (position_components[node.split_axis] < node.split_position)
				node_index = left_child_index;
			else
				node_index = right_child_index;
		}
	}

	HIPRT_DEVICE void atomic_add_illumination_signature(IlluminationAwareKDTreeIlluminationSignature* signatures,
														const unsigned int node_index,
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

	HIPRT_DEVICE void atomic_add_spatial_moments(IlluminationAwareKDTreeSpatialSampleMoments* moments, const unsigned int node_index, const float3_t position)
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
		unsigned int node_index = find_guiding_cell(sample.position);

		// Level zero is the guiding cell.
		//
		// Levels one through six are the lookahead cells along the sample's
		// unique spatial path.
		for (unsigned int level = 0; level <= IlluminationAwareKDTreeMaximumLookaheadDepth; level++)
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

			unsigned int left_child_index	 = node.left_child_index;
			unsigned int right_child_index	 = left_child_index + 1;
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

		unsigned int sample_index = hippt::atomic_fetch_add(training_sample_count, 0u);
		// The counter may exceed capacity, but memory must never be written
		// outside the allocated buffer.
		if (sample_index >= training_sample_capacity)
			return;

		sample_index = hippt::atomic_fetch_add(training_sample_count, 1u);
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
			out_split_axis	   = IlluminationAwareKDTreeNode::INVALID_SPLIT_AXIS;
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

	HIPRT_DEVICE unsigned int reserve_physical_nodes(const unsigned int amount_to_reserve)
	{
		// Read the current value without modifying it.
		unsigned int current_count = hippt::atomic_fetch_add(node_count, 0u);

		// Another thread may modify node_count between our read and write,
		// so retry until we either reserve the range or discover that the
		// pool is full.
		while (true)
		{
			// Writing "current_count + amount > capacity" could overflow
			if (current_count + amount_to_reserve > node_capacity)
				return IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

			unsigned int desired_count	= current_count + amount_to_reserve;
			unsigned int observed_count = hippt::atomic_compare_exchange(node_count, current_count, desired_count);

			if (observed_count == current_count)
				// The compare-and-swap succeeded.
				return current_count;

			// Another thread allocated nodes first, atomicCAS returned the newer counter value, so retry from it.
			current_count = observed_count;
		}
	}

	HIPRT_DEVICE void append_child_pair_to_frontier(unsigned int* next_frontier,
													AtomicType<unsigned int>* next_frontier_count,
													const unsigned int left_child,
													const unsigned int right_child)
	{
		// Atomically reserve two contiguous frontier entries.
		const unsigned int output_index = hippt::atomic_fetch_add(next_frontier_count, 2u);

		next_frontier[output_index + 0] = left_child;
		next_frontier[output_index + 1] = right_child;
	}

	int minimum_sample_count_for_lookahead_creation = 1000;

	IlluminationAwareKDTreeSubdivisionMode subdivision_mode = IlluminationAwareKDTreeSubdivisionMode::DISABLED;

	IlluminationAwareKDTreeNode* nodes			   = nullptr;
	IlluminationAwareKDTreeNodeBounds* node_bounds = nullptr;

	AtomicType<unsigned int>* node_count = nullptr;
	unsigned int node_capacity			 = 0;

	unsigned int* active_guiding_nodes					= nullptr;
	AtomicType<unsigned int>* active_guiding_node_count = nullptr;

	// Two ping ponging frontier buffers for when we create lookahead cells
	//
	// Nodes at the lookahead depth currently being processed.
	unsigned int* current_frontier					 = nullptr;
	AtomicType<unsigned int>* current_frontier_count = nullptr;
	// Children that form the next lookahead depth.
	unsigned int* next_frontier					  = nullptr;
	AtomicType<unsigned int>* next_frontier_count = nullptr;

	IlluminationAwareKDTreeDirectIlluminationTrainingSample* training_samples = nullptr;
	AtomicType<unsigned int>* training_sample_count							  = nullptr;
	unsigned int training_sample_capacity									  = 0;

	IlluminationAwareKDTreeIlluminationSignature* batch_signatures	 = nullptr;
	IlluminationAwareKDTreeIlluminationSignature* history_signatures = nullptr;

	IlluminationAwareKDTreeSpatialSampleMoments* batch_spatial_moments	 = nullptr;
	IlluminationAwareKDTreeSpatialSampleMoments* history_spatial_moments = nullptr;
};

#endif
