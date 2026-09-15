/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_HIERARCHY_H
#define KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_HIERARCHY_H

#include "Device/kernels/HierarchicalAdaptiveSampling/Common.h"

HIPRT_DEVICE void hierarchical_adaptive_sampling_initialize_hierarchy(const HIPRTRenderData& render_data)
{
	int width								= render_data.render_settings.render_resolution.x;
	int height								= render_data.render_settings.render_resolution.y;
	HierarchicalAdaptiveSamplingNode* nodes = render_data.aux_buffers.hierarchical_adaptive_sampling_nodes;

	nodes[0].minimum_x = 0.0f;
	nodes[0].minimum_y = 0.0f;
	nodes[0].maximum_x = static_cast<float>(width);
	nodes[0].maximum_y = static_cast<float>(height);
	nodes[0].depth	   = 0;
	nodes[0].state	   = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

	*render_data.aux_buffers.hierarchical_adaptive_sampling_node_count = 1u;
}

HIPRT_DEVICE void hierarchical_adaptive_sampling_process_node(const HIPRTRenderData& render_data, unsigned int node_index)
{
	HierarchicalAdaptiveSamplingNode* nodes = render_data.aux_buffers.hierarchical_adaptive_sampling_nodes;
	HierarchicalAdaptiveSamplingNode& node	= nodes[node_index];

	unsigned int minimum_x = hierarchical_adaptive_sampling_floor(node.minimum_x);
	unsigned int minimum_y = hierarchical_adaptive_sampling_floor(node.minimum_y);
	unsigned int maximum_x = hierarchical_adaptive_sampling_ceil(node.maximum_x);
	unsigned int maximum_y = hierarchical_adaptive_sampling_ceil(node.maximum_y);

	float region_error_sum = hierarchical_adaptive_sampling_region_sum(render_data, minimum_x, minimum_y, maximum_x, maximum_y);
	float raster_area	   = static_cast<float>((maximum_x - minimum_x) * (maximum_y - minimum_y));
	float region_noise	   = region_error_sum / raster_area;
	float noise_threshold  = render_data.render_settings.hierarchical_adaptive_sampling_target_error;

	if (region_noise <= noise_threshold)
	{
		node.state = HierarchicalAdaptiveSamplingNodeState::COMPLETE;

		return;
	}

	float extent_x	   = node.maximum_x - node.minimum_x;
	float extent_y	   = node.maximum_y - node.minimum_y;
	bool split_x	   = extent_x >= extent_y;
	float split_extent = split_x ? extent_x : extent_y;
	if (split_extent <= 1.0f || node.depth >= static_cast<unsigned int>(render_data.render_settings.hierarchical_adaptive_sampling_max_depth))
	{
		node.state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

		return;
	}

	float split_position = hierarchical_adaptive_sampling_find_split(render_data, node, split_x, region_error_sum);
	float split_minimum	 = split_x ? node.minimum_x : node.minimum_y;
	float split_maximum	 = split_x ? node.maximum_x : node.maximum_y;
	if (split_position <= split_minimum + 1.0e-4f || split_position >= split_maximum - 1.0e-4f)
	{
		node.state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

		return;
	}

	float perpendicular_extent		 = split_x ? extent_y : extent_x;
	float left_child_split_extent	 = split_position - split_minimum;
	float right_child_split_extent	 = split_maximum - split_position;
	float left_child_largest_extent	 = left_child_split_extent > perpendicular_extent ? left_child_split_extent : perpendicular_extent;
	float right_child_largest_extent = right_child_split_extent > perpendicular_extent ? right_child_split_extent : perpendicular_extent;
	float minimum_cell_extent		 = render_data.render_settings.hierarchical_adaptive_sampling_minimum_cell_extent;
	if (left_child_largest_extent < minimum_cell_extent || right_child_largest_extent < minimum_cell_extent)
	{
		node.state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

		return;
	}

	unsigned int node_capacity = render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity;
	// A full binary tree always has an odd node count. Ignore a final unusable slot when the requested capacity is even.
	unsigned int usable_node_capacity = (node_capacity & 1u) == 0u ? node_capacity - 1u : node_capacity;
	unsigned int left_child;

	left_child = hippt::atomic_fetch_add(render_data.aux_buffers.hierarchical_adaptive_sampling_node_count, 2u);
	if (left_child + 2u > usable_node_capacity)
	{
		// Exhausting the bounded tree must only cost performance, never samples, so keeping the node active
		node.state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

		return;
	}

	unsigned int right_child = left_child + 1u;
	nodes[left_child]		 = node;
	nodes[right_child]		 = node;
	nodes[left_child].depth	 = node.depth + 1u;
	nodes[right_child].depth = node.depth + 1u;
	nodes[left_child].state	 = HierarchicalAdaptiveSamplingNodeState::ACTIVE;
	nodes[right_child].state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;
	if (split_x)
	{
		nodes[left_child].maximum_x	 = split_position;
		nodes[right_child].minimum_x = split_position;
		node.state					 = HierarchicalAdaptiveSamplingNodeState::SPLIT_X;
	}
	else
	{
		nodes[left_child].maximum_y	 = split_position;
		nodes[right_child].minimum_y = split_position;
		node.state					 = HierarchicalAdaptiveSamplingNodeState::SPLIT_Y;
	}

	node.split_position = split_position;
	node.left_child		= left_child;
	node.right_child	= right_child;
}

HIPRT_DEVICE void hierarchical_adaptive_sampling_build_hierarchy(const HIPRTRenderData& render_data, unsigned int thread_index, unsigned int build_depth)
{
	if (build_depth == static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::INITIALIZE))
	{
		if (thread_index == 0)
		{
			hierarchical_adaptive_sampling_initialize_hierarchy(render_data);
			*render_data.aux_buffers.hierarchical_adaptive_sampling_level_node_count = 1u;
		}

		return;
	}

	if (build_depth == static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::PREPARE_LEVEL))
	{
		if (thread_index == 0)
		{
			unsigned int node_capacity		  = render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity;
			unsigned int usable_node_capacity = (node_capacity & 1u) == 0u ? node_capacity - 1u : node_capacity;
			unsigned int node_count			  = *render_data.aux_buffers.hierarchical_adaptive_sampling_node_count;

			*render_data.aux_buffers.hierarchical_adaptive_sampling_level_node_count = node_count < usable_node_capacity ? node_count : usable_node_capacity;
		}

		return;
	}

	if (build_depth == static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::FINALIZE))
	{
		if (thread_index == 0)
		{
			unsigned int node_capacity		  = render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity;
			unsigned int usable_node_capacity = (node_capacity & 1u) == 0u ? node_capacity - 1u : node_capacity;
			unsigned int node_count			  = *render_data.aux_buffers.hierarchical_adaptive_sampling_node_count;

			*render_data.aux_buffers.hierarchical_adaptive_sampling_node_count = node_count < usable_node_capacity ? node_count : usable_node_capacity;
		}

		return;
	}

	unsigned int node_capacity		  = render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity;
	unsigned int usable_node_capacity = (node_capacity & 1u) == 0u ? node_capacity - 1u : node_capacity;
	if (thread_index >= usable_node_capacity)
		return;

	unsigned int level_node_count = *render_data.aux_buffers.hierarchical_adaptive_sampling_level_node_count;
	if (thread_index >= level_node_count || render_data.aux_buffers.hierarchical_adaptive_sampling_nodes[thread_index].depth != build_depth)
		return;

	hierarchical_adaptive_sampling_process_node(render_data, thread_index);
}

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) HierarchicalAdaptiveSamplingBuildHierarchy(unsigned int build_depth)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline HierarchicalAdaptiveSamplingBuildHierarchy(HIPRTRenderData render_data, unsigned int thread_index, unsigned int build_depth)
#endif
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA);
	unsigned int thread_index	 = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	hierarchical_adaptive_sampling_build_hierarchy(render_data, thread_index, build_depth);
}

#endif // #ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_HIERARCHY_H
