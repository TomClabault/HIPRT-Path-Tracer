/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_HIERARCHY_H
#define KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_HIERARCHY_H

#include "Device/kernels/HierarchicalAdaptiveSampling/Common.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) HierarchicalAdaptiveSamplingBuildHierarchy()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline HierarchicalAdaptiveSamplingBuildHierarchy(HIPRTRenderData render_data, int thread_index)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA);
	unsigned int thread_index	 = blockIdx.x * blockDim.x + threadIdx.x;
#endif // #ifdef __KERNELCC__
	if (thread_index != 0)
		return;

	int width								= render_data.render_settings.render_resolution.x;
	int height								= render_data.render_settings.render_resolution.y;
	unsigned int node_capacity				= render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity;
	HierarchicalAdaptiveSamplingNode* nodes = render_data.aux_buffers.hierarchical_adaptive_sampling_nodes;

	nodes[0].minimum_x		= 0.0f;
	nodes[0].minimum_y		= 0.0f;
	nodes[0].maximum_x		= static_cast<float>(width);
	nodes[0].maximum_y		= static_cast<float>(height);
	nodes[0].depth			= 0;
	nodes[0].state			= HierarchicalAdaptiveSamplingNodeState::ACTIVE;
	unsigned int node_count = 1;

	float noise_threshold = render_data.render_settings.hierarchical_adaptive_sampling_target_error;

	for (unsigned int node_index = 0; node_index < node_count; node_index++)
	{
		HierarchicalAdaptiveSamplingNode& node = nodes[node_index];
		unsigned int minimum_x				   = hierarchical_adaptive_sampling_floor(node.minimum_x);
		unsigned int minimum_y				   = hierarchical_adaptive_sampling_floor(node.minimum_y);
		unsigned int maximum_x				   = hierarchical_adaptive_sampling_ceil(node.maximum_x);
		unsigned int maximum_y				   = hierarchical_adaptive_sampling_ceil(node.maximum_y);
		float region_error_sum				   = hierarchical_adaptive_sampling_region_sum(render_data, minimum_x, minimum_y, maximum_x, maximum_y);
		float raster_area					   = static_cast<float>((maximum_x - minimum_x) * (maximum_y - minimum_y));
		float region_noise					   = region_error_sum / raster_area;

		if (region_noise <= noise_threshold)
		{
			node.state = HierarchicalAdaptiveSamplingNodeState::COMPLETE;

			continue;
		}

		float extent_x	   = node.maximum_x - node.minimum_x;
		float extent_y	   = node.maximum_y - node.minimum_y;
		bool split_x	   = extent_x >= extent_y;
		float split_extent = split_x ? extent_x : extent_y;
		if (split_extent <= 1.0f || node.depth >= static_cast<unsigned int>(render_data.render_settings.hierarchical_adaptive_sampling_max_depth))
		{
			node.state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

			continue;
		}

		if (node_count + 2u > node_capacity)
		{
			// Exhausting the bounded tree must only cost performance, never samples, so keeping the node active
			node.state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

			continue;
		}

		float split_position = hierarchical_adaptive_sampling_find_split(render_data, node, split_x, region_error_sum);
		float split_minimum	 = split_x ? node.minimum_x : node.minimum_y;
		float split_maximum	 = split_x ? node.maximum_x : node.maximum_y;
		if (split_position <= split_minimum + 1.0e-4f || split_position >= split_maximum - 1.0e-4f)
		{
			node.state = HierarchicalAdaptiveSamplingNodeState::ACTIVE;

			continue;
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

			continue;
		}

		unsigned int left_child	 = node_count++;
		unsigned int right_child = node_count++;
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

	*render_data.aux_buffers.hierarchical_adaptive_sampling_node_count = node_count;
}

#endif // #ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_BUILD_HIERARCHY_H
