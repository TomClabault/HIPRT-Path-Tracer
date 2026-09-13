/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_RESOLVE_MASK_H
#define KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_RESOLVE_MASK_H

#include "Device/kernels/HierarchicalAdaptiveSampling/Common.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) HierarchicalAdaptiveSamplingResolveMask()
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) inline HierarchicalAdaptiveSamplingResolveMask(HIPRTRenderData render_data, int x, int y)
#endif // #ifdef __KERNELCC__
{
#ifdef __KERNELCC__
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA);
	unsigned int x				 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y				 = blockIdx.y * blockDim.y + threadIdx.y;
#endif // #ifdef __KERNELCC__
	int width  = render_data.render_settings.render_resolution.x;
	int height = render_data.render_settings.render_resolution.y;
	if (x >= width || y >= height)
		return;

	HierarchicalAdaptiveSamplingNode* nodes = render_data.aux_buffers.hierarchical_adaptive_sampling_nodes;

	float pixel_center_x	= static_cast<float>(x) + 0.5f;
	float pixel_center_y	= static_cast<float>(y) + 0.5f;
	unsigned int node_index = 0;

	for (unsigned int depth = 0; depth <= static_cast<unsigned int>(render_data.render_settings.hierarchical_adaptive_sampling_max_depth); depth++)
	{
		HierarchicalAdaptiveSamplingNode& node = nodes[node_index];
		if (node.state == HierarchicalAdaptiveSamplingNodeState::ACTIVE || node.state == HierarchicalAdaptiveSamplingNodeState::COMPLETE)
		{
			unsigned int pixel_index										  = x + y * width;
			bool active														  = node.state == HierarchicalAdaptiveSamplingNodeState::ACTIVE;
			render_data.aux_buffers.pixel_active[pixel_index]				  = active;
			render_data.aux_buffers.pixel_converged_sample_count[pixel_index] = active ? -1 : render_data.aux_buffers.pixel_sample_count[pixel_index];
			return;
		}

		bool use_left_child =
			node.state == HierarchicalAdaptiveSamplingNodeState::SPLIT_X ? pixel_center_x < node.split_position : pixel_center_y < node.split_position;
		node_index = use_left_child ? node.left_child : node.right_child;
	}

	// Malformed or deeper-than-configured trees are conservatively sampled.
	unsigned int pixel_index										  = x + y * width;
	render_data.aux_buffers.pixel_active[pixel_index]				  = true;
	render_data.aux_buffers.pixel_converged_sample_count[pixel_index] = -1;
}

#endif // #ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_RESOLVE_MASK_H
