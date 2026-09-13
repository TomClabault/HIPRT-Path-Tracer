/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_COMMON_H
#define KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_COMMON_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

HIPRT_DEVICE float hierarchical_adaptive_sampling_summed_area_value(const HIPRTRenderData& render_data, int x, int y)
{
	if (x < 0 || y < 0)
		return 0.0f;

	int width = render_data.render_settings.render_resolution.x;
	return render_data.aux_buffers.hierarchical_adaptive_sampling_summed_area[x + y * width];
}

HIPRT_DEVICE float hierarchical_adaptive_sampling_region_sum(
	const HIPRTRenderData& render_data, unsigned int minimum_x, unsigned int minimum_y, unsigned int maximum_x, unsigned int maximum_y)
{
	if (maximum_x <= minimum_x || maximum_y <= minimum_y)
		return 0.0f;

	int x0 = static_cast<int>(minimum_x) - 1;
	int y0 = static_cast<int>(minimum_y) - 1;
	int x1 = static_cast<int>(maximum_x) - 1;
	int y1 = static_cast<int>(maximum_y) - 1;

	return hierarchical_adaptive_sampling_summed_area_value(render_data, x1, y1) - hierarchical_adaptive_sampling_summed_area_value(render_data, x0, y1) -
		   hierarchical_adaptive_sampling_summed_area_value(render_data, x1, y0) + hierarchical_adaptive_sampling_summed_area_value(render_data, x0, y0);
}

HIPRT_DEVICE unsigned int hierarchical_adaptive_sampling_floor(float value)
{
	return static_cast<unsigned int>(value);
}

HIPRT_DEVICE unsigned int hierarchical_adaptive_sampling_ceil(float value)
{
	unsigned int integer_value = static_cast<unsigned int>(value);
	return static_cast<float>(integer_value) < value ? integer_value + 1u : integer_value;
}

HIPRT_DEVICE float hierarchical_adaptive_sampling_find_split(const HIPRTRenderData& render_data,
															 const HierarchicalAdaptiveSamplingNode& node,
															 bool split_x,
															 float total_error)
{
	unsigned int minimum_x	= hierarchical_adaptive_sampling_floor(node.minimum_x);
	unsigned int minimum_y	= hierarchical_adaptive_sampling_floor(node.minimum_y);
	unsigned int maximum_x	= hierarchical_adaptive_sampling_ceil(node.maximum_x);
	unsigned int maximum_y	= hierarchical_adaptive_sampling_ceil(node.maximum_y);
	float target_error		= total_error * 0.5f;
	float accumulated_error = 0.0f;

	unsigned int first_coordinate = split_x ? minimum_x : minimum_y;
	unsigned int last_coordinate  = split_x ? maximum_x : maximum_y;
	for (unsigned int coordinate = first_coordinate; coordinate < last_coordinate; coordinate++)
	{
		float slice_error;
		if (split_x)
			slice_error = hierarchical_adaptive_sampling_region_sum(render_data, coordinate, minimum_y, coordinate + 1u, maximum_y);
		else
			slice_error = hierarchical_adaptive_sampling_region_sum(render_data, minimum_x, coordinate, maximum_x, coordinate + 1u);

		if (accumulated_error + slice_error >= target_error)
		{
			float fraction = slice_error > 0.0f ? (target_error - accumulated_error) / slice_error : 0.5f;
			return static_cast<float>(coordinate) + fraction;
		}

		accumulated_error += slice_error;
	}

	return (split_x ? node.minimum_x + node.maximum_x : node.minimum_y + node.maximum_y) * 0.5f;
}

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded structure as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char HIERARCHICAL_ADAPTIVE_SAMPLING_RENDER_DATA[sizeof(HIPRTRenderData)];
}
#endif // #ifdef __KERNELCC__

#endif // #ifndef KERNELS_HIERARCHICAL_ADAPTIVE_SAMPLING_COMMON_H
