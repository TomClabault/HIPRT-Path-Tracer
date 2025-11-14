/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_NODE_H
#define RENDERER_LIGHT_TREE_SG_NODE_H

#include "HostDeviceCommon/Maths/Math.h"

struct LightTreeSGNode
{
	void compute_vmf()
	{
		float mean_axis_length = hippt::length(mean_axis);
		if (mean_axis_length < 1.0e-10f)
		{
			vmf_axis = make_float3(0.0f, 1.0f, 0.0f);
			vmf_sharpness = 0.0f;

			return;
		}
		else
			vmf_axis = mean_axis / mean_axis_length;

		if (mean_axis_length > 0.999999f)
			// Set sharpness to max value to avoid division by zero
			vmf_sharpness = 2199023255552.0f;
		else
			vmf_sharpness = hippt::min((3.0f * mean_axis_length - hippt::pow_3(mean_axis_length)) / (1.0f - hippt::square(mean_axis_length)), 2199023255552.0f);
	}

	float3 spatial_mean = make_float3(0.0f, 0.0f, 0.0f);
	float spatial_variance = 0.0f;

	float3 mean_axis = make_float3(0.0f, 0.0f, 0.0f);
	float total_power = 0.0f;

	AABB bounds;

	float3 vmf_axis = make_float3(0.0f, 0.0f, 0.0f);
	float vmf_sharpness = 0.0f;

	float bounding_sphere_radius = 0.0f;

	unsigned int left_child_index;
	unsigned int first_triangle_index;
	unsigned int triangle_count;
};

#endif
