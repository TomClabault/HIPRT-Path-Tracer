/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
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
			vmf.axis	  = make_float3(0.0f, 1.0f, 0.0f);
			vmf.sharpness = 0.0f;

			return;
		}
		else
			vmf.axis = mean_axis / mean_axis_length;

		if (mean_axis_length > 0.999999f)
			// Set sharpness to max value to avoid division by zero
			vmf.sharpness = 2199023255552.0f;
		else
			vmf.sharpness = hippt::min((3.0f * mean_axis_length - hippt::pow_3(mean_axis_length)) / (1.0f - hippt::square(mean_axis_length)), 2199023255552.0f);
	}

	float3_t get_float_spatial_mean() const
	{
		return make_float3(static_cast<float>(spatial_mean.x), static_cast<float>(spatial_mean.y), static_cast<float>(spatial_mean.z));
	}

	// For adaptive splitting
	float energy_average			 = 0.0f;
	float energy_variance			 = 0.0f;
	unsigned int total_emitter_count = 0;

	// Power-weighted mean and variance of the positions of the emitters of the node, in double precision: the parents accumulate them from
	// their children up to the root
	double3_t spatial_mean	= make_double3(0.0, 0.0, 0.0);
	double spatial_variance = 0.0;

	float3_t mean_axis = make_float3(0.0f, 0.0f, 0.0f);
	double total_power = 0.0;

	float3_t orientation_axis = make_float3(0.0f, 0.0f, 0.0f);
	// Orientation cone angle
	float theta_o = 0.0f;

	AABB bounds;

	VMF vmf;

	float bounding_sphere_radius = 0.0f;

	unsigned int left_child_index;
	unsigned int first_triangle_index;
	unsigned int triangle_count;
};

#endif // #ifndef RENDERER_LIGHT_TREE_SG_NODE_H
