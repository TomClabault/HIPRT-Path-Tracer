/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_NODE_H
#define RENDERER_LIGHT_TREE_SG_NODE_H

#include "HostDeviceCommon/Maths/Math.h"

struct LightTreeSGSpatialLobeBuild
{
	double power	= 0.0;
	double mean_x	= 0.0;
	double mean_y	= 0.0;
	double mean_z	= 0.0;
	double variance = 0.0;
	AABB bounds;
};

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

	// For adaptive splitting
	float energy_average			 = 0.0f;
	float energy_variance			 = 0.0f;
	unsigned int total_emitter_count = 0;

	LightTreeSGSpatialLobeBuild spatial_lobes[2];
	float3_t spatial_mean = make_float3(0.0f, 0.0f, 0.0f);

	float3_t mean_axis = make_float3(0.0f, 0.0f, 0.0f);
	float total_power  = 0.0f;

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

#endif
