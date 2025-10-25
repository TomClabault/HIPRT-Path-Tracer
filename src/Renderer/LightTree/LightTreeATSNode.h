/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_ATS_NODE_H
#define RENDERER_LIGHT_TREE_ATS_NODE_H

#include "Device/includes/ONB.h"
#include "HostDeviceCommon/Color.h"
#include "Renderer/LightTree/LightTreeATSNodeOrientationData.h"
#include "Scene/AABB.h"

struct LightTreeATSNode
{
	void cone_union_with(float3 other_axis, float other_theta_o, float other_theta_e)
	{
		orientation_data.cone_union_with(other_axis, other_theta_o, other_theta_e);
	}

	LightTreeATSNodeOrientationData orientation_data;

	// For adaptive splitting
	float energy_average = 0.0f;
	float energy_variance = 0.0f;
	unsigned int total_emitter_count = 0;

	// Total emissive power of the node
	ColorRGB32F total_power;

	AABB node_bounds;
	unsigned int left_child_index;
	unsigned int first_triangle_index, triangle_count;
	unsigned int bit_trail = 0;
};

#endif
