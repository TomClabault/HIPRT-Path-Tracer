/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_DEVICE_H
#define DEVICE_INCLUDES_LIGHT_TREE_DEVICE_H

#include "HostDeviceCommon/Color.h"

struct LightTreeNodeDevice
{
	// Axis of the cluster
	float3 axis = make_float3(0.0f, 0.0f, 0.0f);
	// Normal bounds
	float theta_o;
	// Emission extents
	float theta_e;
	// Total emissive powxer of the node
	ColorRGB32F total_power;

	float3 bounds_min;
	float3 bounds_max;

	unsigned int left_child_index;
	unsigned int first_triangle_index, triangle_count;
};

struct LightTreeDevice
{
	LightTreeNodeDevice* nodes = nullptr;
	int* indices_array = nullptr;
};

#endif
