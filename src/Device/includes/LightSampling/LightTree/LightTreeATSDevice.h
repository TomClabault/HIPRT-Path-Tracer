/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_ATS_DEVICE_H
#define DEVICE_INCLUDES_LIGHT_TREE_ATS_DEVICE_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/LightTreeATSSettings.h"
#include "Renderer/LightTree/LightTreeATSConstants.h"

struct LightTreeATSNodeDevice
{
	HIPRT_DEVICE bool is_invalid() const
	{
		return axis.x == LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS;
	}

	HIPRT_DEVICE float get_energy_average() const
	{
		return total_power_luminance / total_emitter_count;
	}

	float cos_theta_o;
	float sin_theta_o;
	float total_power_luminance;
	float energy_variance;

	// Axis of the cluster
	float3_t axis					 = make_float3(0.0f, 0.0f, 0.0f);
	unsigned int total_emitter_count = 0;

	float3_t bounds_min = make_float3(0.0f, 0.0f, 0.0f);
	unsigned int left_child_index_or_first_triangle_index;

	float3_t bounds_max = make_float3(0.0f, 0.0f, 0.0f);
	// If triangle count is 0, this contains the left child index
	// If triangle count is > 0, this is the first triangle index in the leaf node
	unsigned int triangle_count;
};

struct LightTreeATSDevice
{
	LightTreeATSSettings settings;

	LightTreeATSNodeDevice* nodes = nullptr;
	int* indices_array			  = nullptr;
	unsigned int* bit_trails	  = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_LIGHT_TREE_ATS_DEVICE_H
