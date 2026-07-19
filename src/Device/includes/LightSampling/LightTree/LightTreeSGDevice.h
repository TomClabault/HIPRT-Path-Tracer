/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H

#include "Device/includes/PathGuiding/VMF.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/LightTreeSGSettings.h"
#include "Renderer/LightTree/LightTreeATSConstants.h"

struct SpatialSGLobeDevice
{
	float3_t mean;
	float variance		 = 0.0f;
	float power			 = 0.0f;
	float support_radius = 0.0f;
};

struct LightTreeSGNodeDevice
{
	HIPRT_DEVICE float get_energy_average() const
	{
		// Returns the raw (non-SG-divided) average power, matching the scale of energy_variance
		return energy_average;
	}

	// Note that this VMF is shared for all lobes of the node. One VMF distribution per each lobe yielded a bit better quality when I tried but the increased
	// size of the nodes made fetching nodes during traversal more expensive and it wasn't worth it overall. So this is instead an "average VMF" fitted over the
	// whole node.
	VMF vmf;

	// Points to this node's contiguous spatial SG lobe range in LightTreeSGDevice storage.
	SpatialSGLobeDevice* spatial_lobes = nullptr;
	unsigned int spatial_lobe_count	   = 1;

	float3_t gaussian_spatial_mean = make_float3(0.0f, 0.0f, 0.0f);

	float3_t orientation_axis = make_float3(0.0f, 0.0f, 0.0f);
	// Orientation cone angle
	float cos_theta_o = 0.0f;
	float sin_theta_o = 0.0f;

	float bounding_sphere_radius = 0.0f;

	// This contains a baked in division by SG_integral(node.vmf.sharpness)
	float total_power				 = 0.0f;
	float energy_variance			 = 0.0f;
	float energy_average			 = 0.0f;
	unsigned int total_emitter_count = 0;

	// If triangle count is 0, this contains the left child index
	// If triangle count is > 0, this is the first triangle index in the leaf node
	unsigned int left_child_index_or_first_triangle_index;
	unsigned int triangle_count;

	float3_t bounds_min;
	float3_t bounds_max;
};

struct LightTreeSGDevice
{
	LightTreeSGSettings settings;

	LightTreeSGNodeDevice* nodes	   = nullptr;
	SpatialSGLobeDevice* spatial_lobes = nullptr;

	int* indices_array		 = nullptr;
	unsigned int* bit_trails = nullptr;
};

#endif
