/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/LightTreeSGSettings.h"
#include "Renderer/LightTree/LightTreeATSConstants.h"

struct SGLobe
{
	float3 axis;
	float  sharpness;
	float  logAmplitude;
};

struct SGLight
{
	float3 position;
	float  variance;
	float3 axis;
	float power;
	float  sharpness;
};

struct LightTreeSGNodeDevice
{
	HIPRT_DEVICE SGLight to_spherical_gaussian_light() const
	{
		SGLight sg_light;
		sg_light.position = gaussian_spatial_mean;
		sg_light.variance = gaussian_spatial_variance;
		sg_light.axis = -vmf_axis;
		sg_light.sharpness = vmf_sharpness;
		sg_light.power = total_power;
		return sg_light;
	}

	float3 vmf_axis = make_float3(0.0f, 0.0f, 0.0f);
	float vmf_sharpness = 0.0f;

	float3 gaussian_spatial_mean = make_float3(0.0f, 0.0f, 0.0f);
	float gaussian_spatial_variance = 0.0f;

	float total_power = 0.0f;

	// If triangle count is 0, this contains the left child index
	// If triangle count is > 0, this is the first triangle index in the leaf node
	unsigned int left_child_index_or_first_triangle_index;

	unsigned int triangle_count;
};

struct LightTreeSGDevice
{
	LightTreeSGSettings settings;

	LightTreeSGNodeDevice* nodes = nullptr;

	int* indices_array = nullptr;
	unsigned int* bit_trails = nullptr;
};

#endif
