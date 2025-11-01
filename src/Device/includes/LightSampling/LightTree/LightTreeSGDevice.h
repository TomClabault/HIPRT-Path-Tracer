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

struct LightTreeSGNodeDevice
{
	float3 vmf_axis = make_float3(0.0f, 0.0f, 0.0f);
	float vmf_sharpness = 0.0f;

	float3 gaussian_spatial_mean = make_float3(0.0f, 0.0f, 0.0f);
	float gaussian_spatial_variance = 0.0f;

	ColorRGB32F total_emission = ColorRGB32F(0.0f, 0.0f, 0.0f);
	float bounding_sphere_radius = 0.0f;

	float3 bounds_min;
	float total_power = 0.0f;

	float3 bounds_max;
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
