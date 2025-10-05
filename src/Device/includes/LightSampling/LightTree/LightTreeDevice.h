/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_DEVICE_H
#define DEVICE_INCLUDES_LIGHT_TREE_DEVICE_H

#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/TriangleSampling.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

struct LightTreeNodeDevice
{
	float3 point;
	ColorRGB32F total_power;

	unsigned int left_child_index;
	unsigned int first_triangle_index, triangle_count;
};

struct LightTreeDevice
{
	LightTreeNodeDevice* nodes = nullptr;
};

#endif
