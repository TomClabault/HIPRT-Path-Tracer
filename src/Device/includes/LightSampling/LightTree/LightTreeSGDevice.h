/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/LightTreeSGSettings.h"
#include "Renderer/LightTree/LightTreeATSConstants.h"

struct LightTreeSGNodeDevice
{
	
};

struct LightTreeSGDevice
{
	LightTreeSGSettings settings;

	LightTreeSGNodeDevice* nodes = nullptr;

	int* indices_array = nullptr;
	unsigned int* bit_trails = nullptr;
};

#endif
