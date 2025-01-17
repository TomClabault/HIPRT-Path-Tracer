/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GMON_CPU_GPU_COMMON_DATA_H
#define RENDERER_GMON_CPU_GPU_COMMON_DATA_H

#include "HostDeviceCommon/KernelOptions/GMoNOptions.h"

struct GMoNCPUGPUCommonData
{
	// Whether or not GMoN is actively being used
	bool use_gmon = false;

	// How much to blend between the non-GMoN output and the GMoN output
	float gmon_blend_factor = 0.0f;
	bool gmon_auto_blend_factor = true;

	int2 current_resolution = make_int2(1280, 720);
	unsigned int current_number_of_sets = GMoNMSetsCount;
};

#endif
