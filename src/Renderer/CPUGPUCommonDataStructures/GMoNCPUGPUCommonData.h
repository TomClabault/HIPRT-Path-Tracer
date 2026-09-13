/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GMON_CPU_GPU_COMMON_DATA_H
#define RENDERER_GMON_CPU_GPU_COMMON_DATA_H

#include "HostDeviceCommon/KernelOptions/GMoNOptions.h"

struct GMoNCPUGPUCommonData
{
	// How much to blend between the non-GMoN output and the GMoN output
	float gmon_blend_factor		= 0.0f;
	bool gmon_auto_blend_factor = true;

	// A zero resolution marks the buffers as not allocated yet. The renderer's default resolution must not
	// be used as an allocation sentinel because GMoN can be enabled at that resolution before any resize event.
	int2_t current_resolution			= make_int2(0, 0);
	unsigned int current_number_of_sets = 0;
};

#endif // #ifndef RENDERER_GMON_CPU_GPU_COMMON_DATA_H
