/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GMON_CPU_DATA_H
#define RENDERER_GMON_CPU_DATA_H

#include "Renderer/CPUGPUCommonDataStructures/GMoNCPUGPUCommonData.h"

/**
 * CPU-side data structure for the implementation of GMoN on the CPU
 *
 * Reference:
 * [1] [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]
 */
struct GMoNCPUData : public GMoNCPUGPUCommonData
{
	void resize(unsigned int render_width, unsigned int render_height)
	{
		sets.resize(render_width * render_height * number_of_sets);

		result_framebuffer = Image32Bit(render_width, render_height, /* channels */ 3);
	}

	// This is one very big buffer that contains all the sets we accumulate into for GMoN
	//
	// For example, if GMoNMSets == 5 and a render resolution of 1280x720,
	// this is going to be a buffer that is 1280*720*5 elements long
	std::vector<ColorRGB32F> sets;

	// This is the buffer that contains the G-median of means result of each pixel and this is going
	// to be displayed in the viewport instead of the regular framebuffer if GMoN is being used
	Image32Bit result_framebuffer;

	unsigned int number_of_sets = GMoNMSetsCount;
};

#endif
