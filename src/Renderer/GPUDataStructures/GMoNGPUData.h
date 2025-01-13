/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GMON_GPU_DATA_H
#define RENDERER_GMON_GPU_DATA_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/Color.h"
#include "OpenGL/OpenGLInteropBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/GMoNCPUGPUCommonData.h"

#include <memory>
#include <vector>

/**
 * CPU-side data structure for the implementation of GMoN on the GPU
 *
 * Reference:
 * [1] [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]
 */
struct GMoNGPUData : public GMoNCPUGPUCommonData
{
	GMoNGPUData()
	{
		result_framebuffer = std::make_shared<OpenGLInteropBuffer<ColorRGB32F>>();
	}

	void resize_sets(unsigned int render_width, unsigned int render_height, unsigned int number_of_sets)
	{
		sets.resize(render_width * render_height * number_of_sets);
		sets.memset_whole_buffer(0);

		current_resolution = make_int2(render_width, render_height);
		current_number_of_sets = number_of_sets;
	}

	void resize_interop(unsigned int new_width, unsigned int new_height)
	{
		result_framebuffer->resize(new_width * new_height);
	}

	void free()
	{
		sets.free();
		result_framebuffer->free();

		current_resolution = make_int2(0, 0);
	}

	ColorRGB32F* map_result_framebuffer()
	{
		if (use_gmon)
			return result_framebuffer->map();
		else
			return nullptr;
	}

	// This is one very big buffer that contains all the sets we accumulate into for GMoN
	//
	// For example, if GMoNMSets == 5 and a render resolution of 1280x720,
	// this is going to be a buffer that is 1280*720*5 elements long
	OrochiBuffer<ColorRGB32F> sets;

	// This is the buffer that contains the G-median of means result of each pixel and this is going
	// to be displayed in the viewport instead of the regular framebuffer if GMoN is being used
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> result_framebuffer = nullptr;
};

#endif
