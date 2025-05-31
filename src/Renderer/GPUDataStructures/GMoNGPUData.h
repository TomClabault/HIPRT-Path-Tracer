/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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

	bool is_freed() const
	{
		return sets.size() == 0 && result_framebuffer->size() == 0;
	}

	ColorRGB32F* map_result_framebuffer()
	{
		if (using_gmon)
			return result_framebuffer->map();
		else
			return nullptr;
	}

	unsigned int get_VRAM_usage_bytes() const
	{
		unsigned int nb_pixels = current_resolution.x * current_resolution.y;

		unsigned int bytes_result_framebuffer = nb_pixels * sizeof(ColorRGB32F);
		unsigned int bytes_sets = nb_pixels * sizeof(ColorRGB32F) * current_number_of_sets;

		return bytes_result_framebuffer + bytes_sets;
	}

	// This is one very big buffer that contains all the sets we accumulate into for GMoN
	//
	// For example, if GMoNMSets == 5 and a render resolution of 1280x720,
	// this is going to be a buffer that is 1280*720*5 elements long
	OrochiBuffer<ColorRGB32F> sets;

	// This is the buffer that contains the G-median of means result of each pixel and this is going
	// to be displayed in the viewport instead of the regular framebuffer if GMoN is being used
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> result_framebuffer = nullptr;

	// These two variables are used for lazy GMoN recomputation:
	// 
	// GMoN isn't recomputed at each sample because we need at least one new sample
	// in each set of GMoN to be able to recompute the median of means.
	// So we should recompute GMoN only every M samples (for M GMoN sets).
	//
	// Even then, that's not exactly what we're doing because recomputing GMoN
	// is a little bit expensive but the viewport of the render window is only
	// refreshed every 5s (the timer varies) so this means that we only need to
	// recompute GMoN every 5s, not every M samples
	//
	// GMoNRenderPass:request_refresh() sets 'm_gmon_recomputation_requested' to true.
	// If 
	bool m_gmon_recomputed = false;
	bool m_gmon_recomputation_requested = false;

	// How many samples were we at when last launched the GMoN kernel
	unsigned int last_recomputed_sample_count = 0;
};

#endif
