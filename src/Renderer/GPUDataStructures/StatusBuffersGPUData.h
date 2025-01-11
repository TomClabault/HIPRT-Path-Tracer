/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HIPRT-Orochi/OrochiBuffer.h"

struct StatusBuffersGPUData
{
	// A single boolean to indicate whether there is still a ray active in
	// the kernel or not. Mostly useful when adaptive sampling is on and we
	// want to know if all pixels have converged or not yet
	OrochiBuffer<unsigned char> still_one_ray_active_buffer;
	// How many pixels have reached the render_settings.stop_pixel_noise_threshold.
	// Warning: This buffer does not count how many pixels have converged according to
	// the adaptive sampling noise threshold. This is only for the stop_pixel_noise_threshold
	OrochiBuffer<unsigned int> pixels_converged_count_buffer;
};