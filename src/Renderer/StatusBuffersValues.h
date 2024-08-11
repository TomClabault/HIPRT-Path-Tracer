/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef STATUS_BUFFERS_VALUES_H
#define STATUS_BUFFERS_VALUES_H

struct StatusBuffersValues
{
	// Is there at least one pixel that is still active
	// (i.e. not all pixels have converged yet)
	// Initializing to true. Otherwise, the first call to rendering_done()
	// will return true and we will never get past the first frame
	bool one_ray_active = true;

	// How many pixels have converged in the image
	// (according to the adaptive sampling or the
	// pixel noise threshold for example)
	unsigned int pixel_converged_count = 0;
};

#endif
