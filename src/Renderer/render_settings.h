/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDER_SETTINGS_H
#define RENDER_SETTINGS_H

// Settings/variables that will be used by the path tracing shader 
// but that are irrelevant to the physical appearance of the world
struct RenderSettings
{
	// How many frames we've renderer so far.
	// This is different from sample_number because we
	// may be rendering more than 1 sample per frame so we could be
	// at sample 100 while only at frame 2 because we're rendering
	// 50 sampels per frame
	int frame_number = 0;

	// How many samples we've rendered so far
	int sample_number = 0;

	int samples_per_frame = 1;
	int nb_bounces = 8;

	// Is true, this means that the user is moving the camera and we're going to
	// render the image at a much lower resolution to allow for smooth camera
	// movements
	bool render_low_resolution = false;
};

#endif