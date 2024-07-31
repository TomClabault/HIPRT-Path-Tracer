/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef APPLICATION_STATE_H
#define APPLICATION_STATE_H

struct ApplicationState
{
	float last_delta_time_ms = 0.0f;
	// How long the current render has been running for in milliseconds
	float current_render_time_ms = 0.0f;
	// Samples per second (computed at each frame based on the number of
	// samples per frame and the time to render the last frame)
	float samples_per_second = 0.0f;
	// Set to true if some settings of the render changed and we need
	// to restart rendering from sample 0
	bool render_dirty = true;
	// If true, this means that the user was interacting with the camera
	// at last frame
	bool interacting_last_frame = false;

	// How long in milliseconds do we still have to stall the GPU for
	float GPU_stall_duration_left = 0;
};

#endif