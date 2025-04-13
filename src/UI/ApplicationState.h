/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef APPLICATION_STATE_H
#define APPLICATION_STATE_H

struct ApplicationState
{
	float last_CPU_frame_delta_time_ms = 0.0f;
	// GLFW timestamp of when was the last time that we submitted a frame to render to the GPU.
	uint64_t last_GPU_submit_time = 0;
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

	// How many times renderer->render() was called since the last dirty frame.
	int frame_number = 0;

	// If true, the viewport is going to be refreshed next frame no matter what
	bool force_viewport_refresh = false;
	// How long has passed since the last time we "uploaded" the renderer
	// frame buffer to OpenGL for display.
	// 
	// This variable is used to minimize how often we upload to OpenGL because
	// all of that is expensive.
	//
	// This is only used for offline rendering and this will have the effect
	// of updating the viewport only once every few seconds to save resources
	uint64_t last_viewport_refresh_timestamp = 0;
};

#endif