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


















	// Whether or not to keep the same resolution on
	// viewport rescale. This means that the render resolution
	// scale will be automatically adjusted
	bool keep_same_resolution = false; // TOOD move to application_settings

	// When keep_same_resolution = true, we're going to automatically 
	// adjust the resolution scaling so that the viewport_width * resolution_scaling
	// and viewport_height * resolution_scaling = target_width and target_height
	// respectively. The values of target_width and target_height are set when the
	// user ticks the 'keep same resolution' checkbox in ImGui
	int target_width, target_height; // TOOD move to application_settings


	bool enable_denoising = false; // TOOD move to application_settings

	// How many frames to wait for before denoising (this basically reduces 
	// the performance penalty of denoising each frame).
	int denoiser_sample_skip = 1; // TOOD move to application_settings
	int denoise_frame_count = 30; // TOOD move to application_settings
};

#endif