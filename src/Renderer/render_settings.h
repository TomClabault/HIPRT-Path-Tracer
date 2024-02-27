#ifndef RENDER_SETTINGS_H
#define RENDER_SETTINGS_H

struct RenderSettings
{
	// How many samples we've rendered so far
	int sample_number = 0;

	// Whether or not to keep the same resolution on
	// viewport rescale. This means that the render resolution
	// scale will be automatically adjusted
	bool keep_same_resolution = false;
	// When keep_same_resolution = true, we're going to automatically 
	// adjust the resolution scaling so that the viewport_width * resolution_scaling
	// and viewport_height * resolution_scaling = target_width and target_height
	// respectively. The values of target_width and target_height are set when the
	// user ticks the 'keep same resolution' checkbox in ImGui
	int target_width, target_height;

	float render_resolution_scale = 1.0f;

	int samples_per_frame = 1;
	int nb_bounces = 8;

	bool enable_denoising = false; // TOOD move to application_settings
	bool denoise_every_frame = false;
	// How many frames to wait for before denoising
	// (this basically reduces the performance penalty
	// of denoising each frame). Only taken into account
	// if denoise_every_frame = false
	int denoise_frame_count = 30;
	float denoising_strength = 0.5f;
};

#endif