#ifndef RENDER_SETTINGS_H
#define RENDER_SETTINGS_H

struct RenderSettings
{
	int frame_number = 0;

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
};

#endif