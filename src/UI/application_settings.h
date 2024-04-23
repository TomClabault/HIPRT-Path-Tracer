/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef APPLICATION_SETTINGS_H
#define APPLICATION_SETTINGS_H

#include <string>
#include <vector>

#include "UI/DisplayView.h"

struct ApplicationSettings
{
	int selected_kernel = 0;
	std::vector<std::string> kernel_files = { DEVICE_KERNELS_DIRECTORY "/path_tracer_kernel.h", DEVICE_KERNELS_DIRECTORY "/normals_kernel.h" };
	std::vector<std::string> kernel_functions = { "PathTracerKernel", "NormalsKernel"};

	DisplayView display_view = DisplayView::DEFAULT;

	bool enable_denoising = false;
	// How many samples were denoised by the last denoiser call
	int last_denoised_sample_count;
	// Denoise only when that maximum sample count is reached
	bool denoise_at_target_sample_count = false;

	// How many frames to wait for before denoising (this basically reduces 
	// the performance penalty of denoising each frame).
	int denoiser_sample_skip = 1;

	// How much to divide the translation distance by when the mouse
	// has been dragged over the window to move the camera
	// This is necessary because if 1 pixel of movement equalled
	// 1 world unit of translation, it would be way too fast!
	double view_translation_sldwn_x = 300.0f, view_translation_sldwn_y = 300.0f;
	double view_rotation_sldwn_x = 3.5f, view_rotation_sldwn_y = 3.5f;
	double view_zoom_sldwn = 5.0f;

	float render_resolution_scale = 1.0f;

	// Whether or not to keep the same resolution on
	// viewport rescale. This means that the render resolution
	// scale will be automatically adjusted
	// This is useful if you want a bigger window on your desktop
	// without having the resolution going up and your GPU kneeling in pain
	bool keep_same_resolution = false;

	// When keep_same_resolution = true, we're going to automatically 
	// adjust the resolution scaling so that the viewport_width * resolution_scaling
	// and viewport_height * resolution_scaling = target_width and target_height
	// respectively. The values of target_width and target_height are set when the
	// user ticks the 'keep same resolution' checkbox in ImGui
	int target_width, target_height;

	// We stop rendering when this number of sample is reached
	int max_sample_count = 0;

	bool do_tonemapping = 1;
	float tone_mapping_gamma = 2.2f;
	float tone_mapping_exposure = 1.0f;
};

#endif