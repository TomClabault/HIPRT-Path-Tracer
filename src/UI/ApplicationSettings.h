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
	// What view is currently displayed in the viewport
	DisplayView display_view = DisplayView::DEFAULT;

	bool enable_denoising = false;
	// How many samples were denoised by the last denoiser call
	bool denoiser_use_albedo = true;
	bool denoiser_denoise_albedo = true;
	bool denoiser_use_normals = true;
	bool denoiser_denoise_normals = true;
	int last_denoised_sample_count = -1;
	// Denoise only when that maximum sample count is reached
	bool denoise_at_max_samples = false;
	// How many frames to wait for before denoising (this basically reduces 
	// the performance penalty of denoising each frame).
	int denoiser_sample_skip = 1;
	// If 1.0f, 100% of the denoised result is displayed in the viewport.
	// If 0.0f, 100% of the noisy framebuffer is displayed in the viewport
	// Linearly interpoalted between the two for intermediate values
	float denoiser_blend = 1.0f;
	// If the denoiser settings changed since last frame
	bool denoiser_settings_changed = false;

	// How much to divide the translation distance by when the mouse
	// has been dragged over the window to move the camera
	// This is necessary because if 1 pixel of movement equalled
	// 1 world unit of translation, it would be way too fast!
	double view_translation_sldwn_x = 300.0f, view_translation_sldwn_y = 300.0f;
	double view_rotation_sldwn_x = 3.5f, view_rotation_sldwn_y = 3.5f;
	double view_zoom_sldwn = 5.0f;

	// How much to scale the render resolution by. 
	// For example, if == 2, and the viewport currently is 1280*720, 
	// the path tracer will compute a 2560*1440 image and display it
	// in the 1280*720 viewport
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
	int target_width = 0, target_height = 0;

	// We stop rendering when this number of sample is reached.
	// 0 is no limit
	int max_sample_count = 0;
	// We stop rendering when the render has been running for that long.
	// In seconds. 0 is no limit
	float max_render_time = 0.0f;

	// if true, the number of samples per frame will be adjusted automatically to target 20 FPS. 
	// This is meant to keep the GPU busy mostly when adaptive sampling is on.
	// This is because with adaptive sampling on, FPS will keep increasing as the number of
	// pixels that yet have to converge decreases. And with high FPS count, we get the risk
	// of being CPU bound since we'll have to display many frames per second.
	bool auto_sample_per_frame = true;

	// Whether or not to do tonemapping for display fragment shader that support it
	bool do_tonemapping = 1;
	// Tone mapping gamma
	float tone_mapping_gamma = 2.2f;
	// Tone mapping exposure
	float tone_mapping_exposure = 1.8f;
};

#endif