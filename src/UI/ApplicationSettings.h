/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef APPLICATION_SETTINGS_H
#define APPLICATION_SETTINGS_H

#include <string>
#include <vector>

#include "UI/DisplayViewEnum.h"

struct ApplicationSettings
{
	bool enable_denoising = false;
	bool denoiser_use_albedo = true;
	bool denoiser_denoise_albedo = true;
	bool denoiser_use_normals = true;
	bool denoiser_denoise_normals = true;
	// How many samples were we at when we last denoised a frame
	int last_denoised_sample_count = -1;
	// How many microseconds did it take to denoise (last time we denoised)?
	float last_denoised_duration = 0.0f;
	// Denoise only when that maximum sample count is reached
	bool denoise_when_rendering_done = true;
	// How many frames to wait for before denoising (this basically reduces 
	// the performance penalty of denoising each frame).
	int denoiser_sample_skip = 1;
	// If 1.0f, 100% of the denoised result is displayed in the viewport.
	// If 0.0f, 100% of the noisy framebuffer is displayed in the viewport
	// Linearly interpoalted between the two for intermediate values
	float denoiser_blend = 1.0f;
	// Overrides the blending factor for the blend-2-textures display shader
	// 0.0f displays 100% of texture 1.
	// 1.0f gives 100% of texture 2.
	// -1.0f disables the override
	float blend_override = -1.0f;
	// If the denoiser settings changed since last frame
	bool denoiser_settings_changed = false;

	// How much to divide the rotation by when the mouse
	// has been dragged over the window to move the camera
	// This is necessary because if 1 pixel of movement equalled
	// 1 degree of rotation, it would be way too fast!
	double view_rotation_sldwn_x = 3.5f, view_rotation_sldwn_y = 3.5f;

	// How much to scale the render resolution by. 
	// For example, if == 2, and the viewport currently is 1280*720, 
	// the path tracer will compute a 2560*1440 image and display it
	// in the 1280*720 viewport
	float render_resolution_scale = 1.0f;
	// This variable is meant to keep the GPU busy when using "automatic number of samples"
	// per frame. The idea is to adjust the number of samples per frame such that the GPU
	// always has a bunch of work to do.
	// For example, let's say that after a while, the adaptive sampling has judged that only
	// 1000 pixels are left to converge out of the ~2M of 1080p image. 1000 pixels to ray trace
	// is a joke for the GPU. It's going to be extremely fast. So fast that the application is
	// going to be CPU bound for displaying the image etc...
	// To avoid being CPU bound, we adjust the work of the GPU such that it still has a significant
	// amount of work to process.
	// The amount of work is adjusted by adjusting the number of samples per frame. We adjust the
	// samples per frame such that the GPU takes (1000ms / target_GPU_framerate) milliseconds
	// to compute a frame
	float target_GPU_framerate = 5.0f;
	// If > 0.0f, stalls the GPU for a certain amount of time (based on the percentage and the
	// time taken to render the last frame). This feature is only there to help limit GPU heating
	// at the cost of longer render times
	float GPU_stall_percentage = 0.0f;

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