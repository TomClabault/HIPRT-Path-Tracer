/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_SETTINGS_H
#define HOST_DEVICE_COMMON_RENDER_SETTINGS_H

#include "HostDeviceCommon/ReSTIRDISettings.h"

#include <hiprt/hiprt_common.h>

struct RISSettings
{
	// Whether or not to use a geometry term in the target function when resampling
	// light samples
	bool geometry_term_in_target_function = false;
	// How many candidate lights to sample for RIS (Resampled Importance Sampling)
	int number_of_light_candidates = 4;
	// How many candidates samples from the BSDF to use in combination
	// with the light candidates for RIS
	int number_of_bsdf_candidates = 1;
};

struct HIPRTRenderSettings
{
	// If true, the camera ray kernel will reset all buffers to their default values.
	// This is mainly useful for the first frame of the render
	bool need_to_reset = true;

	// If true, then the kernels are allowed to modify the status buffers (how many pixels have converged so far, ...)
	// 
	// Why is this useful?
	// There is a "status" buffer that contains the number of pixels that have converged for a kernel launch.
	// It is a simple counter that threads of the kernel increment if the pixel corresponding to the thread has converged.
	// Because thread keep incrementing this counter, we need to reset it to 0 before each kernel launch.
	// 
	// To simulate multiple samples per frame and reduce CPU overhead, we can launch multiple times the kernels per frame.
	// We would thus need to reset the status buffer before each kernel launch but this is a synchronous operation which then
	// slows down the UI. This means that we cannot reset the status buffer before each kernel launch, we can only reset it
	// at each frame before GPURenderer::render() is called.
	// 
	// In the case where we have 5 samples per pixel for example, we would have each kernel launch increment the status
	// buffer and that would largely go above 100% of pixels converged (which doesn't make sense). 
	// What we do instead is that we only allow the last kernel launch of the frame to increment the status buffers.
	//
	// This is the variable that enables / disables the increment of status buffers
	bool do_update_status_buffers = false;

	// Whether or not to accumulate each frame to allow progressive rendering. If false,
	// each frame will be displayed on screen and discarded on the next frame without accumulation
	bool accumulate = false;

	// How many times the render kernel was called (updates after
	// the call to the kernel so it start at 0)
	int frame_number = 0;

	// Number of samples rendered so far before the kernel call
	// This is the sum of samples_per_frame for all frames
	// that have been rendered.
	int sample_number = 0;

	int samples_per_frame = 1;
	// Maximum number of bounces of rays in the scene. 
	// 1 is direct light only.
	int nb_bounces = 1;

	// Whether or not to "freeze" random number generation so that each frame uses
	// exactly the same random number. This allows every ray to follow the exact
	// same path every frame, allowing for more stable benchmarking.
	int freeze_random = false;

	// If true, NaNs encountered during rendering will be rendered as very bright pink. 
	// Useful for debugging only.
	bool display_NaNs = false;

	// If true, then rendering at low resolution will be performed if 'wants_render_low_resolution'
	// is also true.
	// This boolean basically is an additional condition for rendering at low resolution:
	//	 - If we're interacting with the camera, we *want* to render at low resolution
	//	 but if rendering at low resolution is not allowed (this boolean), then we will still
	//	 not render at low resolution
	bool allow_render_low_resolution = false;
	// If true, this means that the user is moving the camera and we want to
	// render the image at a much lower resolution to allow for smoother
	// interaction. Having this flag at true isn't sufficient for rendering at low
	// resolution. The user must also *allow* rendering at low resolution
	// with the 'allow_render_low_resolution' flag
	int wants_render_low_resolution = false;
	// How to divide the render resolution by when rendering at low resolution
	// (when interacting with the camera)
	int render_low_resolution_scaling = 4;

	int enable_adaptive_sampling = false;
	// How many samples before the adaptive sampling actually kicks in.
	// This is useful mainly for the per-pixel adaptive sampling method
	// where you want to be sure that each pixel in the image has had enough
	// chance find a path to a potentially 
	int adaptive_sampling_min_samples = 96;
	// Adaptive sampling noise threshold
	float adaptive_sampling_noise_threshold = 0.4f;

	// A percentage in [0, 100] that dictates the proportion of pixels that must
	// have reached the given noise threshold (stop_pixel_noise_threshold
	// variable) before we stop rendering.
	// For example, if this variable is 90, we will stop rendering when 90% of all
	// pixels have reached the stop_pixel_noise_threshold
	float stop_pixel_percentage_converged = 90.0f;
	// Noise threshold for use with the stop_pixel_percentage_converged stopping
	// condition
	float stop_pixel_noise_threshold = 0.0f;



	// Clamp direct lighting contribution to reduce fireflies
	float direct_contribution_clamp = 0.0f;
	// Clamp envmap contribution to reduce fireflies
	float envmap_contribution_clamp = 0.0f;
	// Clamp indirect lighting contribution to reduce fireflies
	float indirect_contribution_clamp = 0.0f;

	// Settings for RIS (direct light sampling)
	RISSettings ris_settings;

	// Settings for ReSTIR DI
	ReSTIRDISettings restir_di_settings;

	/**
	 * Returns true if the current frame should be renderer at low resolution, false otherwise.
	 * 
	 * This function is a simple helper that combines 
	 */
	HIPRT_HOST_DEVICE bool do_render_low_resolution() const
	{
		return wants_render_low_resolution && allow_render_low_resolution;
	}

	/**
	 * Returns true if the adaptive sampling buffers are ready for use, false otherwise.
	 *
	 * Adaptive sampling buffers are "ready for use" if the adaptive sampling is enabled or
	 * if the pixel stop noise threshold is enabled. Otherwise, the adaptive sampling buffers
	 * are freed to save VRAM so they cannot be used.
	 */
	HIPRT_HOST_DEVICE bool has_access_to_adaptive_sampling_buffers() const
	{
		bool has_access = false;

		has_access |= stop_pixel_noise_threshold > 0.0f;
		has_access |= enable_adaptive_sampling == 1;
		// Cannot use adaptive sampling without accumulation
		has_access &= !accumulate;

		return has_access;
	}
};

#endif
