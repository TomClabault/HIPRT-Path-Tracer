/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_SETTINGS_H
#define HOST_DEVICE_COMMON_RENDER_SETTINGS_H

#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"

#include "HostDeviceCommon/PathRussianRoulette.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/RIS/RISSettings.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRCommonSettings.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRDISettings.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRGISettings.h"
#include "HostDeviceCommon/Math.h"

// Just used for initializing some structure members below
#define local_min_macro(a, b) ((a) < (b) ? (a) : (b))

class GPURenderer;

struct HIPRTRenderSettings
{
	int2 render_resolution = make_int2(1280, 720);

	// If true, the camera ray kernel will reset all buffers to their default values.
	// This is mainly useful for the first frame of the render
	bool need_to_reset = true;

	// TODO DEBUG REMOVE THIS
	static constexpr float MULTIPLIER = 100000.0f;
	static constexpr int SAMPLE_STOP = 4096;

	// TODO DEBUG REMOVE THIS
	bool enable_direct = true;
	AtomicType<int>* DEBUG_SUM_COUNT = nullptr;
	AtomicType<float>* DEBUG_SUMS = nullptr;
	bool debug_lambertian = false;
	int debug_x = 138, debug_y = 151;
	int debug_x2 = 136, debug_y2 = 147;
	bool DEBUG_DO_ONLY_NEIGHBOR = false;
	int debug_size = 0;
	int debug_count_multiplier = 2;
	int precision = 256;
	int stop_value = SAMPLE_STOP * MULTIPLIER - 10;

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
	bool accumulate = true;

	// How many samples were accumulated in the denoiser's AOV buffers (albedo & normals)
	// This is used mainly for the normals AOVs because we want a way to accumulate the normals.
	// However, we still want to feed the normalized normals to the denoiser. 
	// This means that we need to store normalized normals in the normals AOV GPU buffer. 
	// But if we also want to accumulate, we also need to get the normals back from "normalized"
	// to their "accumulated" value. We can then add the normal of the first hit of our current
	// frame to that "accumulated" value and then normalize again.
	// 
	// We need denoiser_AOV_accumulation_counter to multiply the normalized normals of the buffer with
	// and get that "accumulated" normals value.
	int denoiser_AOV_accumulation_counter = 0;

	// Number of samples rendered so far before the kernel call
	// This is the sum of samples_per_frame for all frames
	// that have been rendered.
	unsigned int sample_number = 0;

	// How many samples to compute per pixel per frame
	// Higher values reduce CPU overhead since the GPU spends
	// more time computing per frame but reduces interactivity
	int samples_per_frame = 1;
	// Maximum number of bounces of rays in the scene. 
	// 1 is direct light only.
	int nb_bounces = 3000;

	bool use_russian_roulette = false;
	// After how many bounces can russian roulette kick in?
	// 0 means that the camera ray hits, and then the next bounce
	// is already susceptible to russian roulette termination
	int russian_roulette_min_depth = local_min_macro(5, nb_bounces / 2);
	// After applying russian roulette(dividing by the continuation probability)
	// the energy added to the ray throughput is clamped to this maximum value.
	// 
	// This is biased and darkens the image the lower the threshold but it helps
	// reduce variance and fireflies introduced by the russian roulette --> faster
	// convergence.
	//
	// 0 for no clamping.
	float russian_roulette_throughput_clamp = 10.0f;

	// What Russian roulette method to use to determine the path termination
	// probability
	PathRussianRoulette path_russian_roulette_method = PathRussianRoulette::MAX_THROUGHPUT;

	// Whether or not to "freeze" random number generation so that each frame uses
	// exactly the same random number. This allows every ray to follow the exact
	// same path every frame, allowing for more stable benchmarking.
	int freeze_random = false;

	// If true, NaNs encountered during rendering will be rendered as very bright pink. 
	// Useful for debugging only.
	bool display_NaNs = true;

	// If true, then rendering at low resolution will be performed if 'wants_render_low_resolution'
	// is also true.
	// This boolean basically is an additional condition for rendering at low resolution:
	//	 - If we're interacting with the camera, we *want* to render at low resolution
	//	 but if rendering at low resolution is not allowed (this boolean), then we will still
	//	 not render at low resolution
	// This boolean is controlled by the user in Imgui
	bool allow_render_low_resolution = true;
	// If true, this means that the user is moving the camera and we want to
	// render the image at a much lower resolution to allow for smoother
	// interaction. Having this flag at true isn't sufficient for rendering at low
	// resolution. The user must also *allow* rendering at low resolution
	// with the 'allow_render_low_resolution' flag
	bool wants_render_low_resolution = false;
	// How to divide the render resolution by when rendering at low resolution
	// (when interacting with the camera)
	int render_low_resolution_scaling = 2;

	bool enable_adaptive_sampling = false;
	// How many samples before the adaptive sampling actually kicks in.
	// This is useful mainly for the per-pixel adaptive sampling method
	// where you want to be sure that each pixel in the image has had enough
	// chance find a path to a potentially 
	int adaptive_sampling_min_samples = 96;
	// Adaptive sampling noise threshold
	float adaptive_sampling_noise_threshold = 0.1f;

	// If true, the rendering will stop after a certain proportion (defined by 'stop_pixel_percentage_converged')
	// of pixels of the image have converged. "converged" here is defined according to the adaptive sampling if
	// enabled or according to 'stop_pixel_noise_threshold' if adaptive sampling is not enabled.
	//
	// If false, the render will not stop until all pixels have converged
	bool enable_pixel_stop_noise_threshold = true;
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
	float direct_contribution_clamp = 4.0f;
	// Clamp envmap contribution to reduce fireflies
	float envmap_contribution_clamp = 0.0f;
	// Clamp indirect lighting contribution to reduce fireflies
	float indirect_contribution_clamp = 0.0f;

	// If a selected light (for direct lighting estimation) contributes at a given
	// point less than this 'minimum_light_contribution' value then the light sample is discarded
	// 0.0f to disable
	float minimum_light_contribution = 0.0f;

	// How many light samples to take and shade per each vertex of the
	// ray's path.
	//
	// Said otherwise, we're going to run next-event estimation that many
	// times per each intersection point along the ray.
	// 
	// This is good because this amortizes camera rays and bounce rays i.e.
	// we get better shading quality for as many camera rays and bounce rays
	//
	// This is not supported by ReSTIR DI because this would require recomputing
	// a new reservoir = full re-run of ReSTIR = too expensive.
	// It does apply to the secondary bounces shading when using ReSTIR DI for the
	// primary bounce though.
	int number_of_nee_samples = 1;

	// Whether or not to do alpha testing for geometry with transparent base color textures
	bool do_alpha_testing = true;
	// At what bounce to stop doing alpha testing
	// 
	// A value of 0 means that alpha testing isn't done at bounce 0 which means that even camera
	// rays do not do alpha testing --> alpha testing is disable
	// 
	// A value of 1 means that camera rays do alpha testing but the next bounce rays do not do alpha
	// testing
	//
	// Shadow rays for NEE are also affected by this setting
	int alpha_testing_indirect_bounce = 2;

	// Whether or not to do normal mapping at all
	// If false, geometric normals will always be used
	bool do_normal_mapping = true;

	// Settings for RIS (direct light sampling)
	RISSettings ris_settings;

	// Settings for ReSTIR DI
	ReSTIRDISettings restir_di_settings;
	// Settings for ReSTIR GI
	ReSTIRGISettings restir_gi_settings;

	/**
	 * Returns true if the current frame should be renderer at low resolution, false otherwise.
	 * 
	 * This function is a simple helper that combines a few flags to make sure that we
	 * actually want to render at low resolution
	 */
	HIPRT_HOST_DEVICE bool do_render_low_resolution() const
	{
		return wants_render_low_resolution && allow_render_low_resolution && accumulate;
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
		has_access |= enable_adaptive_sampling;
		// Cannot use adaptive sampling without accumulation
		has_access &= accumulate;

		return has_access;
	}

	/**
	 * Returns true if the renderer needs the G-buffer of the previous frame.
	 * 
	 * The boolean parameter is some additional condition that must be satisfied
	 * for the G-buffer to be needed
	 * 
	 * We need two overrides of this function: one for use in the shaders and one 
	 * for use in the C++ CPU side code.
	 * 
	 * This is because to determine whether or not we need the g-buffer of last
	 * frame, we need to check if ReSTIR DI is being used or not. On the CPP side, this
	 * can be done with the GPURenderer instance by checking the path tracer
	 * options and check if the DirectLightSamplingStrategy is equal to
	 * LSS_RESTIR_DI. On the device however, we don't have access to the
	 * GPURenderer instance but instead, we can check directly using the 
	 * DirectLightSamplingStrategy macro (and we don't want the GPURenderer parameter 
	 * because that doesn't exist on the device).
	 */
	HIPRT_DEVICE bool use_prev_frame_g_buffer() const
	{
		// If ReSTIR DI isn't used, we don't need the last frame's g-buffer
		// (as far as the codebase goes at the time of writing this function anyways)
		bool need_g_buffer = false;
		need_g_buffer |= DirectLightSamplingStrategy == LSS_RESTIR_DI && restir_di_settings.common_temporal_pass.do_temporal_reuse_pass;
		need_g_buffer |= PathSamplingStrategy == PSS_RESTIR_GI && restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass;

		return need_g_buffer;
	}

	// Only need this one on the host
#ifndef __KERNELCC__
	HIPRT_HOST bool use_prev_frame_g_buffer(GPURenderer* renderer) const;
#endif
};

#endif
