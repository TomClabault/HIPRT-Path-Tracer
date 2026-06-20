/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_SETTINGS_H
#define HOST_DEVICE_COMMON_RENDER_SETTINGS_H

#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/Maths/Math.h"
#include "HostDeviceCommon/PathRussianRoulette.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRCommonSettings.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRDISettings.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRGISettings.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRPGSettings.h"
#include "HostDeviceCommon/ReSTIR/ReSTIRPTSettings.h"
#include "HostDeviceCommon/RIS/RISSettings.h"
#include "HostDeviceCommon/RISLTC/RISLTCSettings.h"

#ifndef __KERNELCC__
#include "HIPRT-Orochi/OrochiBuffer.h"
#endif

// Just used for initializing some structure members below
#define local_min_macro(a, b) ((a) < (b) ? (a) : (b))

class GPURenderer;

struct HIPRTRenderSettings
{
	int2_t render_resolution = make_int2(1280, 720);

	// If true, the camera ray kernel will reset all buffers to their default values.
	// This is mainly useful for the first frame of the render
	bool need_to_reset = true;
	// Whether or not to reset the random seeds of each pixel to their default values when the render is reset (i.e. when need_to_reset is true above). The
	// resetting is done by the camera rays kernel
	bool need_to_reset_random_seeds = true;

	// This is a random number variable that can be set just before launching a kernel. That random number can then be used in that kernel to shuffle random
	// seeds for example. This allows launching the same kernel 2 times in a row but with a different RNG if that random number is used to generate per pixel
	// seeds for example
	unsigned int random_number = 42;

	// TODO DEBUG REMOVE THESE
	////////////////////////////////////////////////////

	int DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS				   = 4;
	int DEBUG_REGIR_PRE_INTEGRATION_SAMPLE_COUNT_PER_RESERVOIR = 32;

	static constexpr unsigned long long int DEBUG_DEFAULT_ULL = 4242424242;
	static constexpr float DEBUG_DEFAULT_FLOAT				  = -4242.0f;
	static constexpr int DEBUG_STRING_MAX_LENGTH			  = 96;

	AtomicType<unsigned long long int>* DEBUG_BUFFER_ULL_1 = nullptr;
	AtomicType<unsigned long long int>* DEBUG_BUFFER_ULL_2 = nullptr;
	AtomicType<float>* DEBUG_BUFFER_FLOAT				   = nullptr;
	char* DEBUG_BUFFER_STRINGS							   = nullptr;

	HIPRT_DEVICE void write_debug_string(const char debug_string[HIPRTRenderSettings::DEBUG_STRING_MAX_LENGTH], int index) const
	{
		for (int str_index = 0; debug_string[str_index]; str_index++)
			DEBUG_BUFFER_STRINGS[index * HIPRTRenderSettings::DEBUG_STRING_MAX_LENGTH + str_index] = debug_string[str_index];
	}

#ifndef __KERNELCC__
	void print_debug_floats(int max_number_of_values)
	{
		std::vector<float> debug		   = OrochiBuffer<float>::download_data((float*)DEBUG_BUFFER_FLOAT, 1024);
		std::vector<char> debug_string_CPU = OrochiBuffer<char>::download_data(DEBUG_BUFFER_STRINGS, 1024 * HIPRTRenderSettings::DEBUG_STRING_MAX_LENGTH);

		printf("\n\n-----------------\n");
		for (int i = 0; i < max_number_of_values; i++)
		{
			if (debug[i] != DEBUG_DEFAULT_FLOAT)
			{

				std::string debug_str(&debug_string_CPU[i * HIPRTRenderSettings::DEBUG_STRING_MAX_LENGTH]);
				// std::string debug_str = GPURenderer::read_debug_buffer_string(DEBUG_BUFFER_STRINGS, i);
				printf("\t(%d) %s: %f\n", i, debug_str.c_str(), debug[i]);
			}
		}
	}
#endif

	////////////////////////////////////////////////////

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

	// Number of samples rendered so far before (before means that this counter starts at 0) the kernel call
	//
	// This is the sum of samples_per_frame for all frames that have been rendered.
	unsigned int sample_number = 0;
	// See the DisplayOnlySampleN kernel option
	int output_debug_sample_N = 1;

	// How many samples to compute per pixel per frame
	// Higher values reduce CPU overhead since the GPU spends
	// more time computing per frame but reduces interactivity
	int samples_per_frame = 1;
	// Maximum number of bounces of rays in the scene.
	// 1 is direct light only.
	int nb_bounces = 1;

	bool do_russian_roulette = false;
	// After how many bounces can russian roulette kick in?
	// 0 means that the camera ray hits, and then the next bounce
	// is already susceptible to being terminated by russian roulette
	int russian_roulette_min_depth = local_min_macro(5, nb_bounces / 2);
	// After applying russian roulette(dividing by the continuation probability)
	// the energy added to the ray throughput is clamped to this maximum value.
	//
	// This is biased and darkens the image the lower the threshold but it helps
	// reduce variance and fireflies introduced by the russian roulette --> faster
	// convergence.
	//
	// 0 for no clamping.
	float russian_roulette_throughput_clamp = 0.0f;

	// What Russian roulette method to use to determine the path termination
	// probability
	PathRussianRoulette path_russian_roulette_method = PathRussianRoulette::MAX_THROUGHPUT;

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
	bool allow_render_low_resolution = false;
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
	float adaptive_sampling_noise_threshold = 0.075f;

	// If true, the rendering will stop after a certain proportion (defined by 'stop_pixel_percentage_converged')
	// of pixels of the image have converged. "converged" here is defined according to the adaptive sampling if
	// enabled or according to 'stop_pixel_noise_threshold' if adaptive sampling is not enabled.
	//
	// If false, the render will not stop until all pixels have converged
	bool use_pixel_stop_noise_threshold = false;
	// A percentage in [0, 100] that dictates the proportion of pixels that must
	// have reached the given noise threshold (stop_pixel_noise_threshold
	// variable) before we stop rendering.
	//
	// For example, if this variable is 90, we will stop rendering when 90% of all
	// pixels have reached the stop_pixel_noise_threshold
	float stop_pixel_percentage_converged = 55.0f;
	// Noise threshold for use with the stop_pixel_percentage_converged stopping
	// condition
	float stop_pixel_noise_threshold = 0.075f;

	// Whether or not to enable direct lighting (direct incoming light at the primary hit)
	bool enable_direct_lighting = true;

	// Clamp direct lighting contribution to reduce fireflies
	float direct_contribution_clamp = 0.0f;
	// Clamp envmap contribution to reduce fireflies
	float envmap_contribution_clamp = 0.0f;
	// Clamp indirect lighting contribution to reduce fireflies
	float indirect_contribution_clamp = 0.0f;

	// Whether or not to do alpha testing for geometry with transparent base color textures
	bool do_alpha_testing = true;

	// Whether or not to do normal mapping at all
	// If false, geometric normals will always be used
	bool do_normal_mapping = true;

	// If using projected solid angle sampling, the triangle must cover at least this much
	// to be sampled with projected solid angle sampling, otherwise it will be sampled with
	// solid angle sampling alone (not projected) which is much cheaper
	//
	// The default value is arbitrary and observed to reduce variance in important places
	float projected_solid_angle_sampling_threshold = 0.06f;

	// Settings for RIS (direct light sampling)
	RISSettings ris_settings;
	RISLTCSettings risltc_settings;

	// Settings for ReSTIR DI
	ReSTIRDISettings restir_di_settings;
	// Settings for ReSTIR GI
	ReSTIRGISettings restir_gi_settings;
	// Settings for ReSTIR PT
	ReSTIRPTSettings restir_pt_settings;
	// Settings for ReSTIR Path Guiding
	//
	// TODO
	// Roughness based MIS weight
	// Stop training after N samples
	// Fix everything for principled BSDF
	ReSTIRPGSettings restir_pg_settings;
	// Settings for ReGIR
	ReGIRSettings regir_settings;

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

		has_access |= (stop_pixel_noise_threshold > 0.0f && use_pixel_stop_noise_threshold);
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
	 * options and check if the DirectLightNEEEstimator is equal to
	 * LSS_RESTIR_DI. On the device however, we don't have access to the
	 * GPURenderer instance but instead, we can check directly using the
	 * DirectLightNEEEstimator macro (and we don't want the GPURenderer parameter
	 * because that doesn't exist on the device).
	 */
	HIPRT_DEVICE bool use_prev_frame_g_buffer() const
	{
		// If ReSTIR DI isn't used, we don't need the last frame's g-buffer
		// (as far as the codebase goes at the time of writing this function anyways)
		bool need_g_buffer = false;
		need_g_buffer |= DirectLightNEEEstimator == LSS_RESTIR_DI && restir_di_settings.common_temporal_pass.do_temporal_reuse_pass;
		need_g_buffer |= PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI && restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass;
		need_g_buffer |= PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT && restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass;

		return need_g_buffer;
	}

	// Only need this one on the host
#ifndef __KERNELCC__
	HIPRT_HOST bool use_prev_frame_g_buffer(GPURenderer* renderer) const;
#endif
};

#endif
