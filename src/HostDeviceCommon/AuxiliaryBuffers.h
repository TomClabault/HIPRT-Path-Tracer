/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_AUXILIARY_BUFFERS_H
#define HOST_DEVICE_COMMON_AUXILIARY_BUFFERS_H

#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "HostDeviceCommon/Color.h"

struct AuxiliaryBuffers
{
	// Whether or not the pixel at a given index in the buffer is active or not. 
	// 
	// A pixel can be inactive when we're rendering at low resolution for example
	// (and so some pixels are not rendered) or when adaptive sampling has
	// judged that the pixel was converged enough and doesn't need more samples
	unsigned char* pixel_active = nullptr;

	// World space normals for the denoiser
	// These normals should already be divided by the number of samples
	float3* denoiser_normals = nullptr;

	// Albedo for the denoiser
	// The albedo should already be divided by the number of samples
	ColorRGB32F* denoiser_albedo = nullptr;

	// Per pixel sample count. Useful when doing adaptive sampling
	// where each pixel can have a different number of sample
	int* pixel_sample_count = nullptr;

	// Per pixel sum of squared luminance of samples. Used for adaptive sampling
	// This buffer should not be pre-divided by the number of samples
	float* pixel_squared_luminance = nullptr;

	// If a given pixel has converged, this buffer contains the number of samples
	// that were necessary for the convergence.
	// 
	// If the pixel hasn't converged yet, the buffer contains the -1 value for that pixel
	int* pixel_converged_sample_count = nullptr;

	// A single boolean (contained in a buffer, hence the pointer) 
	// to indicate whether at least one single ray is still active in the kernel.
	// This is an unsigned char instead of a boolean because std::vector<bool>.data()
	// isn't standard
	unsigned char* still_one_ray_active = nullptr;

	// If render_settings.stop_pixel_noise_threshold > 0.0f, this buffer
	// (consisting of a single unsigned int) counts how many pixels have reached the
	// noise threshold. If this value is equal to the number of pixels of the
	// framebuffer, then all pixels have converged according to the given
	// noise threshold.
	AtomicType<unsigned int>* pixel_count_converged_so_far = nullptr;

	// Same for ReSTIR GI
	ReSTIRGIReservoir* restir_gi_reservoir_buffer_1 = nullptr;
	ReSTIRGIReservoir* restir_gi_reservoir_buffer_2 = nullptr;
	ReSTIRGIReservoir* restir_gi_reservoir_buffer_3 = nullptr;
};

#endif
