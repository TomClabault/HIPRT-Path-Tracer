/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_DATA_H
#define HOST_DEVICE_COMMON_RENDER_DATA_H

#include "Device/includes/GBufferDevice.h"
#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/NEE++/NEE++.h"

#include "HostDeviceCommon/BSDFsData.h"
#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/RenderBuffers.h"
#include "HostDeviceCommon/RenderSettings.h"
#include "HostDeviceCommon/WorldSettings.h"

#include <hiprt/hiprt_device.h>
#include <Orochi/Orochi.h>

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
	// If the pixel hasn't converged yet, the buffer contains the -1 value
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
	AtomicType<unsigned int>* stop_noise_threshold_converged_count = nullptr;

	// Pointers to the buffers allocated on the GPU. These pointers
	// exist basically only to be reset in reset_render(). They should not
	// be manipulated directly in the ReSTIR passes. 
	// 
	// The buffers that should be used by the ReSTIR passes kernels are the 
	// 'input_reservoirs' / 'output_reservoirs' buffers of the 'initial_candidates',
	// 'temporal_pass' and 'spatial_pass' settings
	ReSTIRDIReservoir* restir_di_reservoir_buffer_1 = nullptr;
	ReSTIRDIReservoir* restir_di_reservoir_buffer_2 = nullptr;
	ReSTIRDIReservoir* restir_di_reservoir_buffer_3 = nullptr;

	// Same for ReSTIR GI
	ReSTIRGIReservoir* restir_gi_reservoir_buffer_1 = nullptr;
	ReSTIRGIReservoir* restir_gi_reservoir_buffer_2 = nullptr;
	ReSTIRGIReservoir* restir_gi_reservoir_buffer_3 = nullptr;
};

/**
 * The CPU and GPU use the same kernel code but the CPU still need some specific data
 * (the CPU BVH for example) which is stored in this structure
 */

class BVH;
struct CPUData
{
	BVH* bvh = nullptr;
};

/*
 * A structure containing all the information about the scene
 * that the kernel is going to need for the render (vertices of the triangles, 
 * vertices indices, skysphere data, ...)
 */
struct HIPRTRenderData
{
	// Random number that is updated by the CPU and that can help generate a
	// random seed on the GPU for the random number generator to get started
	unsigned int random_number = 42;

	// HIPRT BVH
	hiprtGeometry GPU_BVH = nullptr;
	// GPU Intersection functions (for alpha testing for example)
	hiprtFuncTable hiprt_function_table = nullptr;

	// Size of the *global* stack per thread. Default is 32.
	int global_traversal_stack_buffer_size = 32;
	hiprtGlobalStackBuffer global_traversal_stack_buffer = { 0, 0, nullptr };

	RenderBuffers buffers;
	BRDFsData bsdfs_data;
	AuxiliaryBuffers aux_buffers;
	GBufferDevice g_buffer;
	GBufferDevice g_buffer_prev_frame;

	HIPRTRenderSettings render_settings;
	WorldSettings world_settings;

	// Data for NEE++
	NEEPlusPlusDevice nee_plus_plus;

	// Camera for the current frame
	HIPRTCamera current_camera;
	// Camera of the last frame
	HIPRTCamera prev_camera;

	// Data only used by the CPU
	CPUData cpu_only;
};

#endif
