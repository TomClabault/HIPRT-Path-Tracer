/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_DATA_H
#define HOST_DEVICE_COMMON_RENDER_DATA_H

#include "Device/includes/GBufferDevice.h"
#include "Device/includes/NEE++/NEE++.h"

#include "Device/includes/LightSampling/LightTree/LightTreeATSDevice.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGDevice.h"
#include "HostDeviceCommon/AuxiliaryBuffers.h"
#include "HostDeviceCommon/BSDFsData.h"
#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/RenderBuffers.h"
#include "HostDeviceCommon/RenderSettings.h"
#include "HostDeviceCommon/SSBNPermutationSettings.h"
#include "HostDeviceCommon/WorldSettings.h"

#ifdef __KERNELCC__
#include <hiprt/hiprt_device.h>
#include <Orochi/Orochi.h>
#endif

/**
 * The CPU and GPU use the same kernel code but the CPU still need some specific data
 * (the CPU BVH for example) which is stored in this structure
 */

class BVH;
struct CPUData
{
	// BVH built over all the triangles of the scene
	BVH* bvh = nullptr;

	// BVH built over the emissive triangles of the scene only
	BVH* light_bvh = nullptr;
};

/*
 * A structure containing all the information about the scene
 * that the kernel is going to need for the render (vertices of the triangles,
 * vertices indices, skysphere data, ...)
 */
struct HIPRTRenderData
{
	// HIPRT BVH built over all the triangles of the scene
	hiprtGeometry GPU_BVH = nullptr;
	// HIPRT BVH built over the emissive triangles of the scene only
	hiprtGeometry light_GPU_BVH = nullptr;
	// GPU Intersection functions (for alpha testing for example)
	hiprtFuncTable hiprt_function_table = nullptr;

	// Size of the *global* stack per thread. Default is 32.
	int global_traversal_stack_buffer_size				 = 32;
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
	LightTreeATSDevice light_tree_ats;
	LightTreeSGDevice light_tree_sg;

	// Data for SSBN permutations
	SSBNPermutationSettings ssbn_settings;

	// Camera for the current frame
	HIPRTCamera current_camera;
	// Camera of the last frame
	HIPRTCamera prev_camera;

	// Data only used by the CPU
	CPUData cpu_only;

	HIPRT_DEVICE unsigned int get_input_random_seed(int pixel_index) const
	{
		return buffers.input_random_seeds[pixel_index];
	}

	HIPRT_DEVICE unsigned int get_updated_random_seed(int pixel_index) const
	{
		return buffers.updated_random_seeds[pixel_index];
	}

	HIPRT_DEVICE void store_input_random_seed(int pixel_index, unsigned int seed) const
	{
		buffers.input_random_seeds[pixel_index] = seed;
	}

	HIPRT_DEVICE void store_updated_random_seed(int pixel_index, unsigned int seed) const
	{
		buffers.updated_random_seeds[pixel_index] = seed;

#if SSBNPermutationEnabled == KERNEL_OPTION_FALSE
		// If we don't have the SSBN permutation render pass enabled, then we need to update the input random seeds such that the next frame reads the updated
		// seeds and produces new random numbers and converges correctly. If we don't store the input seeds here, then the kernels of the next frame are going
		// to read the same input seeds = produce the exact same frame and we will not have convergence
		buffers.input_random_seeds[pixel_index] = seed;
#endif
	}
};

#endif
