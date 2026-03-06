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
	// Random numbers that is updated by the CPU and that can help generate a
	// random seed on the GPU for the random number generator to get started
	//
	// This is a fullscreen buffer of random seeds, one for each pixel
	unsigned int* random_seeds = nullptr;
	// If true, the next camera rays kernel call will reset the random seeds.
	bool need_to_reset_random_seeds = true;

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

	// Camera for the current frame
	HIPRTCamera current_camera;
	// Camera of the last frame
	HIPRTCamera prev_camera;

	// Data only used by the CPU
	CPUData cpu_only;
};

#endif
