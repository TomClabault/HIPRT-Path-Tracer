/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_SCENE_DATA_H
#define HIPRT_SCENE_DATA_H

#include "HostDeviceCommon/Material.h"
#include "HostDeviceCommon/Math.h"

struct HIPRTRenderSettings
{
	// How many times the render kernel was called (updates after
	// the call to the kernel so it start at 0)
	int frame_number = 0;

	// Number of samples rendered so far before the kernel call
	// This is the sum of samples_per_frame for all frames
	// that have been rendered
	//
	// TODO still relevant with new per-pixel sample count buffer used
	// with adaptive sampling ?
	int sample_number = 0;

	int samples_per_frame = 1;
	int nb_bounces = 8;

	// If true, this means that the user is moving the camera and we're going to
	// render the image at a much lower resolution to allow for smooth camera
	// movements
	bool render_low_resolution = false;

	bool enable_adaptive_sampling = true;
	// How many samples before the adaptive sampling actually kicks in.
	// This is useful mainly for the per-pixel adaptive sampling method
	// where you want to be sure that each pixel in the image has had enough
	// chance find a path to a potentially 
	int adaptive_sampling_min_samples = 64;
	// Adaptive sampling noise threshold
	float adaptive_sampling_noise_threshold = 0.1f;
};

struct WorldBuffers
{
	// Sum of samples color per pixel. Should not be
	// pre-divided by the number of samples
	ColorRGB* pixels = nullptr;

	// A device pointer to the buffer of triangles indices
	int* triangles_indices = nullptr;
	// A device pointer to the buffer of triangle vertices positions
	float3* triangles_vertices = nullptr;
	// A device pointer to a buffer filled with 0s and 1s that
	// indicates whether or not a vertex normal is available for
	// the given vertex index
	unsigned char* normals_present = nullptr;
	// The smooth normal at each vertex of the scene
	// Needs to be indexed by a vertex index
	float3* vertex_normals = nullptr;

	// Index of the material used by each triangle of the scene
	int* material_indices = nullptr;
	// Materials array to be indexed by an index retrieved from the 
	// material_indices array
	RendererMaterial* materials_buffer = nullptr;
	int emissive_triangles_count = 0;
	int* emissive_triangles_indices = nullptr;
};

struct AuxiliaryBuffers
{
	int* debug_pixel_active = nullptr;

	// World space normals for the denoiser
	// These normals should already be divided by the number of samples
	float3* denoiser_normals = nullptr;

	// Albedo for the denoiser
	// The albedo should already be divided by the number of samples
	ColorRGB* denoiser_albedo = nullptr;

	// Per pixel sample count. Useful when doing adaptative sampling
	// where each pixel can have a different number of sample
	int* pixel_sample_count;

	// Per pixel sum of squared luminance of samples. Used for adaptative sampling
	// This buffer should not be pre-divided by the number of samples
	float* pixel_squared_luminance;
};

struct WorldSettings
{
	bool use_ambient_light = true;
	ColorRGB ambient_light_color = ColorRGB(0.5f);
};

/**
 * The CPU and GPU use the same kernel code but the CPU still need some specific data
 * (the CPU BVH for example) which is stored in this structure
 */

class BVH;
struct CPUData
{
	BVH* bvh;
};

/*
 * A structure containing all the information about the scene
 * that the kernel is going to need for the render (vertices of the triangles, 
 * vertices indices, skysphere data, ...)
 */
struct HIPRTRenderData
{
	hiprtGeometry geom = nullptr;

	WorldBuffers buffers;
	AuxiliaryBuffers aux_buffers;
	WorldSettings world_settings;

	HIPRTRenderSettings render_settings;

	CPUData cpu_only;
};

#endif
