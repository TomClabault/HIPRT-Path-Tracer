/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_SCENE_DATA_H
#define HIPRT_SCENE_DATA_H

#include "HostDeviceCommon/Material.h"
#include "HostDeviceCommon/Math.h"

#include <hiprt/hiprt_device.h>
#include <Orochi/Orochi.h>

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

struct RenderBuffers
{
	// Sum of samples color per pixel. Should not be
	// pre-divided by the number of samples
	ColorRGB* pixels = nullptr;

	// A device pointer to the buffer of triangles vertex indices
	// triangles_indices[0], triangles_indices[1] and triangles_indices[2]
	// represent the indices of the vertices of the first triangle for example
	int* triangles_indices = nullptr;
	// A device pointer to the buffer of triangle vertices positions
	float3* vertices_positions = nullptr;
	// A device pointer to a buffer filled with 0s and 1s that
	// indicates whether or not a vertex normal is available for
	// the given vertex index
	unsigned char* has_vertex_normals = nullptr;
	// The smooth normal at each vertex of the scene
	// Needs to be indexed by a vertex index
	float3* vertex_normals = nullptr;
	// Texture coordinates at each vertices
	float2* texcoords = nullptr;

	// Index of the material used by each triangle of the scene
	int* material_indices = nullptr;
	// Materials array to be indexed by an index retrieved from the 
	// material_indices array
	RendererMaterial* materials_buffer = nullptr;
	int emissive_triangles_count = 0;
	int* emissive_triangles_indices = nullptr;

	// A pointer either to a list of ImageRGBA or to a list of
	// oroTextureObject_t whether if CPU or GPU renderer respectively
	// This pointer can be cast for the textures to be be retrieved.
	void* material_textures = nullptr;
	// Whether the texture at the given index in this buffer is sRGB.
	// Note that we could be using hardware sRGB to linear conversion in the sampler
	// but it seems to be broken (?) with Orochi so we're doing it in software in the
	// shader.
	// Also, we're using unsigned char instead of bool here because std::vector<bool> doesn't
	// have a .data() member function which is annoying to manipulate
	unsigned char* texture_is_srgb;
};

struct AuxiliaryBuffers
{
	// World space normals for the denoiser
	// These normals should already be divided by the number of samples
	float3* denoiser_normals = nullptr;

	// Albedo for the denoiser
	// The albedo should already be divided by the number of samples
	ColorRGB* denoiser_albedo = nullptr;

	// Per pixel sample count. Useful when doing adaptive sampling
	// where each pixel can have a different number of sample
	int* pixel_sample_count = nullptr;

	// Per pixel sum of squared luminance of samples. Used for adaptive sampling
	// This buffer should not be pre-divided by the number of samples
	float* pixel_squared_luminance = nullptr;
};

enum AmbientLightType
{
	NONE,
	UNIFORM,
	ENVMAP
};

struct WorldSettings
{
	AmbientLightType ambient_light_type = AmbientLightType::UNIFORM;
	ColorRGB uniform_light_color = ColorRGB(0.5f);

	unsigned int envmap_width = 0, envmap_height = 0;
	// This void pointer is a either a pointer to float* for the CPU
	// or a pointer to oroTextureObject_t for the GPU.
	// Proper reinterpreting of the pointer is done in the kernel
	void* envmap = nullptr;
	// Cumulative distribution function. 1D float array of length width * height for
	// importance sampling the envmap
	float* envmap_cdf;
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
	hiprtGeometry geom = nullptr;

	RenderBuffers buffers;
	AuxiliaryBuffers aux_buffers;
	WorldSettings world_settings;

	HIPRTRenderSettings render_settings;

	CPUData cpu_only;
};

#endif
