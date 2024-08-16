/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_DATA_H
#define HOST_DEVICE_COMMON_RENDER_DATA_H

#include "Device/includes/Reservoir.h"
#include "Device/includes/GBuffer.h"
#include "HostDeviceCommon/Material.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/RenderSettings.h"

#include <hiprt/hiprt_device.h>
#include <Orochi/Orochi.h>

#ifdef __KERNELCC__
template <typename T>
using AtomicType = T;
#else
#include <atomic>

template <typename T>
using AtomicType = std::atomic<T>;
#endif

struct RenderBuffers
{
	// Sum of samples color per pixel. Should not be
	// pre-divided by the number of samples
	ColorRGB32F* pixels = nullptr;

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

	// A pointer either to an array of Image8Bit or to an array of
	// oroTextureObject_t whether if CPU or GPU rendering respectively
	// This pointer can be cast for the textures to be be retrieved.
	void* material_textures = nullptr;
	// Widths of the textures. Necessary for using texel coordinates in [0, width - 1]
	// in the shader (required because Orochi doesn't support normalized texture coordinates).
	int2* textures_dims = nullptr;
};

struct AuxiliaryBuffers
{
	// Whether or not the pixel at a given index in the buffer is active or not. A pixel can be
	// inactive when we're rendering at low resolution for example or when adaptive sampling has
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
	AtomicType<unsigned int>* stop_noise_threshold_count = nullptr;

	// Pointers to the buffers allocated on the GPU. These pointers
	// exist basically only to be reset in reset_render(). They should not
	// be manipulated directly in the ReSTIR passes. 
	// The buffers that should be used by the ReSTIR passes kernels are the 
	// 'input_reservoirs' / 'output_reservoirs' buffers of the 'initial_candidates',
	// 'temporal_pass' and 'spatial_pass' settings
	Reservoir* initial_reservoirs;
	Reservoir* temporal_pass_output_reservoirs;
	Reservoir* final_reservoirs;
};

enum AmbientLightType
{
	NONE,
	UNIFORM,
	ENVMAP
};

struct WorldSettings
{
	AmbientLightType ambient_light_type = AmbientLightType::NONE;
	ColorRGB32F uniform_light_color = ColorRGB32F(0.5f);

	// Width and height in pixels. Both in the range [1, XXX]
	unsigned int envmap_width = 0, envmap_height = 0;
	// Simple scale multiplier on the envmap color read from the envmap texture
	// in the shader
	float envmap_intensity = 1.0f;
	// If true, the background of the scene (where rays directly miss any geometry
	// and we directly see the skysphere) will scale with the envmap_intensity coefficient.
	// This can be visually unpleasing because the background will most likely
	// become completely white and blown out.
	int envmap_scale_background_intensity = false;
	// This void pointer is a either a float* for the CPU
	// or a oroTextureObject_t for the GPU.
	// Proper reinterpreting of the pointer is done in the kernel.
	void* envmap = nullptr;
	// Cumulative distribution function. 1D float array of length width * height for
	// importance sampling the envmap
	float* envmap_cdf = nullptr;
	// Rotation matrix for rotating the envmap around
	float4x4 envmap_rotation_matrix = float4x4{ { {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 1.0f } } };
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
	unsigned int random_seed = 42;

	hiprtGeometry geom = nullptr;

	RenderBuffers buffers;
	AuxiliaryBuffers aux_buffers;
	WorldSettings world_settings;
	GBuffer g_buffer;

	HIPRTRenderSettings render_settings;

	CPUData cpu_only;
};

#endif
