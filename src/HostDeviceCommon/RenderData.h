/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_DATA_H
#define HOST_DEVICE_COMMON_RENDER_DATA_H

#include "Device/includes/ReSTIR/DI/Reservoir.h"
#include "Device/includes/GBuffer.h"

#include "HostDeviceCommon/HIPRTCamera.h"
#include "HostDeviceCommon/Material.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/RenderSettings.h"
#include "HostDeviceCommon/WorldSettings.h"

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

struct BRDFsData
{
	// 32x32 texture containing the precomputed parameters of the LTC
	// fitted to approximate the SSGX sheen volumetric layer.
	// See SheenLTCFittedParameters.h
	void* sheen_ltc_parameters_texture = nullptr;

	// 2D texture for the precomputed directional albedo
	// for the GGX BRDFs used in the principled BSDF for energy conservation
	// of conductors
	void* GGX_Ess = nullptr;

	// 3D texture for the precomputed directional albedo of the base layer
	// of the principled BSDF (specular GGX layer + diffuse below)
	void* glossy_dielectric_Ess = nullptr;

	// 3D texture (cos_theta_o, roughness, relative_eta) for the precomputed
	// directional albedo used for energy conservation of glass objects when
	// entering a medium
	void* GGX_Ess_glass = nullptr;
	// Table when leaving a medium
	void* GGX_Ess_glass_inverse = nullptr;

	// Whether or not to use the texture unit's hardware texel interpolation
	// when fetching the LUTs. It's faster but less precise.
	bool use_hardware_tex_interpolation = false;
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

	// If a given pixel has converged, this buffer contains the number of samples
	// that were necessary for the convergence. 
	// 
	// If the pixel hasn't converged yet, the buffer contains the -1 value
	int * pixel_converged_sample_count = nullptr;

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
	// The buffers that should be used by the ReSTIR passes kernels are the 
	// 'input_reservoirs' / 'output_reservoirs' buffers of the 'initial_candidates',
	// 'temporal_pass' and 'spatial_pass' settings
	ReSTIRDIReservoir* restir_reservoir_buffer_1 = nullptr;
	ReSTIRDIReservoir* restir_reservoir_buffer_2 = nullptr;
	ReSTIRDIReservoir* restir_reservoir_buffer_3 = nullptr;
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

	// GPU BVH
	hiprtGeometry geom = nullptr;
	// GPU Intersection functions (for alpha testing for example)
	hiprtFuncTable hiprt_function_table = nullptr;

	// Size of the *global* stack per thread. Default is 32.
	int global_traversal_stack_buffer_size = 32;
	hiprtGlobalStackBuffer global_traversal_stack_buffer = { 0, 0, nullptr };

	RenderBuffers buffers;
	BRDFsData brdfs_data;
	AuxiliaryBuffers aux_buffers;
	GBuffer g_buffer;
	GBuffer g_buffer_prev_frame;

	HIPRTRenderSettings render_settings;
	WorldSettings world_settings;

	// Camera for the current frame
	HIPRTCamera current_camera;
	// Camera of the last frame
	HIPRTCamera prev_camera;

	// Data only used by the CPU
	CPUData cpu_only;
};

#endif
