#ifndef HIPRT_SCENE_DATA_H
#define HIPRT_SCENE_DATA_H

#include "Kernels/includes/HIPRT_common.h"

struct HIPRTRenderSettings
{
	// How many times the render kernel was called (updates after
	// the call to the kernel so it start at 0)
	int frame_number = 0;
	// Number of samples rendered so far before the kernel call
	// This is the sum of samples_per_frame for all frames
	// that have been rendered
	int sample_number = 0;

	int samples_per_frame = 1;
	int nb_bounces = 8;

	bool render_low_resolution = false;
};

struct WorldBuffers
{
	Color* pixels = nullptr;
	// World space normals and albedo for the denoiser
	hiprtFloat3* denoiser_normals = nullptr;
	Color* denoiser_albedo = nullptr;

	// A device pointer to the buffer of triangles indices
	int* triangles_indices = nullptr;
	// A device pointer to the buffer of triangle vertices
	hiprtFloat3* triangles_vertices = nullptr;
	// A device pointer to a buffer filled with 0s and 1s that
	// indicates whether or not a vertex normal is available for
	// the given vertex index
	unsigned char* normals_present = nullptr;
	// The smooth normal at each vertex of the scene
	// Needs to be indexed by a vertex index
	hiprtFloat3* vertex_normals = nullptr;

	// Index of the material used by each triangle of the scene
	int* material_indices = nullptr;
	// Materials array to be indexed by an index retrieved from the 
	// material_indices array
	RendererMaterial* materials_buffer = nullptr;
	int emissive_triangles_count = 0;
	int* emissive_triangles_indices = nullptr;
};

struct WorldSettings
{
	bool use_ambient_light = true;
	Color ambient_light_color = Color(0.5f);
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
	WorldSettings world_settings;

	HIPRTRenderSettings render_settings;
};

#endif
