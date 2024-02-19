#ifndef HIPRT_SCENE_DATA_H
#define HIPRT_SCENE_DATA_H

#include "Kernels/includes/HIPRT_common.h"

struct HIPRTRenderSettings
{
	int frame_number = 0;

	int samples_per_frame = 1;
	int nb_bounces = 8;
};

/*
 * A structure containing all the information about the scene
 * that the kernel is going to need for the render (vertices of the triangles, 
 * vertices indices, skysphere data, ...)
 */
struct HIPRTRenderData
{
	HIPRTRenderData() : geom(nullptr), triangles_indices(nullptr), triangles_vertices(nullptr),
	normals_present(nullptr), vertex_normals(nullptr),
	material_indices(nullptr), materials_buffer(nullptr), emissive_triangles_count(0),
	emissive_triangles_indices(nullptr) {}

	hiprtGeometry geom;

	// A device pointer to the buffer of triangles indices
	int* triangles_indices;
	// A device pointer to the buffer of triangle vertices
	hiprtFloat3* triangles_vertices;
	// A device pointer to a buffer filled with 0s and 1s that
	// indicates whether or not a vertex normal is available for
	// the given vertex index
	unsigned char* normals_present;
	// The smooth normal at each vertex of the scene
	// Needs to be indexed by a vertex index
	hiprtFloat3* vertex_normals;

	// Index of the material used by each triangle of the scene
	int* material_indices;
	// Materials array to be indexed by an index retrieved from the 
	// material_indices array
	HIPRTRendererMaterial* materials_buffer;
	int emissive_triangles_count;
	int* emissive_triangles_indices;

	HIPRTRenderSettings render_settings;
};

#endif
