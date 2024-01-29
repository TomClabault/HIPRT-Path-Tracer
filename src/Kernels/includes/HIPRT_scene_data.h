#ifndef HIPRT_SCENE_DATA_H
#define HIPRT_SCENE_DATA_H

#include <hiprt/hiprt_vec.h>

/*
 * A structure containing all the information about the scene
 * that the kernel is going to need for the render (vertices of the triangles, 
 * vertices indices, skysphere data, ...)
 */
struct HIPRTSceneData
{
	HIPRTSceneData() : triangles_indices(nullptr), triangles_vertices(nullptr) {}

	// A device pointer to the buffer of triangles indices
	int* triangles_indices;
	// A device pointer to the buffer of triangle vertices
	hiprtFloat3* triangles_vertices;
};

#endif
