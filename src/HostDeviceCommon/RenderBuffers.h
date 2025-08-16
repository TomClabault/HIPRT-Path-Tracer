/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RENDER_BUFFERS_H
#define HOST_DEVICE_COMMON_RENDER_BUFFERS_H

#include "Device/includes/AliasTable.h"
#include "Device/includes/EmissiveMeshesAliasTables.h"
#include "Device/includes/GMoN/GMoNDevice.h"

#include "HostDeviceCommon/Material/MaterialPackedSoA.h"
#include "HostDeviceCommon/PrecomputedEmissiveTrianglesDataSoADevice.h"

struct RenderBuffers
{
	// Sum of samples color per pixel. Should not be
	// pre-divided by the number of samples i.e. this buffer
	// contains pure accumulation of pixel colors
	ColorRGB32F* accumulated_ray_colors = nullptr;

	// Data for the GMoN estimator
	GMoNDevice gmon_estimator;

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
	// Precomputed areas of all triangles of the scene
	float* triangles_areas = nullptr;

	// Index of the material used by each triangle of the scene
	int* material_indices = nullptr;
	// Materials array to be indexed by an index retrieved from the 
	// material_indices array
	DevicePackedTexturedMaterialSoA materials_buffer;
	// A buffer that can be indexed by a material_id.
	// 
	// If indexing this buffer returns true, then the material is fully opaque
	// and there is no need to test alpha testing for it
	//
	// This is actually a buffer of bools but manipulating bools is annoying so this
	// is unsigned char. But the value of the unsigned char is either 0 or 1
	unsigned char* material_opaque = nullptr;

	int emissive_triangles_count = 0;
	// A buffer that contains the primitive indices of the emissive triangles of the scene
	// Does not contains the indices of the emissive triangles that have emissive textures
	int* emissive_triangles_primitive_indices = nullptr;
	// Same as 'emissive_triangles_primitive_indices' but does contain the indices of the emissive triangles
	// that have emissive textures
	int* emissive_triangles_primitive_indices_and_emissive_textures = nullptr;
	// Alias table for sampling emissives lights according to power
	AliasTableDevice emissive_triangles_power_alias_table;
	EmissiveMeshesAliasTablesDevice emissive_meshes_alias_tables;

	// A pointer either to an array of Image8Bit or to an array of
	// oroTextureObject_t whether if CPU or GPU rendering respectively
	// This pointer can be cast for the textures to be be retrieved.
	void* material_textures = nullptr;
};

#endif
