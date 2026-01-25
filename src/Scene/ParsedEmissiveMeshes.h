/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SCENE_PARSED_EMISSIVE_MESHES_H
#define SCENE_PARSED_EMISSIVE_MESHES_H

struct ParsedEmissiveMesh
{
	// Alias table built on the power of all the emissive triangles of the mesh
	std::vector<float> alias_probas;
	std::vector<int> alias_aliases;

	// Average of all the vertices of the emissive mesh
	float3 average_mesh_point = make_float3(0.0f, 0.0f, 0.0f);
	// Representative normal of the mesh
	// 
	// If no good representative normal could be extracted from the mesh at scene parse time
	// then the buffer will contain value float3(INVALID_NORMAL, 0.0f, 0.0f) for that mesh
	float3 representative_normal = make_float3(0.0f, 0.0f, 0.0f);

	float total_mesh_emissive_power = 0.0f;
	unsigned int emissive_triangle_count = 0;
};

struct ParsedEmissiveMeshes
{
	// Contains the list of all emissive meshes of the scene. This list is going to be used by some light
	// sampling scheme such as ReGIR
	// 
	// Any emissive mesh that contains emissive textures is NOT in that list because emissive textures
	// aren't importance sampled
	std::vector<ParsedEmissiveMesh> emissive_meshes;

	// PDF that a given triangle in a given emissive mesh is sampled by the sampler
	// that samples triangles in meshes.
	//
	// For example, if the emissive mesh [0] of the scene has 5 emissive triangles
	// then entries [0], [1], ... [4] of this vector will contain the PDF that triangles
	// 0, 1, ..., 4 are sampled within mesh [0]
	//
	// The PDF is assumed to be power proportional
	std::vector<float> emissive_meshes_triangles_PDFs;

	// For a given triangle index in the whole scene, gives the index of the emissive mesh in
	// [0, alias_table_count - 1] that this triangle belongs to. If the given triangle index doesn't
	// belong to an emissive mesh, the buffer contains -1 at that index
	std::vector<int> global_triangle_index_to_emissive_mesh_index;
};

#endif
