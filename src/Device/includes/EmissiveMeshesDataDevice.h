/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_EMISSIVE_MESHES_ALIAS_TABLES_H
#define DEVICE_INCLUDES_EMISSIVE_MESHES_ALIAS_TABLES_H

#include "Device/includes/EmissiveMeshAliasTableDevice.h"
#include "Device/includes/AliasTable.h"

/**
 * Contains the alias tables probas and aliases of all the emissive meshes in the scene
 * 
 * The alias tables of each emissive mesh are computed during scene load and all these
 * alias table (probas and aliases) are concatenated into the single 'alias_tables_probas'
 * and 'alias_tables_aliases' of this structure
 * 
 * 'offsets' and 'individual_alias_tables_sizes' are used to find where the alias table of a
 * given mesh is in the big concatenated buffer of all alias tables
 * 
 * Also contains an alias table for sampling a mesh amongst all the meshes of the scene
 * according to its total emissive power
 */
struct EmissiveMeshesDataDevice
{
	// An alias table for sampling a mesh according to its total emissive power
	// amongst all the meshes of the scene
	AliasTableDevice meshes_alias_table;
	// PDF that the 'meshes_alias_table' samples a given mesh index
	// 
	// This buffer is 'alias_table_count' entries long
	float* meshes_PDFs = nullptr;
	// Average of all the vertices of the mesh
	float3* meshes_average_points = nullptr;
	// Sum of the emissive power of all the triangles of the mesh
	float* meshes_total_power = nullptr;
	// The faces of the emissive mesh are going to be binned according to their normal orientation
	// for better sampling later.
	//
	// All the indices of the faces are going to be found in 'binned_faces_indices'.
	// This buffer contains all the faces of all the meshes concatenated in this single buffer.
	// 
	// The faces of a given mesh index starts at 'offsets[mesh_index]'
	//
	// There are as many bins as ParsedEmissiveMesh::BinningNormals.size()
	//
	// In the binned_faces_indices buffer (which is a big concatenated buffer):
	// Bin[0] starts at 'binned_faces_start_index[0]' and contains 'binned_faces_counts[0]' faces
	// Bin[1] starts at 'binned_faces_start_index[1]' and contains 'binned_faces_counts[1]' faces
	//
	// And also add to that above ^ the offset of the mesh in the big concatenated buffer: offsets[mesh_index]
	unsigned int* binned_faces_indices = nullptr;
	unsigned int* binned_faces_start_index = nullptr;
	unsigned int* binned_faces_counts = nullptr;
	// Total power of all the faces in each bin of a mesh. This is used to sample a bin and also to evaluate
	// a more precise contribution of the mesh to a point
	float* binned_faces_total_power = nullptr;
	// Given a face index in [0, faceCountOfMEsh - 1], this contains the bin index that this face belongs
	// to for that mesh
	//
	// Should be indexed as binned_faces_mesh_face_index_to_bin_index[offsets[emissive_mesh_index] + face_index]
	// for example.
	unsigned int* binned_faces_mesh_face_index_to_bin_index = nullptr;

	// How many alias tables are in the big concatenated buffer of alias tables.
	// This is also how many emissive meshes there are in the scene.
	unsigned int alias_table_count = 0;
	// Offsets a the alias table of a given mesh index in the big concatenated buffer
	// of all the alias tables 'alias_tables_probas' and 'alias_tables_aliases'
	// 
	// This buffer is 'alias_table_count' entries long
	unsigned int* offsets = nullptr;
	// Size of the alias table of a given mesh index. This is also the number of
	// emissive faces per mesh
	// 
	// This buffer is 'alias_table_count' entries long
	unsigned int* individual_alias_tables_sizes = nullptr;

	// These 3 buffers are as long as there are emissive triangles in the scene
	float* alias_tables_probas = nullptr;
	int* alias_tables_aliases = nullptr;
	float* meshes_emissive_triangles_PDFs = nullptr;
	int* meshes_triangle_indices = nullptr;

	static constexpr unsigned int BIN_NORMAL_COUNT = 14;

	/**
	 * Returns the binning normal used for binning the emissive triangles of an emissive
	 * meshes by orientation used during scene parsing.
	 * 
	 * This is basically an utility function to generate the ParsedEmissiveMesh::BinningNormals
	 * normals but based on an index
	 */
	HIPRT_DEVICE static float3 get_binning_normal(int normal_index)
	{
		// Valid normal_index: 0..13

		if (normal_index < 6) 
		{
			// Faces of a cube for the 6 first normals

			// 0 -> x, 1 -> y, 2 -> z
			int axis = normal_index % 3;
			// 0..2 => +, 3..5 => -
			int sign = (normal_index / 3) ? -1 : 1;
			if (axis == 0) 
				return make_float3((float)sign, 0.0f, 0.0f);
			else if (axis == 1) 
				return make_float3(0.0f, (float)sign, 0.0f);
			else
				return make_float3(0.0f, 0.0f, (float)sign);
		}
		else 
		{
			// 0..7
			int j = normal_index - 6;

			float sx = (j & 1) ? 1.0f : -1.0f;
			float sy = (j & 2) ? 1.0f : -1.0f;
			float sz = (j & 4) ? 1.0f : -1.0f;

			const float inv_sqrt3 = 1.0f / sqrtf(3.0f);

			return make_float3(sx * inv_sqrt3, sy * inv_sqrt3, sz * inv_sqrt3);
		}
	}

	HIPRT_DEVICE float get_power_sampled_triangle_PDF_in_mesh(unsigned int emissive_mesh_index, float sampled_triangle_area, const ColorRGB32F& sampled_triangle_emission) const
	{
		return sampled_triangle_area * sampled_triangle_emission.luminance() / meshes_total_power[emissive_mesh_index];
	}

	HIPRT_DEVICE float get_binned_faces_sampled_triangle_PDF_in_mesh(unsigned int emissive_mesh_index, unsigned int triangle_index_in_mesh, float3 mesh_representative_point, float3 surface_point) const
	{
		unsigned int bin_index = binned_faces_mesh_face_index_to_bin_index[offsets[emissive_mesh_index] + triangle_index_in_mesh];

		// A bin is sampled according to its contribution given a surface normal
		float3 direction_to_mesh = hippt::normalize(mesh_representative_point - surface_point);
		float total_power = 0.0f;
		for (int i = 0; i < BIN_NORMAL_COUNT; i++)
			total_power += binned_faces_total_power[emissive_mesh_index * BIN_NORMAL_COUNT + i] * hippt::max(0.0f, hippt::dot(direction_to_mesh, -EmissiveMeshesDataDevice::get_binning_normal(i)));
		if (total_power == 0.0f)
			return 0.0f;

		float bin_proba = binned_faces_total_power[emissive_mesh_index * BIN_NORMAL_COUNT + bin_index] * hippt::max(0.0f, hippt::dot(direction_to_mesh, -EmissiveMeshesDataDevice::get_binning_normal(bin_index))) / total_power;
		float triangle_in_bin_proba = 1.0f / binned_faces_counts[emissive_mesh_index * BIN_NORMAL_COUNT + bin_index];

		return bin_proba * triangle_in_bin_proba;
	}

	HIPRT_DEVICE EmissiveMeshAliasTableDevice get_emissive_mesh_alias_table(unsigned int emissive_mesh_index) const
	{
		EmissiveMeshAliasTableDevice out;

		unsigned int offset = offsets[emissive_mesh_index];
		out.alias_table_alias = alias_tables_aliases + offset;
		out.alias_table_probas = alias_tables_probas + offset;
		out.PDFs = meshes_emissive_triangles_PDFs + offset;
		out.triangle_indices = meshes_triangle_indices + offset;
		out.size = individual_alias_tables_sizes[emissive_mesh_index];

		return out;
	}

	HIPRT_DEVICE EmissiveMeshAliasTableDevice sample_one_emissive_mesh(Xorshift32Generator& rng, float& out_pdf, unsigned int& out_emissive_mesh_index_sampled) const
	{
		out_emissive_mesh_index_sampled = meshes_alias_table.sample(rng);
		out_pdf = meshes_PDFs[out_emissive_mesh_index_sampled];

		return get_emissive_mesh_alias_table(out_emissive_mesh_index_sampled);
	}

	HIPRT_DEVICE EmissiveMeshAliasTableDevice sample_one_emissive_mesh(Xorshift32Generator& rng, float& out_pdf) const
	{
		unsigned int emissive_mesh_index_trash;

		return sample_one_emissive_mesh(rng, out_pdf, emissive_mesh_index_trash);
	}
};

#endif
