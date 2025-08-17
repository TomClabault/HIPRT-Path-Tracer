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
struct EmissiveMeshesAliasTablesDevice
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

	// How many alias tables are in the big concatenated buffer of alias tables
	// This is also how many emissive meshes there are in the scene.
	unsigned int alias_table_count = 0;
	// Offsets a the alias table of a given mesh index in the big concatenated buffer
	// of all the alias tables 'alias_tables_probas' and 'alias_tables_aliases'
	// 
	// This buffer is 'alias_table_count' entries long
	unsigned int* offsets = nullptr;
	// Size of the alias table of a given mesh index
	// 
	// This buffer is 'alias_table_count' entries long
	unsigned int* individual_alias_tables_sizes = nullptr;

	// These 3 buffers are as long as there are emissive triangles in the scene
	float* alias_tables_probas = nullptr;
	int* alias_tables_aliases = nullptr;
	float* meshes_emissive_triangles_PDFs = nullptr;
	int* meshes_triangle_indices = nullptr;

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

	HIPRT_DEVICE EmissiveMeshAliasTableDevice sample_one_emissive_mesh(Xorshift32Generator& rng, float& out_pdf) const
	{
		int emissive_mesh_index = meshes_alias_table.sample(rng);
		out_pdf = meshes_PDFs[emissive_mesh_index];

		return get_emissive_mesh_alias_table(emissive_mesh_index);
	}
};

#endif
