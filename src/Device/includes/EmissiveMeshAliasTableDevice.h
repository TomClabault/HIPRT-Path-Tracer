/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_EMISSIVE_MESH_ALIAS_TABLE_H
#define DEVICE_INCLUDES_EMISSIVE_MESH_ALIAS_TABLE_H

#include "HostDeviceCommon/Xorshift.h"

struct EmissiveMeshAliasTableDevice
{
	HIPRT_HOST_DEVICE int sample_one_triangle_power(Xorshift32Generator& rng, float& out_pdf) const
	{
		int emissive_triangle_index_within_the_mesh = rng.random_index(size);
		float probability = alias_table_probas[emissive_triangle_index_within_the_mesh];
		if (rng() > probability)
			// Picking the alias
			emissive_triangle_index_within_the_mesh = alias_table_alias[emissive_triangle_index_within_the_mesh];

		out_pdf = PDFs[emissive_triangle_index_within_the_mesh];
		return triangle_indices[emissive_triangle_index_within_the_mesh];
	}

	int* alias_table_alias = nullptr;
	float* alias_table_probas = nullptr;

	// When we sampling the alias table, we will only get an index from 0 to 'size - 1' but
	// what we want is the index of an emissive triangle in the scene so we're going to
	// index that buffer with the alias table index to get the index of the corresponding
	// emissive triangle in the scene
	int* triangle_indices = nullptr;
	// We could technically recompute the PDF that the a given triangle index is sampled
	// in the current mesh but that would require quite a lot more information such as the emissive
	// power of the triangle and its area (for power-proportional sampled triangles)
	//
	// So instead of needing all these pieces of information and having to carry them all the way
	// to this function, the PDFs are precomputed in this buffer
	float* PDFs = nullptr;

	unsigned int size = 0;
};

#endif
