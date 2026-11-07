/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_LIGHT_DISTRIBUTIONS_PACKED_CDFS_BUILD_H
#define DEVICE_KERNELS_REGIR_LIGHT_DISTRIBUTIONS_PACKED_CDFS_BUILD_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/HashGrid.h"

/**
 * This kernel is dispatched with 1 block per cell / light distribution
 */
GLOBAL_KERNEL_SIGNATURE(void)
inline ReGIR_LightDistributionsBuildPackedCDFsBuild(const unsigned short int* __restrict__ light_distribution_sizes,
													const unsigned int* __restrict__ light_distribution_offsets,
													const float* __restrict__ prefix_scanned_CDFs_device_pointer,
													const unsigned int* __restrict__ grid_cell_alive_list,
													const unsigned int nb_cells_alive,
													const unsigned int cell_offset,
													const unsigned int max_cells_per_iteration,
													const unsigned int emissive_mesh_count,
													unsigned short int* __restrict__ output_CDF_elements)
{
	unsigned int dispatch_local_cell_index = blockIdx.x;
	if (dispatch_local_cell_index >= max_cells_per_iteration)
		// Greater than the number of cells we can compute at once per iteration in ReGIRRenderPass.cpp
		return;

	unsigned int global_cell_index = blockIdx.x + cell_offset;
	if (global_cell_index >= nb_cells_alive)
		return;

	unsigned int hash_grid_cell_index = grid_cell_alive_list[global_cell_index];
	if (hash_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		return;

	unsigned int light_distribution_size   = light_distribution_sizes[hash_grid_cell_index];
	unsigned int light_distribution_offset = light_distribution_offsets[hash_grid_cell_index];

	float cell_best_contribution_sum = prefix_scanned_CDFs_device_pointer[dispatch_local_cell_index * emissive_mesh_count + light_distribution_size - 1];
	if (cell_best_contribution_sum <= 0.0f)
	{
		// No mesh contributes to that cell. The CDF buffer is reallocated at each build so its content is undefined: writing an all-zero
		// CDF (cdf[0] == 0 marks the cell as having no positive weight) instead of leaving whatever previously lived in that memory
		for (int tid = threadIdx.x; tid < light_distribution_size; tid += blockDim.x)
			output_CDF_elements[light_distribution_offset + tid] = 0;

		return;
	}

	float normalization_factor = 1.0f / cell_best_contribution_sum;

	// Following the CDFDeviceU16 convention: cdf[0] is a 'positive weight' marker and cdf[i] is where entry i starts, i.e. the
	// sum of the weights of the entries [0, i - 1]. 'prefix_scanned_CDFs_device_pointer' is an inclusive scan so entry i starts at
	// the inclusive scan value of entry i - 1
	for (int tid = threadIdx.x; tid < light_distribution_size; tid += blockDim.x)
	{
		unsigned int contribution_index = light_distribution_offset + tid;

		if (tid == 0)
			output_CDF_elements[contribution_index] = 65535;
		else
		{
			float entry_start = prefix_scanned_CDFs_device_pointer[dispatch_local_cell_index * emissive_mesh_count + tid - 1];

			output_CDF_elements[contribution_index] = (unsigned short int)(entry_start * normalization_factor * 65535.0f);
		}
	}
}

#endif // #ifndef DEVICE_KERNELS_REGIR_LIGHT_DISTRIBUTIONS_PACKED_CDFS_BUILD_H
