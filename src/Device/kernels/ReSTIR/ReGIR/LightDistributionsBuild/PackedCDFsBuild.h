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
		return;

	float normalization_factor = 1.0f / cell_best_contribution_sum;

	for (int tid = threadIdx.x; tid < light_distribution_size; tid += blockDim.x)
	{
		float prefix_sum_contribution	= prefix_scanned_CDFs_device_pointer[dispatch_local_cell_index * emissive_mesh_count + tid];
		unsigned int contribution_index = light_distribution_offset + tid;

		output_CDF_elements[contribution_index] = (unsigned short int)(prefix_sum_contribution * normalization_factor * 65535.0f);
	}
}

#endif
