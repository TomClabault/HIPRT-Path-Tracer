/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_LIGHT_DISTRIBUTIONS_BUILD_SCATTER_CDF_ELEMENTS_H
#define DEVICE_KERNELS_REGIR_LIGHT_DISTRIBUTIONS_BUILD_SCATTER_CDF_ELEMENTS_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/HashGrid.h"

/**
 * This kernel is dispatched with 1 block per cell / light distribution
 */
GLOBAL_KERNEL_SIGNATURE(void)
inline ReGIR_Cell_Light_Distributions_Scatter_CDFs_Elements(const unsigned short int* __restrict__ light_distribution_sizes,
															const unsigned int* __restrict__ light_distribution_offsets,
															const float* __restrict__ prefix_scanned_CDFs_device_pointer,
															unsigned int DEBUGCDFSIZE,
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

	unsigned int global_cell_index		   = blockIdx.x + cell_offset;
	if (global_cell_index >= nb_cells_alive)
		return;

	unsigned int hash_grid_cell_index = grid_cell_alive_list[global_cell_index];
	if (hash_grid_cell_index == HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX)
		return;

	unsigned int light_distribution_size   = light_distribution_sizes[hash_grid_cell_index];
	unsigned int light_distribution_offset = light_distribution_offsets[hash_grid_cell_index];

	if ((dispatch_local_cell_index * emissive_mesh_count + light_distribution_size - 1) >= DEBUGCDFSIZE)
	{
		printf("Debug size: %u | global cell index: %u | dispatch_local_cell_index * emissive_mesh_count + light_distribution_size - 1 = %u * %u + %u - 1 = %u\n\tnb_cells_alive: %u\n", DEBUGCDFSIZE, global_cell_index, dispatch_local_cell_index, emissive_mesh_count, light_distribution_size, dispatch_local_cell_index * emissive_mesh_count + light_distribution_size - 1, nb_cells_alive);

		return;
	}

	float cell_best_contribution_sum = prefix_scanned_CDFs_device_pointer[dispatch_local_cell_index * emissive_mesh_count + light_distribution_size - 1];
	if (cell_best_contribution_sum <= 0.0f)
				return;

	float normalization_factor = 1.0f / cell_best_contribution_sum;

	for (int tid = threadIdx.x; tid < light_distribution_size; tid += blockDim.x)
	{
		float prefix_sum_contribution	= prefix_scanned_CDFs_device_pointer[dispatch_local_cell_index * emissive_mesh_count + tid];
		unsigned int contribution_index = light_distribution_offset + tid;

		// if (contribution_index > 6800 && contribution_index < 6900)
		// {
		// 	// DEBUG VARIABLES
		// 	printf("local_cell_index: %u, global_cell %u, hash grid cell index: %u, tid %u, contribution_index %u, cell_best_contribution_sum %f, "
		// 		   "prefix_sum_contribution %f, "
		// 		   "normalization_factor %f\n",
		// 		   dispatch_local_cell_index, global_cell_index, hash_grid_cell_index, tid, contribution_index, cell_best_contribution_sum,
		// 		   prefix_sum_contribution, normalization_factor);
		// 	if (!hippt::is_finite(normalization_factor))
		// 	{
		// 		printf("Cell best contrib <= 0.0f: %d\n", cell_best_contribution_sum <= 0.0f);
		// 	}
		// }
				if (contribution_index == 0)
				printf("\ttid %u, Setting %u to %d\n", tid, contribution_index, (unsigned short int)(prefix_sum_contribution * normalization_factor * 65535.0f));
		output_CDF_elements[contribution_index] = (unsigned short int)(prefix_sum_contribution * normalization_factor * 65535.0f);
	}
}

#endif
