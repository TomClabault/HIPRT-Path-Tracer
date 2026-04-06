/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_RESTIR_REGIR_COMPUTE_CELLS_LIGHT_DISTRIBUTION_SIZES_H
#define DEVICE_KERNELS_RESTIR_REGIR_COMPUTE_CELLS_LIGHT_DISTRIBUTION_SIZES_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

GLOBAL_KERNEL_SIGNATURE(void)
inline ReGIR_Compute_Cells_Light_Distribution_Sizes(const float* __restrict__ sum_all_contributions_per_cell,
													const float* __restrict__ sum_best_contributions_per_cell,
													const float* __restrict__ light_distribution_CDFs_per_cell,
													unsigned short int* __restrict__ out_sizes_buffer,
													const unsigned int* __restrict__ grid_cell_alive_list,
													unsigned int dispatch_size,
													unsigned int cell_offset,
													unsigned int emissive_mesh_count_per_cell,
													float light_distribution_incoming_light_energy_target,
													unsigned int non_compacted_effective_light_distribution_size)
{
	unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= dispatch_size)
		return;

	unsigned int cell_index			  = thread_index / emissive_mesh_count_per_cell;
	unsigned int hash_grid_cell_index = grid_cell_alive_list[cell_index + cell_offset];
	float sum_all_contributions		  = sum_all_contributions_per_cell[cell_index];
	float sum_best_contributions	  = sum_best_contributions_per_cell[cell_index];

	unsigned int thread_index_in_cell = thread_index % emissive_mesh_count_per_cell;
	if (sum_all_contributions == 0.0f)
	{
		if (thread_index_in_cell == 0)
			out_sizes_buffer[hash_grid_cell_index] = emissive_mesh_count_per_cell;

		return;
	}

	if (sum_best_contributions / sum_all_contributions * 100.0f < light_distribution_incoming_light_energy_target)
	{
		// If even with the maximum amount of lights allowed in the light distribution, we're not covering
		// the target amount of incoming radiance, then we're going to use the full size of the light
		// distribution
		if (thread_index_in_cell == 0)
			out_sizes_buffer[hash_grid_cell_index] = non_compacted_effective_light_distribution_size;
	}
	else
	{
		// If the light distribution is covering more than necessary, compute just the right size
		// such that we cover just the right amount of the total incoming radiance
		float accumulated_contribution = light_distribution_CDFs_per_cell[thread_index];
		if (accumulated_contribution / sum_all_contributions * 100.0f >= light_distribution_incoming_light_energy_target)
		{
			unsigned int contribution_index = thread_index % emissive_mesh_count_per_cell;

			hippt::atomic_min_gpu(&out_sizes_buffer[hash_grid_cell_index], (unsigned short int)(contribution_index + 1));
		}
	}
}

#endif
