/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_REGIR_LIGHT_DISTRIBUTIONS_PACK_DISTRIBUTIONS_MESH_INDICES_H
#define DEVICE_KERNELS_REGIR_LIGHT_DISTRIBUTIONS_PACK_DISTRIBUTIONS_MESH_INDICES_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/HashGrid.h"
#include "Device/includes/ReSTIR/ReGIR/CellsLightDistributionsSoADevice.h"

GLOBAL_KERNEL_SIGNATURE(void)
inline ReGIR_LightDistributionsBuildPackDistributionsMeshIndices(
						const unsigned int* __restrict__ sorted_mesh_indices,
						const unsigned short int* __restrict__ light_distribution_sizes,
						const unsigned int* __restrict__ packed_mesh_indices_offsets,
						const unsigned int* __restrict__ grid_cell_alive_list,
						const unsigned int nb_cells_alive,
						const unsigned int cell_offset,
						const unsigned int max_cells_per_iteration,
						const unsigned int emissive_mesh_count,
						const unsigned int bits_per_mesh_index,
						ReGIRCellsLightDistributionsSoADevice::ReGIRCellsLightDistributionsMeshIndicesPackingType* __restrict__ packed_mesh_indices)
{
	using PackingType = ReGIRCellsLightDistributionsSoADevice::ReGIRCellsLightDistributionsMeshIndicesPackingType;

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

	unsigned int light_distribution_size = light_distribution_sizes[hash_grid_cell_index];
	unsigned int output_offset			 = packed_mesh_indices_offsets[hash_grid_cell_index];

	constexpr unsigned int BITS_PER_ELEMENT = sizeof(PackingType) * 8;
	for (unsigned int tid = threadIdx.x; tid < light_distribution_size; tid += blockDim.x)
	{
		unsigned int sorted_mesh_index			 = sorted_mesh_indices[dispatch_local_cell_index * emissive_mesh_count + tid];
		unsigned int element_index				 = tid * bits_per_mesh_index / BITS_PER_ELEMENT;
		unsigned int bit_offset_start_in_element = (tid * bits_per_mesh_index) % BITS_PER_ELEMENT;

		if (bit_offset_start_in_element + bits_per_mesh_index > BITS_PER_ELEMENT)
		{
			unsigned int bits_in_first_element		 = BITS_PER_ELEMENT - bit_offset_start_in_element;
			unsigned int bits_in_second_element		 = bits_per_mesh_index - bits_in_first_element;
			unsigned int bits_in_first_element_mask	 = (1 << bits_in_first_element) - 1;
			unsigned int bits_in_second_element_mask = (1 << bits_in_second_element) - 1;

			PackingType first_part	= static_cast<PackingType>(sorted_mesh_index & bits_in_first_element_mask) << bit_offset_start_in_element;
			PackingType second_part = (sorted_mesh_index >> bits_in_first_element) & bits_in_second_element_mask;

			hippt::atomic_or_gpu(&packed_mesh_indices[output_offset + element_index], first_part);
			hippt::atomic_or_gpu(&packed_mesh_indices[output_offset + element_index + 1], second_part);
		}
		else
		{
			// If the mesh index is fully contained in a single element
			PackingType bitmask = (1 << bits_per_mesh_index) - 1;
			PackingType bits	= (sorted_mesh_index & bitmask) << bit_offset_start_in_element;

			hippt::atomic_or_gpu(&packed_mesh_indices[output_offset + element_index], bits);
		}
	}
}

#endif
