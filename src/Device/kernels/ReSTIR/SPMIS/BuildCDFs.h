/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_BUILD_CDFS_H
#define KERNELS_RESTIR_SPMIS_BUILD_CDFS_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_BuildCDFs(unsigned int* cell_non_zero_reservoir_counters,
					   unsigned int* cell_offsets,
					   unsigned int* cell_alive_list,
					   unsigned int* pixel_indices_sorted,
					   ReSTIRPTReservoir* input_reservoirs,
					   float* out_cdfs,
					   unsigned int size)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_BuildCDFs(AtomicType<unsigned int>* cell_non_zero_reservoir_counters,
							  unsigned int* cell_offsets,
							  unsigned int* cell_alive_list,
							  unsigned int cell_alive_count,
							  unsigned int* pixel_indices_sorted,
							  ReSTIRPTReservoir* input_reservoirs,
							  float* out_cdfs,
							  [[maybe_unused]] unsigned int size			 = 0, // Unused in CPU path, just so that intellisense is happy for the GPU path
							  [[maybe_unused]] unsigned int cell_alive_index = 0) // Unused in CPU path, just so that intellisense is happy for the GPU path
#endif
{
#ifndef __KERNELCC__
	// Completely different path for the CPU
	for (int cell_alive_index = 0; cell_alive_index < cell_alive_count; cell_alive_index++)
	{
		unsigned int cell_index		= cell_alive_list[cell_alive_index];
		unsigned int non_zero_count = cell_non_zero_reservoir_counters[cell_index];
		if (non_zero_count > 0)
		{
			unsigned int cell_offset = cell_offsets[cell_index];
			float running_sum		 = 0.0f;

			for (unsigned int i = 0; i < non_zero_count; i++)
			{
				unsigned int pixel_index = pixel_indices_sorted[cell_offset + i];

				out_cdfs[cell_offset + i] = running_sum;
				running_sum += input_reservoirs[pixel_index].UCW * input_reservoirs[pixel_index].sample.target_function * input_reservoirs[pixel_index].M;
			}

			// Normalization
			for (unsigned int i = 0; i < non_zero_count; i++)
				out_cdfs[cell_offset + i] /= running_sum;
			out_cdfs[cell_offset] = running_sum;
		}
	}

	return;
#endif

#ifdef __KERNELCC__
	const uint32_t cell_alive_index = blockIdx.x;
#endif

	unsigned int cell_index = cell_alive_list[cell_alive_index];
	if (cell_index >= size)
		return;

	unsigned int non_zero_count = cell_non_zero_reservoir_counters[cell_index];
	if (non_zero_count == 0)
		// No CDF to build
		return;

	// Loading the input value for this thread
	unsigned int index_in_cell = threadIdx.x;
	unsigned int cell_offset   = cell_offsets[cell_index];
	unsigned int index		   = cell_offset + index_in_cell;

	float pixel_importance = 0.0f;
	if (index_in_cell < non_zero_count)
	{
		unsigned int pixel_index		  = pixel_indices_sorted[index];
		ReSTIRPTReservoir input_reservoir = input_reservoirs[pixel_index];
		pixel_importance				  = input_reservoir.UCW * input_reservoir.sample.target_function * input_reservoir.M;
	}

	__syncthreads();

	float prefix_scanned = block_prefix_scan_exclusive<1024>(pixel_importance);

	__shared__ float weight_sum;
	if (index_in_cell == non_zero_count - 1)
		weight_sum = prefix_scanned + pixel_importance;

	__syncthreads();

	// Store the pixel importance in the output CDF array
	if (index_in_cell < non_zero_count)
	{
		if (index_in_cell == 0)
			// Storing the total weight sum of the cell in the first element of the CDF array for this cell, so that we can use it later to compute the PDF when
			// sampling
			out_cdfs[index] = weight_sum;
		else
			out_cdfs[index] = prefix_scanned / weight_sum;
	}
}

#endif
