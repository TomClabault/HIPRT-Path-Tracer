/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_BUILD_CDFS_H
#define KERNELS_RESTIR_SPMIS_BUILD_CDFS_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_BuildCDFs(unsigned short int* cell_non_zero_reservoir_counters,
					   unsigned int* cell_offsets,
					   unsigned int* cell_alive_list,
					   unsigned int* pixel_indices_sorted,
					   ReSTIRPTReservoir* input_reservoirs,
					   float* out_cdfs,
					   unsigned short int* cell_cdf_luts,
					   unsigned int* cell_cdf_luts_offsets,
					   unsigned int size)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_BuildCDFs(AtomicType<unsigned short int>* cell_non_zero_reservoir_counters,
							  unsigned int* cell_offsets,
							  unsigned int* cell_alive_list,
							  unsigned int cell_alive_count,
							  unsigned int* pixel_indices_sorted,
							  ReSTIRPTReservoir* input_reservoirs,
							  float* out_cdfs,
							  unsigned short int* cell_cdf_luts,
							  unsigned int* cell_cdf_luts_offsets,
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
				running_sum +=
					input_reservoirs[pixel_index].UCW * input_reservoirs[pixel_index].sample.target_function * input_reservoirs[pixel_index].confidence;
			}

			// Normalization
			for (unsigned int i = 0; i < non_zero_count; i++)
				out_cdfs[cell_offset + i] /= running_sum;
			// Writing the sum in [0], used to compute the probability at sampling time (which is neighbor_importance / sum)
			out_cdfs[cell_offset] = running_sum;

			// Now computing the LUT
			unsigned int current_cdf_index = 0;
			for (unsigned int bin = 0; bin < ReSTIR_PT_SPMISCDFLUTSize; bin++)
			{
				float bin_lower_bound = bin / static_cast<float>(ReSTIR_PT_SPMISCDFLUTSize);

				while (current_cdf_index < non_zero_count - 1 && out_cdfs[cell_offset + current_cdf_index + 1] <= bin_lower_bound)
					current_cdf_index++;

				cell_cdf_luts[cell_alive_index * ReSTIR_PT_SPMISCDFLUTSize + bin] = static_cast<unsigned short int>(current_cdf_index);
			}

			cell_cdf_luts_offsets[cell_index] = cell_alive_index * ReSTIR_PT_SPMISCDFLUTSize;
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
		pixel_importance				  = input_reservoir.UCW * input_reservoir.sample.target_function * input_reservoir.confidence;
	}

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

	// Now building the LUTs for sampling CDFs faster than a naive binary search
	if (threadIdx.x == 0)
		cell_cdf_luts_offsets[cell_index] = cell_alive_index * ReSTIR_PT_SPMISCDFLUTSize;

	__shared__ float shared_cdf[1024];
	shared_cdf[index_in_cell] = prefix_scanned / weight_sum;

	__syncthreads();

	unsigned int bin_index = index_in_cell;
	// The while loop is for supporting LUT sizes > 1024
	while (bin_index < ReSTIR_PT_SPMISCDFLUTSize)
	{
		float bin_lower_bound = bin_index / static_cast<float>(ReSTIR_PT_SPMISCDFLUTSize);

		unsigned short int left	 = 0;
		unsigned short int right = non_zero_count;
		while (left < right)
		{
			unsigned short int mid = (left + right) / 2;
			if (bin_lower_bound >= shared_cdf[mid])
				left = mid + 1;
			else
				right = mid;
		}

		left -= (left > 0) ? 1 : 0; // To get the last index that is lower than the bin lower bound

		unsigned int lut_start_index			   = cell_alive_index * ReSTIR_PT_SPMISCDFLUTSize;
		cell_cdf_luts[lut_start_index + bin_index] = static_cast<unsigned short int>(left);

		bin_index += blockDim.x;
	}
}

#endif
