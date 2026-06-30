/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_COMPUTE_CELLS_VARIANCE_H
#define KERNELS_RESTIR_SPMIS_COMPUTE_CELLS_VARIANCE_H

#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/ReSTIR/PT/Reservoir.h"

#define PER_CELL_VARIANCE_MIN_PIXEL_COUNT 16

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
ReSTIR_SPMIS_ComputeCellsVariance(unsigned short int* cell_pixels_counters,
								  unsigned int* cell_offsets,
								  unsigned int* cell_alive_list,
								  unsigned int* pixel_indices_sorted,
								  ReSTIRPTReservoir* input_reservoirs,
								  float* out_cell_variance,
								  unsigned int size)
#else
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_SPMIS_ComputeCellsVariance(AtomicType<unsigned short int>* cell_pixels_counters,
										 unsigned int* cell_offsets,
										 unsigned int* cell_alive_list,
										 unsigned int cell_alive_count,
										 unsigned int* pixel_indices_sorted,
										 ReSTIRPTReservoir* input_reservoirs,
										 float* out_cell_variance,
										 [[maybe_unused]] unsigned int size				= 0,
										 [[maybe_unused]] unsigned int cell_alive_index = 0)
#endif
{
#ifndef __KERNELCC__
	// CPU path: iterate over all alive cells
	for (int i = 0; i < cell_alive_count; i++)
	{
		unsigned int cell_index	 = cell_alive_list[i];
		unsigned int total_count = cell_pixels_counters[cell_index];
		if (total_count < PER_CELL_VARIANCE_MIN_PIXEL_COUNT)
		{
			out_cell_variance[cell_index] = -1.0f;
			continue;
		}

		unsigned int cell_offset = cell_offsets[cell_index];
		float sum				 = 0.0f;
		float sum_sq			 = 0.0f;

		for (unsigned int j = 0; j < total_count; j++)
		{
			unsigned int pixel_index	= pixel_indices_sorted[cell_offset + j];
			ReSTIRPTReservoir reservoir = input_reservoirs[pixel_index];
			float luminance_val			= reservoir.sample.rc_vertex_incident_radiance.luminance();
			float contribution			= reservoir.UCW * reservoir.sample.target_function * luminance_val;

			sum += contribution;
			sum_sq += contribution * contribution;
		}

		float mean							   = sum / total_count;
		float variance						   = hippt::max(0.0f, sum_sq / total_count - mean * mean);
		float log_variance					   = hippt::intrin_logf(1.0f + variance);
		float relative_variance				   = variance / (mean * mean + 1e-6f);
		float variation_coefficient			   = hippt::sqrt(relative_variance);
		float variation_coefficient_normalized = variation_coefficient / (1.0f + variation_coefficient);

		out_cell_variance[cell_index] = variation_coefficient_normalized;
	}

	return;
#endif

#ifdef __KERNELCC__
	const uint32_t cell_alive_index = blockIdx.x;
#endif

	unsigned int cell_index = cell_alive_list[cell_alive_index];
	if (cell_index >= size)
		return;

	unsigned int total_count = cell_pixels_counters[cell_index];
	if (total_count < PER_CELL_VARIANCE_MIN_PIXEL_COUNT)
	{
		if (threadIdx.x == 0)
			out_cell_variance[cell_index] = -1.0f;

		return;
	}

	unsigned int cell_offset = cell_offsets[cell_index];
	unsigned int index		 = cell_offset + threadIdx.x;

	float contribution = 0.0f;
	if (threadIdx.x < total_count)
	{
		unsigned int pixel_index	= pixel_indices_sorted[index];
		ReSTIRPTReservoir reservoir = input_reservoirs[pixel_index];
		contribution				= reservoir.UCW * reservoir.sample.target_function;
	}

	float sum	 = block_reduce<1024>(contribution);
	float sum_sq = block_reduce<1024>(contribution * contribution);

	if (threadIdx.x == 0)
	{
		float mean							   = sum / total_count;
		float variance						   = hippt::max(0.0f, sum_sq / total_count - mean * mean);
		float log_variance					   = hippt::intrin_logf(1.0f + variance);
		float relative_variance				   = variance / (mean * mean + 1e-6f);
		float variation_coefficient			   = hippt::sqrt(relative_variance);
		float variation_coefficient_normalized = variation_coefficient / (1.0f + variation_coefficient);

		out_cell_variance[cell_index] = variation_coefficient_normalized;
	}
}

#endif
