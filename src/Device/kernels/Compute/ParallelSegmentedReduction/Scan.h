/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_COMPUTE_PARALLEL_SEGMENTED_REDUCTION_SCAN_H
#define DEVICE_KERNELS_COMPUTE_PARALLEL_SEGMENTED_REDUCTION_SCAN_H

#include "Device/includes/Compute/Common/DataTransforms.h"
#include "Device/includes/Compute/Common/WarpBlockReduce.h"
#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/Compute/ParallelReductionCommon.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

HIPRT_DEVICE unsigned int load_flag(const unsigned int* __restrict__ flags,
									unsigned int global_tid,
									unsigned int input_size,
									unsigned int evenly_spaced_segment_size)
{
	unsigned int warp_id = global_tid >> 5;
	unsigned int lane_id = global_tid & 31;
	unsigned int flag;
	if (evenly_spaced_segment_size > 0)
		// If we have evenly spaced segments, we can calculate the flag without reading from memory by checking if the global thread ID is a multiple of the
		// segment size
		flag = (global_tid < input_size) && (global_tid % evenly_spaced_segment_size == 0) ? 1 : 0;
	else
		// Otherwise, we read the flag from memory. Each unsigned int in 'flags' contains 32 flags for 32 consecutive threads, so we need to extract the
		// relevant bit.
		flag = (global_tid < input_size) ? flags[warp_id] & (1u << lane_id) : 0;

	if (global_tid == 0)
		flag = 1; // Ensure the very first element of the array is a head / segment start

	return flag;
}

GLOBAL_KERNEL_SIGNATURE(void)
ParallelSegmentedReduction_Reduce(const InputDataType* __restrict__ input,
								  const unsigned int* __restrict__ flags,
								  const unsigned int* __restrict__ segment_ids,
								  const unsigned int evenly_spaced_segment_size,
								  const unsigned int segment_length,
								  OutputDataType* __restrict__ output,
								  unsigned int input_size)
{
	int tid = threadIdx.x;

	// Input load
	unsigned int bid						  = blockIdx.x;
	unsigned int global_tid					  = bid * PARALLEL_REDUCTION_CHUNK_SIZE + tid;
	InputDataType thread_input_value_original = (global_tid < input_size) ? input[global_tid] : 0;
	TransformedDataType thread_input_value	  = ComputeDataTransforms::input_value_transform(thread_input_value_original, global_tid);
	if (segment_length > 0 && evenly_spaced_segment_size > 0)
	{
		unsigned int index_in_segment = global_tid % evenly_spaced_segment_size;
		if (index_in_segment >= segment_length)
			thread_input_value = 0;
	}

#define NAIVE_ATOMIC		   0
#define SHARED_MEM_ATOMIC	   1
#define BLOCK_SEGMENTED_REDUCE 2

#define COMPUTE_VARIANT NAIVE_ATOMIC

#if COMPUTE_VARIANT == NAIVE_ATOMIC

	unsigned int segment_id;
	if (evenly_spaced_segment_size == 0)
		segment_id = (global_tid < input_size) ? segment_ids[global_tid] : 0;
	else
		segment_id = global_tid / evenly_spaced_segment_size;
	if (global_tid < input_size)
		hippt::atomic_fetch_add_gpu(&output[segment_id], ComputeDataTransforms::output_value_transform(thread_input_value, global_tid));

#elif COMPUTE_VARIANT == SHARED_MEM_ATOMIC // #if COMPUTE_VARIANT == NAIVE_ATOMIC

	unsigned int segment_id_ = (global_tid < input_size) ? segment_ids[global_tid] : 0;

	__shared__ DataType shared_mem_accumulations[PARALLEL_REDUCTION_CHUNK_SIZE];
	__shared__ unsigned int first_segment_id;
	if (tid == 0)
		first_segment_id = segment_id_;
	shared_mem_accumulations[tid] = 0;

	__syncthreads();

	hippt::atomic_fetch_add_gpu(&shared_mem_accumulations[segment_id_ - first_segment_id], thread_input_value);

	__syncthreads();

	unsigned int next_flag = load_flag(flags, global_tid + 1, input_size, evenly_spaced_segment_size);

	bool end_of_segment		   = next_flag != 0;
	bool last_element_of_block = (tid == PARALLEL_REDUCTION_CHUNK_SIZE - 1) || (global_tid == input_size - 1);
	bool needs_to_accumulate   = end_of_segment || last_element_of_block;

	if (needs_to_accumulate && global_tid < input_size)
		hippt::atomic_fetch_add_gpu(&output[segment_id_],
									ComputeDataTransforms::output_value_transform(shared_mem_accumulations[segment_id_ - first_segment_id], global_tid));

#elif COMPUTE_VARIANT == BLOCK_SEGMENTED_REDUCE // #if COMPUTE_VARIANT == NAIVE_ATOMIC

	unsigned int flag = load_flag(flags, global_tid, input_size, evenly_spaced_segment_size);

	DataType reduced = block_segmented_reduce<PARALLEL_REDUCTION_CHUNK_SIZE>(thread_input_value, flag, tid);

	if (global_tid < input_size)
	{
		unsigned int next_flag = load_flag(flags, global_tid + 1, input_size, evenly_spaced_segment_size);

		bool end_of_segment		   = next_flag != 0;
		bool last_element_of_block = (tid == PARALLEL_REDUCTION_CHUNK_SIZE - 1) || (global_tid == input_size - 1);
		bool needs_to_accumulate   = end_of_segment || last_element_of_block;

		if (needs_to_accumulate)
		{
			unsigned int segment_id = (global_tid < input_size) ? segment_ids[global_tid] : 0;

			hippt::atomic_fetch_add_gpu(&output[segment_id], ComputeDataTransforms::output_value_transform(reduced, global_tid));
		}
	}

#endif // #if COMPUTE_VARIANT == NAIVE_ATOMIC
}

#endif // #ifndef DEVICE_KERNELS_COMPUTE_PARALLEL_SEGMENTED_REDUCTION_SCAN_H
