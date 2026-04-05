/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

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
ParallelSegmentedReduction_Reduce(const DataType* __restrict__ input,
								  const unsigned int* __restrict__ flags,
								  const unsigned int* __restrict__ segment_ids,
								  const unsigned int evenly_spaced_segment_size,
								  DataType* __restrict__ output,
								  unsigned int input_size)
{
	int tid = threadIdx.x;

	// Input load
	unsigned int bid					 = blockIdx.x;
	unsigned int global_tid				 = bid * PARALLEL_REDUCTION_CHUNK_SIZE + tid;
	DataType thread_input_value_original = (global_tid < input_size) ? input[global_tid] : 0;
	DataType thread_input_value			 = ComputeDataTransforms::input_value_transform(thread_input_value_original, global_tid);

	unsigned int flag = load_flag(flags, global_tid, input_size, evenly_spaced_segment_size);

	DataType reduced = block_segmented_reduce<PARALLEL_REDUCTION_CHUNK_SIZE>(thread_input_value, flag, tid);

	unsigned int segment_id = (global_tid < input_size) ? segment_ids[global_tid] : 0;
	if (global_tid < input_size)
	{
		unsigned int next_flag = load_flag(flags, global_tid + 1, input_size, evenly_spaced_segment_size);

		bool end_of_segment = next_flag != 0;
		bool last_element	= global_tid == input_size - 1;

		hippt::atomic_fetch_add_gpu(&output[segment_id], ComputeDataTransforms::output_value_transform(reduced, global_tid));
	}

	// Output store
	if (global_tid < input_size)
		output[global_tid] = ComputeDataTransforms::output_value_transform(reduced, global_tid);

	__shared__ DataType lane_input_values[PARALLEL_REDUCTION_CHUNK_SIZE];
	__shared__ DataType lane_reduced_values[PARALLEL_REDUCTION_CHUNK_SIZE];
	__shared__ unsigned int lane_flags[PARALLEL_REDUCTION_CHUNK_SIZE];

	if (tid < PARALLEL_REDUCTION_CHUNK_SIZE)
	{
		lane_input_values[tid]	 = thread_input_value;
		lane_reduced_values[tid] = reduced;
		lane_flags[tid]			 = flag;
	}

	__syncthreads();

	unsigned int warp_id = global_tid >> 5;
	unsigned int lane_id = global_tid & 31;
	if (bid == 0 && warp_id == 0 && lane_id == 0)
	{
		DataType reference_reduced_values[PARALLEL_REDUCTION_CHUNK_SIZE];

		// Running a single threaded reference segmented reduction to compare with the warp segmented reduction result for debugging purposes
		DataType running_sum = 0;
		for (int i = 0; i < PARALLEL_REDUCTION_CHUNK_SIZE; i++)
		{
			if (lane_flags[i] != 0)
				running_sum = lane_input_values[i];
			else
				running_sum = OperatorSum<DataType>::apply(running_sum, lane_input_values[i]);
			reference_reduced_values[i] = i >= input_size ? 0 : running_sum;
		}

		// Comparing the results
		bool missed = false;
		for (int i = 0; i < PARALLEL_REDUCTION_CHUNK_SIZE; i++)
		{
			if (reference_reduced_values[i] != lane_reduced_values[i])
			{
				printf("Mismatch at lane %d: expected %u, got %u\n", i, reference_reduced_values[i], lane_reduced_values[i]);
				missed = true;

				break;
			}
		}

		// Printing all warps input values and flags, 32 by 32
		if (missed)
		{
			// Printing the reference, 32 by 32
			printf("Ref:\n");
			for (int w = 0; w < PARALLEL_REDUCTION_CHUNK_SIZE / 32; w++)
			{
				printf("%4d - %4d [", w * 32, (w + 1) * 32);
				for (int i = 0; i < 32; i++)
					printf("%2u, ", reference_reduced_values[w * 32 + i]);
				printf("]\n");
			}

			// Printing the warp segmented reduction result, 32 by 32
			printf("Res:\n");
			for (int w = 0; w < PARALLEL_REDUCTION_CHUNK_SIZE / 32; w++)
			{
				printf("%4d - %4d [", w * 32, (w + 1) * 32);
				for (int i = 0; i < 32; i++)
					printf("%2u, ", lane_reduced_values[w * 32 + i]);
				printf("]\n");
			}

			printf("Input values and flags:\n");
			for (int w = 0; w < PARALLEL_REDUCTION_CHUNK_SIZE / 32; w++)
			{
				printf("%4d - %4d Values [", w * 32, (w + 1) * 32);
				for (int i = 0; i < 32; i++)
					printf("%2u, ", lane_input_values[w * 32 + i]);
				printf("]\n");
			}

			for (int w = 0; w < PARALLEL_REDUCTION_CHUNK_SIZE / 32; w++)
			{
				printf("%4d - %4d  Flags [", w * 32, (w + 1) * 32);
				for (int i = 0; i < 32; i++)
					printf("%2u, ", lane_flags[w * 32 + i] != 0);
				printf("]\n");
			}
		}
	}
}
