/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Compute/ParallelPrefixScanDecoupledLookbackBlockDescriptor.h"
#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "HostDeviceCommon/Maths/Math.h"

//#define PRINT(...) printf(__VA_ARGS__)
#define PRINT(...)

__shared__ unsigned int smem_inclusive_block_sum[PARALLEL_PREFIX_SCAN_CHUNK_SIZE];

__device__ void inclusive_prefix_block_sum(
	unsigned int block_index,
	unsigned int thread_input_value,
	const unsigned int* __restrict__ input,
	unsigned int input_size,
	unsigned int local_thread_count)
{
	// Hillis-Steele inclusive prefix block scan
	// 
	// Reference: https://gpu-primitives-course.github.io/

	int tid = threadIdx.x;
	int global_tid = block_index * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + threadIdx.x;

	if ((unsigned int)tid < local_thread_count)
		smem_inclusive_block_sum[tid] = thread_input_value;
	else
		smem_inclusive_block_sum[tid] = 0u;

	__syncthreads();

	for (int offset = 1; offset < PARALLEL_PREFIX_SCAN_CHUNK_SIZE; offset *= 2)
	{
		unsigned int val = 0;
		if (tid - offset >= 0)
			val = smem_inclusive_block_sum[tid - offset];

		__syncthreads();

		smem_inclusive_block_sum[tid] += val;

		__syncthreads();
	}
}

GLOBAL_KERNEL_SIGNATURE(void) ParallelPrefixScanDecoupledLookback_Scan(
	const unsigned int* __restrict__ input,
	unsigned int* __restrict__ output,
	ParallelPrefixScanDecoupledLookbackBlockDescriptor* __restrict__ block_descs,
	unsigned int* __restrict__ g_global_block_index_counter,
	unsigned int input_size)
{
	int tid = threadIdx.x;

	__shared__ unsigned int block_index;
	if (tid == 0)
		block_index = hippt::atomic_fetch_add(g_global_block_index_counter, 1u);
	__syncthreads();

	unsigned int num_blocks = (input_size + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE;
	if (block_index >= num_blocks)
		return;

	unsigned int global_tid = block_index * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + threadIdx.x;
	unsigned int bid = block_index;

	unsigned int thread_input_value = 0;
	if (global_tid < input_size)
		thread_input_value = input[global_tid];

	unsigned int local_thread_count = hippt::min(PARALLEL_PREFIX_SCAN_CHUNK_SIZE, input_size - PARALLEL_PREFIX_SCAN_CHUNK_SIZE * block_index);
	inclusive_prefix_block_sum(block_index, thread_input_value, input, input_size, local_thread_count);

	if (tid == 0)
	{
		ParallelPrefixScanDecoupledLookbackBlockDescriptor new_desc;

		new_desc.set_inclusive_sum(smem_inclusive_block_sum[local_thread_count - 1]); // Will be set later
		if (bid == 0)
			// First block, no prefix to add
			new_desc.set_status(DecoupledLookbackStatus::P);
		else
			new_desc.set_status(DecoupledLookbackStatus::A);

		ParallelPrefixScanDecoupledLookbackBlockDescriptor::atomic_write(block_descs, bid, new_desc);
	}

	__syncthreads();

	// temp_sems has been filled with the inclusive prefix scan but this kernel is for exclusive prefix scan
	// so we need to substract the input value to go from inclusive to exclusive
	smem_inclusive_block_sum[tid] -= thread_input_value;

	__shared__ unsigned int block_prefix;
	__shared__ int base_lookback_block_index;
	if (tid == 0)
	{
		block_prefix = 0;
		base_lookback_block_index = bid - 1;
	}

	__syncthreads();

	if (tid < bid && tid <= hippt::warp_size() - 1)
	{
		while (base_lookback_block_index >= 0)
		{
			ParallelPrefixScanDecoupledLookbackBlockDescriptor previous_block_descriptor;
			previous_block_descriptor.set_status(DecoupledLookbackStatus::X);

			unsigned int active_mask = hippt::warp_activemask();
			int active_lanes_count = hippt::popc(active_mask);

			// Wait until the prefix is available
			int lookback_block_index = base_lookback_block_index - (active_lanes_count - 1 - tid);
			while (previous_block_descriptor.get_status() == DecoupledLookbackStatus::X && lookback_block_index >= 0)
			{
				// Volatile read is needed here such that the compiler doesn't optimize the
				// 64b read below, which could lead to tearing or incoherent reads for example
				unsigned long long int volatile_read_value = *reinterpret_cast<volatile unsigned long long int*>(&block_descs[lookback_block_index]);

				previous_block_descriptor = *reinterpret_cast<ParallelPrefixScanDecoupledLookbackBlockDescriptor*>(&volatile_read_value);
			};

			hippt::syncwarp(0xFFFFFFFF);

			// Substracting the number of threads that participated in the lookback
			base_lookback_block_index -= active_lanes_count;
				if (bid == 2 && tid == 0)
					PRINT("TID %d analyze block %d found status %s ; sum: %u\n", tid, lookback_block_index, (previous_block_descriptor.get_status() == DecoupledLookbackStatus::P) ? "P" : ((previous_block_descriptor.get_status() == DecoupledLookbackStatus::A) ? "A" : "X"), previous_block_descriptor.get_inclusive_sum());

			unsigned int ballot_status_P = hippt::warp_ballot(0xFFFFFFFF, previous_block_descriptor.get_status() == DecoupledLookbackStatus::P);
			unsigned int ballot_status_A = hippt::warp_ballot(0xFFFFFFFF, previous_block_descriptor.get_status() == DecoupledLookbackStatus::A);

			if (bid == 2 && tid == 0)
				PRINT("Ballot P: %u | Ballot A: %u | looking at bid: %d\n", ballot_status_P, ballot_status_A, lookback_block_index);

			// The thread 0 analyzes the ballots to determine what to add to the block prefix
			//if (tid == 0)
			{
				for (int i = active_lanes_count - 1; i >= 0; i--)
				{
					if (ballot_status_A & (1 << i))
					{
						if (bid == 2 && tid == 0)
							PRINT("Found A at TID %d. Adding %u\n", i, hippt::warp_shfl(previous_block_descriptor.get_inclusive_sum(), i));

						unsigned int other_block_inclusive_sum = hippt::warp_shfl(previous_block_descriptor.get_inclusive_sum(), i);
						if (tid == 0)
							block_prefix += other_block_inclusive_sum;
					}
					else if (ballot_status_P & (1 << i))
					{
						if (bid == 2 && tid == 0)
							PRINT("Found P at TID %d. Adding %u\n", i, hippt::warp_shfl(previous_block_descriptor.get_inclusive_sum(), i));
						unsigned int other_block_inclusive_sum = hippt::warp_shfl(previous_block_descriptor.get_inclusive_sum(), i);
						if (tid == 0)
							block_prefix += other_block_inclusive_sum;

						// We've found a 'P' status, we can stop looking back
						base_lookback_block_index = -1;

						if (bid == 2 && tid == 0)
							PRINT("Breaking\n");

						break;
					}
					else
						if (bid == 2 && tid == 0)
							PRINT("------------------- WHAT\n");
				}
			}

			hippt::syncwarp(0xFFFFFFFF);
		}

		if (tid == 0)
		{
			ParallelPrefixScanDecoupledLookbackBlockDescriptor block_descriptor = block_descs[bid];
			block_descriptor.set_inclusive_sum(block_descriptor.get_inclusive_sum() + block_prefix);
			block_descriptor.set_status(DecoupledLookbackStatus::P);

			ParallelPrefixScanDecoupledLookbackBlockDescriptor::atomic_write(block_descs, bid, block_descriptor);
		}
	}

	__syncthreads();

	/*if (input_size == 699 && tid == 0)
	{
		for (int i = 0; i < 32; i++)
			PRINT("%u; input: %u\n", smem_inclusive_block_sum[i], input[i]);

		PRINT("\n");
	}*/

	if (global_tid < input_size)
		output[global_tid] = smem_inclusive_block_sum[tid] + block_prefix;
}
