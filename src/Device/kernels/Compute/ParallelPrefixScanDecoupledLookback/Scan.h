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

			// Identify where 'P' (Prefix) status occurred
			// This tells us we don't need to look further back than this lane.
			unsigned int ballot_P = hippt::warp_ballot(active_mask, previous_block_descriptor.get_status() == DecoupledLookbackStatus::P);

			unsigned int val_to_add = previous_block_descriptor.get_inclusive_sum();

			// Filter out values we don't need
			// We want to sum values from the "closest" block (highest TID) 
			// down to the first block that reported status 'P'.
			if (ballot_P != 0)
			{
				// clz() counts leading zeros.
				// 
				// Since we map higher TIDs to "closer" blocks (bid-1 is TID 31 or active_lanes-1),
				// we want the P with the highest TID.
				// 
				// Example: P at bit 31 -> clz=0 -> highest_p_lane=31.
				int highest_p_lane = 31 - hippt::clz(ballot_P);

				// We create a mask that keeps all lanes >= highest_p_lane
				unsigned int keep_mask = 0xFFFFFFFF << highest_p_lane;

				// Zero out values from blocks "older" than the found P such that they are not
				// counted in the sum reduction that follows
				if ((keep_mask & (1u << tid)) == 0)
					val_to_add = 0;
			}

			// Instead of a serial loop, we sum all 32 lanes in log2 steps.
			// 
			// We use shfl_down because we want to accumulate everything into lower TIDs,
			// eventually landing the total sum in TID 0.
			for (int offset = 16; offset > 0; offset /= 2)
				val_to_add += hippt::warp_shfl_down(val_to_add, offset);

			// Update the block prefix
			if (tid == 0)
				block_prefix += val_to_add;

			if (ballot_P != 0)
				// We found a P, so we have the complete prefix. Stop looking back.
				base_lookback_block_index = -1;
			else
				// All were 'A', move window back by the number of lanes processed
				base_lookback_block_index -= active_lanes_count;

			// -------------------------------------------------------------------------
			// OPTIMIZATION END
			// -------------------------------------------------------------------------

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

	if (global_tid < input_size)
		output[global_tid] = smem_inclusive_block_sum[tid] + block_prefix;
}
