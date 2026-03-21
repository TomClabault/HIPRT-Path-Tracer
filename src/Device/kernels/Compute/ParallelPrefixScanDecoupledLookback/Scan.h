/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/Common/WarpBlockScan.h"
#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Device/includes/Compute/ParallelPrefixScanDecoupledLookbackBlockDescriptor.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

HIPRT_DEVICE unsigned int block_scan_early_publish(unsigned int thread_input_value,
												   unsigned int bid,
												   int tid,
												   ParallelPrefixScanDecoupledLookbackBlockDescriptor* __restrict__ block_descs)
{
	// Per-warp scan
	unsigned int warp_prefix = warp_scan_inclusive(thread_input_value);

	// The last lane of each warp holds the sum for that warp
	unsigned int lane	 = tid % 32;
	unsigned int warp_id = tid / 32;

	// Shared memory to hold the sum of each warp
	// (Size = Max Threads / 32). Assuming max 1024 threads -> 32 warps.
	__shared__ unsigned int smem_warp_sums[PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 32];

	if (lane == 31)
		// Storing the total sum of each warp in shared memory
		smem_warp_sums[warp_id] = warp_prefix;

	__syncthreads();

	// Scan the warp sums (only warp 0 does this)
	// This calculates the base value to add to each warp
	unsigned int warp_base = 0;
	if (warp_id == 0)
	{
		unsigned int my_warp_sum = 0;

		if (tid < (PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 32))
			// Only load if the warp actually exists in this block
			my_warp_sum = smem_warp_sums[tid];

		unsigned int inclusive_warp_sum_scan = warp_scan_inclusive(my_warp_sum);

		// Write the inclusive scan back to smem so other warps can read their "base"
		// Note: We shift by 1 index effectively, because Warp N needs the sum of Warps 0..N-1
		// We can store it directly and handle the shift on read.
		smem_warp_sums[tid] = inclusive_warp_sum_scan;

		// The last active thread in Warp 0 holds the sum of ALL warps (the total block sum).
		//
		// We can publish this immediately without having to wait for other warps of the block to finish,
		unsigned int num_warps = PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 32;
		if (tid == num_warps - 1)
		{
			ParallelPrefixScanDecoupledLookbackBlockDescriptor new_desc;
			new_desc.set_inclusive_sum(inclusive_warp_sum_scan);

			if (bid == 0)
				new_desc.set_status(DecoupledLookbackStatus::P);
			else
				new_desc.set_status(DecoupledLookbackStatus::A);

			ParallelPrefixScanDecoupledLookbackBlockDescriptor::atomic_write(block_descs, bid, new_desc);
		}
	}

	__syncthreads();

	// Add the base from previous warps to the local warp prefix
	if (warp_id > 0)
		warp_base = smem_warp_sums[warp_id - 1];

	// This thread's inclusive sum over the whole block
	return warp_prefix + warp_base;
}

GLOBAL_KERNEL_SIGNATURE(void)
ParallelPrefixScanDecoupledLookback_Scan(const unsigned int* __restrict__ input,
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

	unsigned int bid		= block_index;
	unsigned int num_blocks = (input_size + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE;
	if (bid >= num_blocks)
		return;

	// Input load
	unsigned int global_tid			= bid * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + tid;
	unsigned int thread_input_value = (global_tid < input_size) ? input[global_tid] : 0;

	// Inclusive block scan for this thread.
	//
	// It also internally publishes the 'A' status for the block as soon as possible.
	unsigned int inclusive_sum = block_scan_early_publish(thread_input_value, bid, tid, block_descs);

	__syncthreads();

	__shared__ unsigned int block_prefix;
	__shared__ int base_lookback_block_index;
	if (tid == 0)
	{
		block_prefix			  = 0;
		base_lookback_block_index = bid - 1;
	}

	__syncthreads();

	// Decoupled lookback to get the block prefix. Warp-wide lookback.
	if (tid < bid && tid <= hippt::warp_size() - 1)
	{
		while (base_lookback_block_index >= 0)
		{
			ParallelPrefixScanDecoupledLookbackBlockDescriptor previous_block_descriptor;
			previous_block_descriptor.set_status(DecoupledLookbackStatus::X);

			unsigned int active_mask = hippt::warp_activemask();
			int active_lanes_count	 = hippt::popc(active_mask);

			// Wait until the prefix is available
			int lookback_block_index = base_lookback_block_index - (active_lanes_count - 1 - tid);
			while (previous_block_descriptor.get_status() == DecoupledLookbackStatus::X && lookback_block_index >= 0)
			{
				// Volatile read is needed here such that the compiler doesn't optimize the
				// 64b read below, which could lead to tearing or incoherent reads for example
				unsigned long long int volatile_read_value = *reinterpret_cast<volatile unsigned long long int*>(&block_descs[lookback_block_index]);

				previous_block_descriptor = *reinterpret_cast<ParallelPrefixScanDecoupledLookbackBlockDescriptor*>(&volatile_read_value);
			};

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
		// - thread_input_value here to get an exclusive scan output. Removing that yields an inclusive scan output
		output[global_tid] = inclusive_sum - thread_input_value + block_prefix;
}
