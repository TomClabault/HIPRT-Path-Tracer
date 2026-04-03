/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/Compute/Common/DataTransforms.h"
#include "Device/includes/Compute/ParallelPrefixScanCommon.h"
#include "Device/includes/Compute/ParallelPrefixScanDecoupledLookbackBlockDescriptor.h"
#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

// Register-based warp inclusive scan
// Flag here is an unsigned int type so that it can be used in shfl instructions but its value should be either 0 or 1.
HIPRT_DEVICE DataType warp_segmented_scan_inclusive(DataType val, unsigned int flag, unsigned int& out_propagated_flag)
{
	unsigned int lane = threadIdx.x % 32;

#pragma unroll
	for (int i = 1; i <= 16; i *= 2)
	{
		DataType n	   = hippt::warp_shfl_up(val, i);
		unsigned int f = hippt::warp_shfl_up(flag, i);

		if (lane >= i)
		{
			val = flag ? val : val + n;
			flag |= f;
		}
	}

	out_propagated_flag = flag;
	return val;
}

HIPRT_DEVICE DataType block_segmented_scan_early_publish(DataType thread_input_value,
														 unsigned int flag,
														 unsigned int bid,
														 int tid,
														 bool& out_thread_needs_block_prefix,
														 bool& out_block_is_open,
														 ParallelPrefixScanDecoupledLookbackBlockDescriptor* __restrict__ block_descs)
{
	unsigned int lane_id = tid % 32;
	unsigned int warp_id = tid >> 5;

	// Per-warp scan
	bool warp_is_open = hippt::warp_shfl(flag, 0) == 0; // Whether or not the warp begins with an open flag
	unsigned int propagated_flag;
	DataType warp_prefix = warp_segmented_scan_inclusive(thread_input_value, flag, propagated_flag);

	// Last value of the last segment of the warp.
	DataType warp_total_sum = hippt::warp_shfl(warp_prefix, 31);

	// Encodes whether this warp is a homogeneous continuation of the previous warp
	unsigned int warp_flag = (hippt::warp_shfl(propagated_flag, 31) != 0) || !warp_is_open;
	// The warp is open and we have no flag up until this lane so we will have to accumulate the previous warp of the block into this lane
	unsigned int accumulate_previous_warp = warp_is_open && (propagated_flag == 0);

	// Shared memory to hold the sum of each warp
	// (Size = Max Threads / 32). Assuming max 1024 threads -> 32 warps.
	constexpr unsigned int num_warps = PARALLEL_PREFIX_SCAN_CHUNK_SIZE / 32;
	__shared__ DataType smem_warp_sums[num_warps];
	__shared__ unsigned int smem_flags[num_warps];
	__shared__ unsigned int smem_inclusive_flag_scan[num_warps];
	__shared__ bool smem_block_is_open;

	if (lane_id == 31)
	{
		// Storing the total sum of each warp in shared memory
		smem_warp_sums[warp_id] = warp_total_sum;
		smem_flags[warp_id]		= warp_flag;
	}

	__syncthreads();

	// Scan the warp sums (only warp 0 does this)
	// This calculates the base value to add to each warp
	if (warp_id == 0)
	{
		DataType my_warp_sum = 0;
		unsigned int my_flag = 0;

		if (tid < num_warps)
		{
			// Only load if the warp actually exists in this block
			my_warp_sum = smem_warp_sums[tid];
			my_flag		= smem_flags[tid];
		}

		unsigned int propagated_all_warp_flags;
		DataType inclusive_warp_sum_scan = warp_segmented_scan_inclusive(my_warp_sum, my_flag, propagated_all_warp_flags);

		// Write the inclusive scan back to smem so other warps can read their "base"
		if (tid < num_warps)
		{
			smem_warp_sums[tid]			  = inclusive_warp_sum_scan;
			smem_inclusive_flag_scan[tid] = propagated_all_warp_flags;
		}

		if (tid == num_warps - 1)
		{
			ParallelPrefixScanDecoupledLookbackBlockDescriptor new_desc;
			new_desc.set_inclusive_sum(inclusive_warp_sum_scan);

			if (bid != 0 && propagated_all_warp_flags == 0)
				new_desc.set_status(DecoupledLookbackStatus::A);
			else
				// bid 0 is always completed. We also mark blocks with flags as completed right away because they don't need to look back at previous blocks
				// since they have segment boundaries inside the block
				new_desc.set_status(DecoupledLookbackStatus::P);

			ParallelPrefixScanDecoupledLookbackBlockDescriptor::atomic_write(block_descs, bid, new_desc);

			smem_block_is_open = propagated_all_warp_flags == 0;
		}
	}

	__syncthreads();

	// A thread needs the global lookback prefix only if:
	//	- It has no flags in its own warp history (propagated_flag == 0)
	//	- All previous warps in the block were also flag-free (smem_inclusive_flag_scan[warp_id - 1] == 0)

	bool warp_open_up_to_current_lane = propagated_flag == 0;
	bool all_previous_warps_open	  = warp_id == 0 || smem_inclusive_flag_scan[warp_id - 1] == 0;
	out_thread_needs_block_prefix	  = warp_open_up_to_current_lane && all_previous_warps_open;
	out_block_is_open				  = smem_block_is_open;

	DataType warp_base = 0;
	// Add the base from previous warps to the local warp prefix
	if (warp_id > 0 && accumulate_previous_warp)
		warp_base = smem_warp_sums[warp_id - 1];

	// This thread's inclusive sum over the whole block
	return warp_prefix + warp_base;
}

GLOBAL_KERNEL_SIGNATURE(void)
ParallelSegmentedPrefixScanDecoupledLookback_Scan(const DataType* __restrict__ input,
												  const unsigned int* __restrict__ flags,
												  const unsigned int evenly_spaced_segment_size,
												  DataType* __restrict__ output,
												  ParallelPrefixScanDecoupledLookbackBlockDescriptor* __restrict__ block_descs,
												  unsigned int* __restrict__ g_global_block_index_counter,
												  unsigned int input_size)
{
	int tid = threadIdx.x;

	__shared__ unsigned int block_index;
	if (tid == 0)
		block_index = hippt::atomic_fetch_add_gpu(g_global_block_index_counter, 1u);
	__syncthreads();

	unsigned int bid		= block_index;
	unsigned int num_blocks = (input_size + PARALLEL_PREFIX_SCAN_CHUNK_SIZE - 1) / PARALLEL_PREFIX_SCAN_CHUNK_SIZE;
	if (bid >= num_blocks)
		return;

	// Input load
	unsigned int global_tid				 = bid * PARALLEL_PREFIX_SCAN_CHUNK_SIZE + tid;
	DataType thread_input_value_original = (global_tid < input_size) ? input[global_tid] : 0;
	DataType thread_input_value			 = input_value_transform(thread_input_value_original);

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

	if (tid == 0 && bid == 0)
		flag = 1; // Ensure the very first element of the array is a head / segment start

	// Inclusive block scan for this thread.
	//
	// It also internally publishes the 'A' status for the block as soon as possible.

	bool thread_needs_block_prefix;
	bool block_is_open;
	DataType inclusive_sum = block_segmented_scan_early_publish(thread_input_value, flag, bid, tid, thread_needs_block_prefix, block_is_open, block_descs);

	__shared__ DataType block_prefix;
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

			DataType val_to_add = previous_block_descriptor.get_inclusive_sum();

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

		if (tid == 0 && block_is_open)
		{
			ParallelPrefixScanDecoupledLookbackBlockDescriptor block_descriptor = block_descs[bid];
			block_descriptor.set_inclusive_sum(block_descriptor.get_inclusive_sum() + block_prefix);
			block_descriptor.set_status(DecoupledLookbackStatus::P);

			ParallelPrefixScanDecoupledLookbackBlockDescriptor::atomic_write(block_descs, bid, block_descriptor);
		}
	}

	__syncthreads();

	DataType prefix_contribution = thread_needs_block_prefix ? block_prefix : 0;

	if (global_tid < input_size)
	{
		DataType output_value = inclusive_sum - thread_input_value + prefix_contribution;
		// - thread_input_value here to get an exclusive scan output. Removing that yields an inclusive scan output
		output[global_tid] = output_value_transform(output_value);
	}
}
