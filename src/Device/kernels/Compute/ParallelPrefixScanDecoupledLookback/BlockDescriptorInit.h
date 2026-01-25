/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Compute/ParallelPrefixScanDecoupledLookbackBlockDescriptor.h"

GLOBAL_KERNEL_SIGNATURE(void) ParallelPrefixScanDecoupledLookback_BlockDescriptorInit(
	ParallelPrefixScanDecoupledLookbackBlockDescriptor* block_descs,
	unsigned int descriptor_count,
	unsigned int* g_global_block_index_counter)
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_tid >= descriptor_count)
		return;

	ParallelPrefixScanDecoupledLookbackBlockDescriptor empty_block_desc;
	empty_block_desc.set_inclusive_sum(0);
	empty_block_desc.set_status(X);

	ParallelPrefixScanDecoupledLookbackBlockDescriptor::atomic_write(block_descs, global_tid, empty_block_desc);

	if (global_tid == 0)
		g_global_block_index_counter[0] = 0;
}
