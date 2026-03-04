/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_PARALLEL_PREFIX_SCAN_DECOUPLED_LOOKBACK_BLOCK_DESCRIPTOR_H
#define DEVICE_INCLUDES_COMPUTE_PARALLEL_PREFIX_SCAN_DECOUPLED_LOOKBACK_BLOCK_DESCRIPTOR_H

#include "HostDeviceCommon/Maths/Math.h"

enum DecoupledLookbackStatus
{
	A, // 'inclusive_sum' available. Indicates the 'inclusive_sum' field has been recorded for the associated block.
	P, // Prefix available. Indicates the 'inclusive_sum' field has been recorded for the associated block.
	X, // Invalid. Indicates no information about the block is available to other thread groups.
};

struct ParallelPrefixScanDecoupledLookbackBlockDescriptor
{
	HIPRT_DEVICE static void atomic_write(ParallelPrefixScanDecoupledLookbackBlockDescriptor* buffer,
										  unsigned int index,
										  const ParallelPrefixScanDecoupledLookbackBlockDescriptor& value)
	{
		unsigned long long int* target_address = reinterpret_cast<unsigned long long int*>(&buffer[index]);
		unsigned long long int old_value	   = *target_address;
		unsigned long long int assumed;

		do
		{
			assumed = old_value;

			// This line doesn't compile on the CPU because it expects std::atomic<> values
			// but we're only using raw unsigned long long int values here so just guarding
			// that such that we don't get CPU compilation errors (this code is only meant
			// to be compiled on the GPU anyways)
#ifdef __KERNELCC__
			old_value = hippt::atomic_compare_exchange(target_address, assumed, *reinterpret_cast<const unsigned long long int*>(&value));
#endif
		} while (assumed != old_value);
	}

	HIPRT_DEVICE DecoupledLookbackStatus get_status() const
	{
		return (DecoupledLookbackStatus)(inclusive_sum_status >> 32);
	}

	HIPRT_DEVICE void set_status(DecoupledLookbackStatus status)
	{
		inclusive_sum_status = (inclusive_sum_status & 0x00000000FFFFFFFF) | ((unsigned long long int)status << 32);
	}

	HIPRT_DEVICE unsigned int get_inclusive_sum() const
	{
		return static_cast<unsigned int>(inclusive_sum_status & 0x00000000FFFFFFFF);
	}

	HIPRT_DEVICE void set_inclusive_sum(unsigned int inclusive_sum)
	{
		inclusive_sum_status = (inclusive_sum_status & 0xFFFFFFFF00000000) | (inclusive_sum & 0x00000000FFFFFFFF);
	}

private:
	// [S | 4 high bytes] Status of the block
	// [A | 4 low bytes] Inclusive sum of the block (and previous blocks when 'P' status)
	unsigned long long int inclusive_sum_status = 0;
};

#endif
