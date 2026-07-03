/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_PARALLEL_PREFIX_SCAN_DECOUPLED_LOOKBACK_BLOCK_DESCRIPTOR_H
#define DEVICE_INCLUDES_COMPUTE_PARALLEL_PREFIX_SCAN_DECOUPLED_LOOKBACK_BLOCK_DESCRIPTOR_H

#include "Device/includes/Compute/Common/KernelDataType.h"
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

			old_value = hippt::atomic_compare_exchange_gpu(target_address, assumed, *reinterpret_cast<const unsigned long long int*>(&value));
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

	HIPRT_DEVICE TransformedDataType get_inclusive_sum() const
	{
		uint32_t bits = static_cast<uint32_t>(inclusive_sum_status & 0x00000000FFFFFFFF);

		TransformedDataType inclusive_sum;
		memcpy(&inclusive_sum, &bits, sizeof(TransformedDataType));
		return inclusive_sum;
	}

	HIPRT_DEVICE void set_inclusive_sum(TransformedDataType inclusive_sum)
	{
		uint32_t bits = 0;
		memcpy(&bits, &inclusive_sum, sizeof(TransformedDataType));

		inclusive_sum_status = (inclusive_sum_status & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(bits);
	}

private:
	// [S | 4 high bytes] Status of the block
	// [A | 4 low bytes] Inclusive sum of the block (and previous blocks when 'P' status)
	unsigned long long int inclusive_sum_status = 0;
};

#endif
