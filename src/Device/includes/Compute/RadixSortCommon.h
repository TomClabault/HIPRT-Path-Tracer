/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_RADIX_SORT_COUNT_H
#define DEVICE_INCLUDES_COMPUTE_RADIX_SORT_COUNT_H

// The input buffer is processed in blocks of this size
#define RADIX_SORT_INPUT_CHUNK_SIZE	 1024
#define RADIX_SORT_THREADS_PER_BLOCK 256

// How many bits to sort at a time
#define RADIX_SORT_RADIX_BITS 8
#define RADIX_SORT_RADIX_MASK 0xFF

// Number of possible radix values (256 for 8 bits)
#define RADIX_SORT_RADIX_SIZE (1 << RADIX_SORT_RADIX_BITS)

#endif // #ifndef DEVICE_INCLUDES_COMPUTE_RADIX_SORT_COUNT_H
