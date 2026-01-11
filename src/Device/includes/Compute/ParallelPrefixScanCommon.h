/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_COMPUTE_PARALLEL_PREFIX_SCAN_COMMON_H
#define DEVICE_INCLUDES_COMPUTE_PARALLEL_PREFIX_SCAN_COMMON_H

#define PARALLEL_PREFIX_SCAN_CHUNK_SIZE 256u

#define NUMBER_OF_BANKS 32
#define LOG2_NUMBER_OF_BANKS 5
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG2_NUMBER_OF_BANKS + ((index) >> (2 * LOG2_NUMBER_OF_BANKS)))

#endif
