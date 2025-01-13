/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_GMON_OPTIONS_H
#define HOST_DEVICE_COMMON_KERNEL_OPTIONS_GMON_OPTIONS_H

/**
 * Kernel options for the implementation of GMoN
 * 
 * Reference:
 * [1] [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]
 */

/**
 * Thread block size dispatched when computing the G-Median of Means per each pixel
 */
#define GMoNComputeMeansKernelThreadBlockSize 8

/**
 * How many sets to use for GMoN. M variable in the paper
 */
#define GMoNMSetsCount 5

/**
 * How many bits to use to sort the means with a radix sort
 */
#define GMoNKeysNbDigitsForRadixSort 32

#endif
