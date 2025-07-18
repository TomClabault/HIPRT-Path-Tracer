/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_GMON_OPTIONS_H
#define HOST_DEVICE_COMMON_KERNEL_OPTIONS_GMON_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

/**
 * Kernel options for the implementation of GMoN
 * 
 * Reference:
 * [1] [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]
 */

 // This block is a security to make sure that we have everything defined otherwise this can lead
 // to weird behavior because of the compiler not knowing about some macros
#ifndef KERNEL_OPTION_TRUE
#error "KERNEL_OPTION_TRUE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#else
#ifndef KERNEL_OPTION_FALSE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#endif
#endif

 /**
  * Options are defined in a #ifndef __KERNELCC__ block because:
  *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
  *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
  *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
  * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
  */
#ifndef __KERNELCC__

/**
 * How many sets to use for GMoN. M variable in the paper
 */
#define GMoNMSetsCount 11

#endif // #ifndef __KERNELCC__

// The options below are not in the "#ifndef __KERNELCC__" guard because they cannot change at runtime
// so we're not passing them as options to the compiler with -D so they need to be know in the
// source file at compile time
/**
 * Thread block size dispatched when computing the G-Median of Means per each pixel
 */
#define GMoNComputeMeansKernelThreadBlockSize 8

 /**
  * How many bits to use to sort the means with a radix sort
  */
#define GMoNKeysNbDigitsForRadixSort 32

/**
 * What radix is used for the radix sort of the means
 */
#define GMoNSortRadixSize 2

#endif // #ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_GMON_OPTIONS_H
