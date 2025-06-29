/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_NEE_PLUS_PLUS_OPTIONS_H
#define HOST_DEVICE_COMMON_NEE_PLUS_PLUS_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

// This block is a security to make sure that we have everything defined otherwise this can lead
// to weird behavior because of the compiler not knowing about some macros
#ifndef KERNEL_OPTION_TRUE
#error "KERNEL_OPTION_TRUE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#else
#ifndef KERNEL_OPTION_FALSE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#endif
#endif

#define NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG 0
#define NEE_PLUS_PLUS_DEBUG_MODE_GRID_CELLS 1

/**
 * The resolution downscale factor to apply for the ReGIR grid prepopulation.
 *
 * The lower the downscale, the more effective the prepoluation but also the more costly
 */
#define NEEPlusPlus_GridPrepoluationResolutionDownscale 2

/**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

/**
* Whether or not to use NEE++ features at all
*/
#define DirectLightUseNEEPlusPlus KERNEL_OPTION_FALSE

/**
* Whether or not to use russian roulette to avoid tracing shadow rays based on the visibility
* information of NEE++
*/
#define DirectLightUseNEEPlusPlusRR KERNEL_OPTION_FALSE

/**
* This a debug option to visualize shadow rays discarded by the NEE++ russian roulette
*/
#define DirectLightNEEPlusPlusDisplayShadowRaysDiscarded KERNEL_OPTION_FALSE

/**
* When using the 'DirectLightNEEPlusPlusDisplayShadowRaysDiscarded' kernel options
* for displaying in the viewport where shadow rays were discarded, this parameter is used
* to determine at what bounce in the scene we should display the shadow ray discarded or not
*
* 0 is the first hit
*/
#define DirectLightNEEPlusPlusDisplayShadowRaysDiscardedBounce 0

/**
 * Maximum number of steps for the linear probing of the NEE++ hash grid
 */
#define NEEPlusPlus_LinearProbingSteps 4

/**
 * Debug mode for displaying some debug infos about NEE++
 */
#define NEEPlusPlusDebugMode NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG

#endif

#endif
