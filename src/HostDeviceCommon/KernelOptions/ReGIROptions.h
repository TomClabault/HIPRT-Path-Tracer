/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_REGIR_OPTIONS_H
#define HOST_DEVICE_COMMON_REGIR_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

 /**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

/**
* How to sample lights in the scene for filling the ReGIR grid.
*
*	- LSS_BASE_UNIFORM
*		Lights are sampled uniformly
*
*	- LSS_BASE_POWER_AREA
*		Lights are sampled proportionally to their 'power * area'
*/
#define ReGIR_GridFillLightSamplingBaseStrategy LSS_BASE_POWER_AREA

/**
 * Light sampling technique used in case the position that we are shading is falling outside of the ReGIR grid
 * 
 * All LSS_BASE_XXX strategies are allowed except LSS_BASE_REGIR
 */
#define ReGIR_FallbackLightSamplingStrategy LSS_BASE_POWER_AREA

/**
 * Debug option to color the scene with the grid cells
 */
#define ReGIR_DisplayGridCells KERNEL_OPTION_FALSE

#endif // #ifndef __KERNELCC__

#endif
