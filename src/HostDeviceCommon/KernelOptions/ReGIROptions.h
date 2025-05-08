/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_REGIR_OPTIONS_H
#define HOST_DEVICE_COMMON_REGIR_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

#define REGIR_DEBUG_MODE_NO_DEBUG 0
#define REGIR_DEBUG_MODE_GRID_CELLS 1
#define REGIR_DEBUG_MODE_AVERAGE_CELL_NON_CANONICAL_RESERVOIR_CONTRIBUTION 2
#define REGIR_DEBUG_MODE_AVERAGE_CELL_CANONICAL_RESERVOIR_CONTRIBUTION 3
#define REGIR_DEBUG_MODE_REPRESENTATIVE_POINTS 4

 // This block is a security to make sure that we have everything defined otherwise this can lead
 // to weird behavior because of the compiler not knowing about some macros
#ifndef KERNEL_OPTION_TRUE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
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
* How to sample lights in the scene for filling the ReGIR grid.
*
*	- LSS_BASE_UNIFORM
*		Lights are sampled uniformly
*
*	- LSS_BASE_POWER
*		Lights are sampled proportionally to their power
*/
#define ReGIR_GridFillLightSamplingBaseStrategy LSS_BASE_POWER

/**
 * Whether or not to use a visibility term in the target function used to resample the reservoirs of the grid cells.
 * 
 * Probably too expensive to be efficient.
 */
#define ReGIR_GridFillTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * Whether or not to use a the cosine term between the direction to the light sample and the
 * representative normal of the grid cell in the target function used to resample the reservoirs of the grid cells.
 *
 * This has no effect is representative points are not being used
 */
#define ReGIR_GridFillTargetFunctionCosineTerm KERNEL_OPTION_TRUE

/**
 * Takes the cosine term at the light source (i.e. the cosine term of the geometry term) into account when
 * evaluating the target function during grid fill
 */
#define ReGIR_GridFillTargetFunctionCosineTermLightSource KERNEL_OPTION_FALSE

/**
 * Whether or not to use a shadow ray in the target function when shading a point at path tracing time.
 * This reduces visibility noise
 */
#define ReGIR_ShadingResamplingTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * Whether or not to include the BSDF at the shading point in the resampling target function when
 * shading a point at path tracing time. This reduces shading noise at an increased computational cost.
 */
#define ReGIR_ShadingResamplingIncludeBSDF KERNEL_OPTION_TRUE

/**
 * Discards reservoirs whose light samples are occluded at grid fill time.
 * 
 * This can be expensive but can also lead to substantial gains in quality
 */
#define ReGIR_DoVisibilityReuse KERNEL_OPTION_FALSE

/**
 * Light sampling technique used in case the position that we are shading is falling outside of the ReGIR grid
 * 
 * All LSS_BASE_XXX strategies are allowed except LSS_BASE_REGIR
 */
#define ReGIR_FallbackLightSamplingStrategy LSS_BASE_POWER

/**
 * Debug option to color the scene with the grid cells
 */
#define ReGIR_DebugMode REGIR_DEBUG_MODE_NO_DEBUG

#endif // #ifndef __KERNELCC__

#endif
