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
#error "KERNEL_OPTION_TRUE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#else
#ifndef KERNEL_OPTION_FALSE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#endif
#endif

/**
 * The resolution downscale factor to apply for the ReGIR grid prepopulation.
 * 
 * The lower the downscale, the more effective the prepoluation but also the more costly
 */
#define ReGIR_GridPrepoluationResolutionDownscale 2

#define REGIR_PRE_INTEGRATION_ITERATIONS 64

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
#define ReGIR_GridFillTargetFunctionCosineTermLightSource KERNEL_OPTION_TRUE

/**
 * Whether or not to include the BSDF in the target function used for the resampling of the initial candidates
 * for the grid fill.
 * 
 * Helps a lot on glossy surfaces but at the first hit only
 */
#define ReGIR_GridFillTargetFunctionBSDF KERNEL_OPTION_TRUE

/**
 * Whether or not to estimate the visibility probability of samples with NEE++ during the grid fill.
 */
#define ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation KERNEL_OPTION_TRUE

/**
 * Whether or not to use a shadow ray in the target function when shading a point at path tracing time.
 * This reduces visibility noise
 */
#define ReGIR_ShadingResamplingTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * Whether or not to use NEE++ to estimate the visibility probability of the reservoir being resampled during
 * shading such that reservoirs that are likely to be occluded will have a lower resampling probability
 * 
 * This option is exclusive with ReGIR_ShadingResamplingTargetFunctionVisibility, the latter taking precedence.
 */
#define ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility KERNEL_OPTION_TRUE

/**
 * Whether or not to include the BSDF at the shading point in the resampling target function when
 * shading a point at path tracing time. This reduces shading noise at an increased computational cost.
 */
#define ReGIR_ShadingResamplingIncludeBSDF KERNEL_OPTION_TRUE

/**
 * Whether or not to incorporate BSDF samples with MIS during shading resampling.
 */
#define ReGIR_ShadingResamplingDoBSDFMIS KERNEL_OPTION_TRUE

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
 * Maximum number of steps for the linear probing in the hash table to resolve collisions
 */
#define ReGIR_LinearProbingSteps 8

/**
 * If using jittering, how many tries to perform to find a good neighbor at shading time?
 *
 * This is because with jittering, our jittered position may end up outside of the grid
 * or in an empty cell, in which case we want to retry with a differently jittered position
 * to try and find a good neighbor
 */
#define ReGIR_ShadingJitterTries 4

/**
 * Whether or not to increase the hash grid precision on surfaces that have a lower roughness
 * such that the BRDF term in the target function of the grid fill (if used at all) has a higher
 * precision and gives better results
 */
#define ReGIR_AdaptiveRoughnessGridPrecision KERNEL_OPTION_TRUE

/**
 * Debug option to color the scene with the grid cells
 */
#define ReGIR_DebugMode REGIR_DEBUG_MODE_NO_DEBUG

#endif // #ifndef __KERNELCC__

#endif
