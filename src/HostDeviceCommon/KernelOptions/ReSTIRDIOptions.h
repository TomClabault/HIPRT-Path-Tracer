/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_DI_OPTIONS_H
#define HOST_DEVICE_COMMON_RESTIR_DI_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRCommonOptions.h"

#define RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT  0
#define RESTIR_DI_LATER_BOUNCES_BSDF			   1
#define RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF	   2
#define RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT 3

#define RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT 64

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
 * Whether or not to use a visibility term in the target function when resampling
 * initial candidates in ReSTIR DI. *
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_InitialTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * Whether or not to use a visibility term in the target function when resampling
 * samples in ReSTIR DI. This applies to the spatial reuse pass only.
 * This option can have a good impact on quality and be worth it in terms of cost.
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_SpatialTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * Whether or not to do a visibility check at the end of the initial candidates sampling.
 * This discards reservoirs (by setting their UCW to 0.0f) whose samples are occluded.
 * This allows following ReSTIR passes (temporal and spatial) to only resample on samples
 * that are not occluded which improves quality quite a bit.
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_DoVisibilityReuse KERNEL_OPTION_TRUE

/**
 * Whether or not to use a visibility term in the MIS weights (MIS-like weights,
 * generalized balance heuristic, pairwise MIS, ...) used to remove bias when
 * resampling neighbors. An additional visibility ray will be traced for MIS-weight
 * evaluated. This effectively means for each neighbor resamples or (for each neighbor resampled)^2
 * if using the generalized balance heuristics (without pairwise-MIS)
 *
 * To guarantee unbiasedness, this needs to be true. A small amount of energy loss
 * may be observed if this value is KERNEL_OPTION_FALSE but the performance cost of the spatial
 * reuse will be reduced noticeably
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_MISWeightsUseVisibility KERNEL_OPTION_TRUE

/**
 * See the doc of the RESTIR_MIS_WEIGHTS_TYPE macros in ReSTIRCommonOptions.h for more details on the different types of MIS weights that can be used when
 * resampling spatial neighbors.
 */
#define ReSTIR_DI_MISWeightsType RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS

/**
 * What direct lighting sampling strategy to use for secondary bounces when ReSTIR DI is used for sampling the first bounce
 *
 * Possible values (the prefix LSS stands for "Light Sampling strategy"):
 *
 *	- RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT
 *		Samples one random light in the scene without MIS
 *
 *	- RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF
 *		Samples one random light in the scene with MIS (Multiple Importance Sampling): light sample + BRDF sample
 *
 *  - RESTIR_DI_LATER_BOUNCES_BSDF
 *		Samples a light using a BSDF sample.
 *		Efficient as long as the light sources in the scene are large.
 *
 *	- RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT
 *		Samples lights in the scene with Resampled Importance Sampling
 */
#define ReSTIR_DI_LaterBouncesSamplingStrategy RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT

#endif // #ifndef __KERNELCC__

#endif
