/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_DIRECT_LIGHT_SAMPLING_OPTIONS_H
#define HOST_DEVICE_COMMON_DIRECT_LIGHT_SAMPLING_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

#define LSS_NO_DIRECT_LIGHT_SAMPLING 0
#define LSS_ONE_LIGHT 1
#define LSS_BSDF 2
#define LSS_MIS_LIGHT_BSDF 3
#define LSS_RIS_BSDF_AND_LIGHT 4
#define LSS_RESTIR_DI 5

#define LSS_BASE_UNIFORM 0
#define LSS_BASE_POWER 1
#define LSS_BASE_REGIR 2

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
* What direct lighting sampling strategy to use.
*
* Possible values (the prefix LSS stands for "Light Sampling strategy"):
*
*	- LSS_NO_DIRECT_LIGHT_SAMPLING
*		No direct light sampling. Emission is only gathered if rays happen to bounce into the lights.
*
*	- LSS_ONE_LIGHT
*		Samples one random light in the scene without MIS.
*		Efficient as long as there are not too many lights in the scene and no glossy surfaces
*
*  - LSS_BSDF
*		Samples lights only using a BSDF sample
*		Efficient as long as light sources in the scene are large
*
*	- LSS_MIS_LIGHT_BSDF
*		Samples one random light in the scene with MIS (Multiple Importance Sampling): light sample + BRDF sample
*
*	- LSS_RIS_BSDF_AND_LIGHT
*		Samples lights in the scene with Resampled Importance Sampling
*
*	- LSS_RESTIR_DI
*		Uses ReSTIR DI to sample direct lighting at the first bounce in the scene.
*		Later bounces use the strategy given by ReSTIR_DI_LaterBouncesSamplingStrategy
*/
#define DirectLightSamplingStrategy LSS_ONE_LIGHT

/**
* How to sample lights in the scene.
* This directly affects the 'DirectLightSamplingStrategy' strategies that sample lights
*
*	- LSS_BASE_UNIFORM
*		Lights are sampled uniformly
*
*	- LSS_BASE_POWER
*		Lights are sampled proportionally to their power
*
*	- LSS_BASE_REGIR
*		Uses ReGIR to sample lights
*		Implementation of [Rendering many lights with grid-based reservoirs, Boksansky, 2021]
*/
#define DirectLightSamplingBaseStrategy LSS_BASE_POWER

/**
 * How many light samples to take and shade per each vertex of the
 * ray's path.
 * 
 * Said otherwise, we're going to run next-event estimation that many
 * times per each intersection point along the ray.
 * 
 * This is good because this amortizes camera rays and bounce rays i.e.
 * we get better shading quality for as many camera rays and bounce rays
 * 
 * This is not supported by ReSTIR DI because this would require recomputing
 * a new reservoir = full re-run of ReSTIR = too expensive.
 * It does apply to the secondary bounces shading when using ReSTIR DI for the
 * primary bounce though.
 */ 
#define DirectLightSamplingNEESampleCount 1

/**
* If this is true, light sampling with NEE (emissive geometry & envmap) will not even
* be attempted on perfectly smooth materials (smooth glass, smooth metals, ...)
*
* This is because these materials are delta distributions and light sampling
* has no chance to give any contribution.
*
* There is no point in disabling that option, this is basically only for
* performance comparisons
*/
#define DirectLightSamplingDeltaDistributionOptimization KERNEL_OPTION_TRUE

/**
 * Whether or not to allow backfacing lights during NEE evaluation.
 * 
 * For most scenes, this is going to have no impact on visuals as lights are generally
 * watertight meshes, meaning that backfacing emissive triangles of those meshes are not visible from
 * the outside. There will thus be no visual difference but a non negligeable boost in 
 * performance/sampling quality as backfacing lights will not be sampled anymore (depending on the sampling strategy)
 */
#define DirectLightSamplingAllowBackfacingLights KERNEL_OPTION_FALSE

#endif // #ifndef __KERNELCC__

#endif
