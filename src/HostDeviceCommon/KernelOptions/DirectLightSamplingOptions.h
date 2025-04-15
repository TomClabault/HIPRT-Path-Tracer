/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_DIRECT_LIGHT_SAMPLING_OPTIONS_H
#define HOST_DEVICE_COMMON_DIRECT_LIGHT_SAMPLING_OPTIONS_H

#define LSS_NO_DIRECT_LIGHT_SAMPLING 0
#define LSS_ONE_LIGHT 1
#define LSS_BSDF 2
#define LSS_MIS_LIGHT_BSDF 3
#define LSS_RIS_BSDF_AND_LIGHT 4
#define LSS_RESTIR_DI 5

#define LSS_BASE_UNIFORM 0
#define LSS_BASE_POWER_AREA 1
#define LSS_BASE_REGIR 2

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
#define DirectLightSamplingStrategy LSS_MIS_LIGHT_BSDF

/**
* How to sample lights in the scene.
* This directly affects the 'DirectLightSamplingStrategy' strategies that sample lights
*
*	- LSS_BASE_UNIFORM
*		Lights are sampled uniformly
*
*	- LSS_BASE_POWER_AREA
*		Lights are sampled proportionally to their 'power * area'
*
*	- LSS_BASE_REGIR
*		Uses ReGIR to sample lights
*		Implementation of [Rendering many lights with grid-based reservoirs, Boksansky, 2021]
*/
#define DirectLightSamplingBaseStrategy LSS_BASE_REGIR
//#define DirectLightSamplingBaseStrategy LSS_BASE_POWER_AREA

/**
* Whether or not to use NEE++ features at all
*/
#define DirectLightUseNEEPlusPlus KERNEL_OPTION_FALSE	

/**
* Whether or not to use russian roulette to avoid tracing shadow rays based on the visibility
* information of NEE++
*/
#define DirectLightUseNEEPlusPlusRR KERNEL_OPTION_TRUE

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

#endif // #ifndef __KERNELCC__

#endif
