/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_DI_OPTIONS_H
#define HOST_DEVICE_COMMON_RESTIR_DI_OPTIONS_H

#define RESTIR_DI_BIAS_CORRECTION_1_OVER_M 0
#define RESTIR_DI_BIAS_CORRECTION_1_OVER_Z 1
#define RESTIR_DI_BIAS_CORRECTION_MIS_LIKE 2
#define RESTIR_DI_BIAS_CORRECTION_MIS_GBH 3
#define RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS 4
#define RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE 5

#define RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT 0
#define RESTIR_DI_LATER_BOUNCES_BSDF 1
#define RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF 2
#define RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT 3

#define RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT 64 // CHANGE THIS ONE TO MODIFY THE NUMBER OF BITS.
#define RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT_INTERNAL (RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT > 64 ? 64 : RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT) // Just another macro for clamping to 64

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
#define ReSTIR_DI_SpatialTargetFunctionVisibility KERNEL_OPTION_TRUE

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
#define ReSTIR_DI_BiasCorrectionUseVisibility KERNEL_OPTION_TRUE

/**
* What bias correction weights to use when resampling neighbors (temporal / spatial)
*
*  - RESTIR_DI_BIAS_CORRECTION_1_OVER_M
*		Very simple biased weights as described in the 2020 paper (Eq. 6).
*		Those weights are biased because they do not account for cases where
*		we resample a sample that couldn't have been produced by some neighbors.
*		The bias shows up as darkening, mostly at object boundaries. In GRIS vocabulary,
*		this type of weights can be seen as confidence weights alone c_i / sum(c_j)
*
*  - RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
*		Simple unbiased weights as described in the 2020 paper (Eq. 16 and Section 4.3)
*		Those weights are unbiased but can have **extremely** bad variance when a neighbor being resampled
*		has a very low target function (when the neighbor is a glossy surface for example).
*		See Fig. 7 of the 2020 paper.
*
*  - RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
*		Unbiased weights as proposed by Eq. 22 of the paper. Way better than 1/Z in terms of variance
*		and still unbiased.
*
*  - RESTIR_DI_BIAS_CORRECTION_MIS_GBH
*		Unbiased MIS weights that use the generalized balance heuristic. Very good variance reduction but O(N^2) complexity,
	N being the number of neighbors resampled.
*		Eq. 36 of the 2022 Generalized Resampled Importance Sampling paper.
*
*	- RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS (and the defensive version)
*		Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.
*		Section 7.1.3 of "A Gentle Introduction to ReSTIR", 2023
*/
#define ReSTIR_DI_BiasCorrectionWeights RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE

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

/**
* If true, lights are presampled in a pre-process pass as described in
* [Rearchitecting Spatiotemporal Resampling for Production, Wyman, Panteleev, 2021]
* https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production.
*
* This improves performance in scenes with dozens of thousands / millions of
* lights by avoiding cache trashing because of the memory random walk that
* light sampling becomes with that many lights
*/
#define ReSTIR_DI_DoLightsPresampling KERNEL_OPTION_TRUE

/**
 * How many bits to use for the directional reuse masks
 *
 * More bits use more VRAM but increase the precision of the directional reuse
 */
#define ReSTIR_DI_SpatialDirectionalReuseBitCount RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT_INTERNAL

#endif // #ifndef __KERNELCC__

#endif
