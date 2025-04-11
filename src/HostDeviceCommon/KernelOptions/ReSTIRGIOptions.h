/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_GI_OPTIONS_H
#define HOST_DEVICE_COMMON_RESTIR_GI_OPTIONS_H

#define RESTIR_GI_BIAS_CORRECTION_1_OVER_M 0
#define RESTIR_GI_BIAS_CORRECTION_1_OVER_Z 1
#define RESTIR_GI_BIAS_CORRECTION_MIS_LIKE 2
#define RESTIR_GI_BIAS_CORRECTION_MIS_GBH 3
#define RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS 4
#define RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE 5
#define RESTIR_GI_BIAS_CORRECTION_SYMMETRIC_RATIO 6
#define RESTIR_GI_BIAS_CORRECTION_ASYMMETRIC_RATIO 7

#define RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT 64 // CHANGE THIS ONE TO MODIFY THE NUMBER OF BITS.
#define RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT_INTERNAL (RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT > 64 ? 64 : RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT) // Just another macro for clamping to 64

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
 * samples in ReSTIR GI. This applies to the spatial reuse pass only.
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_GI_SpatialTargetFunctionVisibility KERNEL_OPTION_TRUE

/** 
 * Whether or not to include the change in BSDF at the sample point when resampling a neighbor.
 * This brings the target function closer to the integrand but at a non-negligeable performance cost.
 * 
 * Not worth it in practice, solely here for experimentation purposes
 */
#define ReSTIRGIDoubleBSDFInTargetFunction KERNEL_OPTION_FALSE

/**
* Whether or not to use a visibility term in the MIS weights (MIS-like weights,
* generalized balance heuristic, pairwise MIS, ...) used to remove bias when
* resampling neighbors. An additional visibility ray will be traced for MIS-weight
* evaluated. This effectively means for each neighbor resampled or (for each neighbor resampled)^2
* if using the generalized balance heuristics (without pairwise-MIS)
*
* To guarantee unbiasedness, this needs to be true. A small amount of energy loss
* may be observed if this value is KERNEL_OPTION_FALSE but the performance cost of the spatial
* reuse will be reduced noticeably
*
*	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
*/
#define ReSTIR_GI_BiasCorrectionUseVisibility KERNEL_OPTION_TRUE

/**
* What bias correction weights to use when resampling neighbors (temporal / spatial)
*
*  - RESTIR_GI_BIAS_CORRECTION_1_OVER_M
*		Very simple biased weights as described in the 2020 paper (Eq. 6).
*		Those weights are biased because they do not account for cases where
*		we resample a sample that couldn't have been produced by some neighbors.
*		The bias shows up as darkening, mostly at object boundaries. In GRIS vocabulary,
*		this type of weights can be seen as confidence weights alone c_i / sum(c_j)
*
*  - RESTIR_GI_BIAS_CORRECTION_1_OVER_Z
*		Simple unbiased weights as described in the 2020 paper (Eq. 16 and Section 4.3)
*		Those weights are unbiased but can have **extremely** bad variance when a neighbor being resampled
*		has a very low target function (when the neighbor is a glossy surface for example).
*		See Fig. 7 of the 2020 paper.
*
*  - RESTIR_GI_BIAS_CORRECTION_MIS_LIKE
*		Unbiased weights as proposed by Eq. 22 of the paper. Way better than 1/Z in terms of variance
*		and still unbiased.
*
*  - RESTIR_GI_BIAS_CORRECTION_MIS_GBH
*		Unbiased MIS weights that use the generalized balance heuristic. Very good variance reduction but O(N^2) complexity,
	N being the number of neighbors resampled.
*		Eq. 36 of the 2022 Generalized Resampled Importance Sampling paper.
*
*	- RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS (and the defensive version RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE)
*		Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.
*		Section 7.1.3 of "A Gentle Introduction to ReSTIR", 2023
* 
* *	- RESTIR_GI_BIAS_CORRECTION_SYMMETRIC_RATIO (and the defensive version RESTIR_GI_BIAS_CORRECTION_ASYMMETRIC_RATIO)
*		A bit more variance than pairwise MIS but way more robust to temporal correlations
* 
*		Implementation of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, Pan et al., 2024]
*/
#define ReSTIR_GI_BiasCorrectionWeights RESTIR_GI_BIAS_CORRECTION_SYMMETRIC_RATIO

/**
 * How many bits to use for the directional reuse masks
 * 
 * More bits use more VRAM but increase the precision of the directional reuse
 */
#define ReSTIR_GI_SpatialDirectionalReuseBitCount RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_BIT_COUNT_INTERNAL

/**
 * Technique presented in [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, Pan et al., 2024]
 *
 * Helps with the pepper noise introduced by not using visibility in the spatial resampling target function
 */
#define ReSTIR_GI_DoOptimalVisibilitySampling KERNEL_OPTION_FALSE

/**
 * Decoupled shading and reuse for the spatial neighbors as proposed in
 * [Rearchitecting Spatiotemporal Resampling for Production, Wyman, Panteleev, 2021]
 *
 * All spatial neighbors will be shaded if this option is true
 */
#define ReSTIR_GI_DoSpatialNeighborsDecoupledShading KERNEL_OPTION_FALSE

#endif // #ifndef __KERNELCC__

#endif
