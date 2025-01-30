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

 /**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

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
*  - RESTIR_GI_BIAS_CORRECTION_MIS_GBH
*		Unbiased MIS weights that use the generalized balance heuristic. Very good variance reduction but O(N^2) complexity,
*		N being the number of neighbors resampled.
*		Eq. 36 of the 2022 Generalized Resampled Importance Sampling paper.
*
*	- RESTIR_GI_BIAS_CORRECTION_PAIRWISE_MIS (and the defensive version)
*		Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.
*		Section 7.1.3 of "A Gentle Introduction to ReSTIR", 2023
*/
#define ReSTIR_GI_BiasCorrectionWeights RESTIR_GI_BIAS_CORRECTION_MIS_LIKE

#endif // #ifndef __KERNELCC__

#endif