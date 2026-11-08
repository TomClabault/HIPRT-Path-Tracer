/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_PT_OPTIONS_H
#define HOST_DEVICE_COMMON_RESTIR_PT_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

// This block is a security to make sure that we have everything defined otherwise this can lead
// to weird behavior because of the compiler not knowing about some macros
#ifndef KERNEL_OPTION_TRUE
#error "KERNEL_OPTION_TRUE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#else
#ifndef KERNEL_OPTION_FALSE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#endif
#endif // #ifndef KERNEL_OPTION_TRUE

/**
 * What MIS weights to use when resampling neighbors (temporal / spatial)
 *
 *  - RESTIR_PT_MIS_WEIGHTS_TYPE_1_OVER_M
 *		Very simple biased weights as described in the 2020 paper (Eq. 6).
 *		Those weights are biased because they do not account for cases where
 *		we resample a sample that couldn't have been produced by some neighbors.
 *		The bias shows up as darkening, mostly at object boundaries. In GRIS vocabulary,
 *		this type of weights can be seen as confidence weights alone c_i / sum(c_j)
 *
 *  - RESTIR_PT_MIS_WEIGHTS_TYPE_1_OVER_Z
 *		Simple unbiased weights as described in the 2020 paper (Eq. 16 and Section 4.3)
 *		Those weights are unbiased but can have **extremely** bad variance when a neighbor being resampled
 *		has a very low target function (when the neighbor is a glossy surface for example).
 *		See Fig. 7 of the 2020 paper.
 *
 *  - RESTIR_PT_MIS_WEIGHTS_TYPE_MIS_LIKE
 *		Unbiased weights as proposed by Eq. 22 of the paper. Way better than 1/Z in terms of variance
 *		and still unbiased.
 *
 *  - RESTIR_PT_MIS_WEIGHTS_TYPE_MIS_GBH
 *		Unbiased MIS weights that use the generalized balance heuristic. Very good variance reduction but O(N^2) complexity,
 *		N being the number of neighbors resampled.
 *		Eq. 36 of the 2022 Generalized Resampled Importance Sampling paper.
 *
 *	- RESTIR_PT_MIS_WEIGHTS_TYPE_PAIRWISE_MIS (and the defensive version RESTIR_PT_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE)
 *		Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.
 *		Section 7.1.3 of "A Gentle Introduction to ReSTIR", 2023
 *
 * 	- RESTIR_PT_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO (and the defensive version RESTIR_PT_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO)
 *		A bit more variance than pairwise MIS but way more robust to temporal correlations
 *
 *		Implementation of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, Pan et al., 2024]
 *
 * - RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS (and the defensive version RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
 *		Importance sampling version of pairwise MIS that importance samples spatial neighbors based on the luminance of their sample, following "Stochastic
 *		Pairwise MIS for Unbiased Large-Kernel Reuse in Real-Time, Hedstrom et al. 2026"
 */
#define RESTIR_PT_MIS_WEIGHTS_TYPE_1_OVER_M							 0
#define RESTIR_PT_MIS_WEIGHTS_TYPE_1_OVER_Z							 1
#define RESTIR_PT_MIS_WEIGHTS_TYPE_MIS_LIKE							 2
#define RESTIR_PT_MIS_WEIGHTS_TYPE_MIS_GBH							 3
#define RESTIR_PT_MIS_WEIGHTS_TYPE_PAIRWISE_MIS						 4
#define RESTIR_PT_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE			 5
#define RESTIR_PT_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO					 6
#define RESTIR_PT_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO					 7
#define RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS			 8
#define RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE 9

#define ReSTIR_PT_SpatialDirectionalReuseBitCount 64

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
 * samples in ReSTIR PT. This applies to the spatial reuse pass only.
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_PT_SpatialTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * See the doc of the RESTIR_PT_MIS_WEIGHTS_TYPE macros at the top of this file for more details on the different types of MIS weights that can be used when
 * resampling spatial neighbors.
 */
#define ReSTIR_PT_MISWeightsType RESTIR_PT_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS

/**
 * Technique presented in [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, Pan et al., 2024]
 *
 * Helps with the pepper noise introduced by not using visibility in the spatial resampling target function
 */
#define ReSTIR_PT_DoOptimalVisibilitySampling KERNEL_OPTION_FALSE

/**
 * This is a compile time switch to enable the debug view that only outputs initial candidates to the viewport. Other debug views generally don't have compile
 * time switch but because this option can performance implications even if not selected, it's guarded by a compile time switch
 */
#define ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled KERNEL_OPTION_FALSE

#endif // #ifndef __KERNELCC__

#define ReSTIR_PT_SPMISHashGridCollisionResolutionMaxSteps 256
#define ReSTIR_PT_SPMISCDFLUTSize						   1024

#endif // #ifndef HOST_DEVICE_COMMON_RESTIR_PT_OPTIONS_H
