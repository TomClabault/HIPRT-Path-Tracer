/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_COMMON_OPTIONS_H
#define HOST_DEVICE_COMMON_RESTIR_COMMON_OPTIONS_H

/**
 * What MIS weights to use when resampling neighbors (temporal / spatial)
 *
 *  - RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M
 *		Very simple biased weights as described in the 2020 paper (Eq. 6).
 *		Those weights are biased because they do not account for cases where
 *		we resample a sample that couldn't have been produced by some neighbors.
 *		The bias shows up as darkening, mostly at object boundaries. In GRIS vocabulary,
 *		this type of weights can be seen as confidence weights alone c_i / sum(c_j)
 *
 *  - RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z
 *		Simple unbiased weights as described in the 2020 paper (Eq. 16 and Section 4.3)
 *		Those weights are unbiased but can have **extremely** bad variance when a neighbor being resampled
 *		has a very low target function (when the neighbor is a glossy surface for example).
 *		See Fig. 7 of the 2020 paper.
 *
 *  - RESTIR_MIS_WEIGHTS_TYPE_MIS_LIKE
 *		Unbiased weights as proposed by Eq. 22 of the paper. Way better than 1/Z in terms of variance
 *		and still unbiased.
 *
 *  - RESTIR_MIS_WEIGHTS_TYPE_MIS_GBH
 *		Unbiased MIS weights that use the generalized balance heuristic. Very good variance reduction but O(N^2) complexity,
 *		N being the number of neighbors resampled.
 *		Eq. 36 of the 2022 Generalized Resampled Importance Sampling paper.
 *
 *	- RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS (and the defensive version RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE)
 *		Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.
 *		Section 7.1.3 of "A Gentle Introduction to ReSTIR", 2023
 *
 * 	- RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO (and the defensive version RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO)
 *		A bit more variance than pairwise MIS but way more robust to temporal correlations
 *
 *		Implementation of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, Pan et al., 2024]
 *
 * - RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS (and the defensive version RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
 *		Importance sampling version of pairwise MIS that importance samples spatial neighbors based on the luminance of their sample, following "Stochastic
 *		Pairwise MIS for Unbiased Large-Kernel Reuse in Real-Time, Hedstrom et al. 2026"
 */
#define RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M						  0
#define RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z						  1
#define RESTIR_MIS_WEIGHTS_TYPE_MIS_LIKE						  2
#define RESTIR_MIS_WEIGHTS_TYPE_MIS_GBH							  3
#define RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS					  4
#define RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE			  5
#define RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO					  6
#define RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO				  7
#define RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS			  8
#define RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE 9

#define ReSTIR_VARIANT_DI 0
#define ReSTIR_VARIANT_GI 1
#define ReSTIR_VARIANT_PT 2

#define ReSTIR_SpatialDirectionalReuseBitCount 64

#endif
