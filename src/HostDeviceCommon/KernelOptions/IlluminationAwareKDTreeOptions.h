/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_OPTIONS_H
#define HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

#define ILLUMINATION_AWARE_KD_TREE_IS_LEARNING_TO_CLUSTER(nee_estimator, sampling_strategy)                                                                    \
	((sampling_strategy) == LSS_BASE_LIGHT_TREE_SG && (nee_estimator) == LSS_SG_TREE_LEARNING_TO_CLUSTER)

#define ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_NO_DEBUG							 0
#define ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_SOLID				 1
#define ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE				 2
#define ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE_AND_LOOKAHEAD 3

#define IlluminationAwareKDTreeTreeCutInitializationBlockSize 1024
#define IlluminationAwareKDTreeInitialLightCutSize			  4
#define IlluminationAwareKDTreeMaximumLightCutSize			  64
#define IlluminationAwareKDTreeLightClusteringBlockSize		  64

/**
 * Returns whether the given direct-lighting options use neural importance sampling for many lights.
 *
 * The arguments may be runtime compiler-option
 * values on the host or compile-time kernel option macros on the device.
 */
#define ILLUMINATION_AWARE_KD_TREE_IS_NISML(nee_estimator, sampling_strategy)                                                                                  \
	((sampling_strategy) == LSS_BASE_LIGHT_TREE_SG && (nee_estimator) == LSS_NEURAL_MANY_LIGHTS)

/**
 * Returns whether the given direct-lighting options use the illumination-aware KD-tree.
 *
 * The arguments may be runtime compiler-option values on the
 * host or compile-time kernel option macros on the device.
 */
#define ILLUMINATION_AWARE_KD_TREE_IS_ENABLED(nee_estimator, sampling_strategy)                                                                                \
	(ILLUMINATION_AWARE_KD_TREE_IS_NISML(nee_estimator, sampling_strategy) ||                                                                                  \
	 ILLUMINATION_AWARE_KD_TREE_IS_LEARNING_TO_CLUSTER(nee_estimator, sampling_strategy))

/**
 * Options are defined in a #ifndef __KERNELCC__ block because the GPU compiler
 * receives their values through -D compiler options.
 */
#ifndef __KERNELCC__

/**
 * Maximum depth of physical lookahead nodes created below a guiding cell.
 */
#define IlluminationAwareKDTreeMaximumLookaheadLevelCount 6

/**
 * Debug view for the illumination-aware KD-tree.
 */
#define IlluminationAwareKDTreeDebugMode				 ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_NO_DEBUG
#define IlluminationAwareKDTreeDebugRepresentativePoints KERNEL_OPTION_FALSE

#endif // #ifndef __KERNELCC__

#endif
