/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_LEANING_TO_CLUSTER_OPTIONS_H
#define HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_LEANING_TO_CLUSTER_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

#define LearningToClusterTreeCutInitializationBlockSize 1024
#define LearningToClusterInitialLightCutSize			4

#define LEARNING_TO_CLUSTER_DEBUG_MODE_NO_DEBUG				  0
#define LEARNING_TO_CLUSTER_DEBUG_MODE_LIGHT_CUT_SIZE_HEATMAP 1

/**
 * Options are defined in a #ifndef __KERNELCC__ block because the GPU compiler
 * receives their values through -D compiler options.
 */
#ifndef __KERNELCC__

/**
 * Maximum number of nodes in a learning to cluster light cut. The light cut cannot be refined beyond that number of nodes.
 */
#define LearningToClusterMaximumLightCutSize 256

/**
 * Initialize light-cluster Q0 values using the SG node's total power instead of the view-dependent SG node importance.
 *
 * KERNEL_OPTION_TRUE for using total power, KERNEL_OPTION_FALSE for using view-dependent importance (sg_node_importance).
 */
#define LearningToClusterQ0UseTotalPower KERNEL_OPTION_FALSE

/**
 * Debug view for learning to cluster.
 */
#define LearningToClusterDebugMode LEARNING_TO_CLUSTER_DEBUG_MODE_NO_DEBUG

#endif // #ifndef __KERNELCC__

#ifdef LearningToClusterMaximumLightCutSize
static_assert(LearningToClusterMaximumLightCutSize <= 1024, "Learning to cluster maximum cut size cannot exceed 1024");
#endif

#endif
