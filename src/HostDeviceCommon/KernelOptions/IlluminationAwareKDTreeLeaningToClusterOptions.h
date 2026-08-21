/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_LEANING_TO_CLUSTER_OPTIONS_H
#define HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_LEANING_TO_CLUSTER_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

#define LearningToClusterTreeCutInitializationBlockSize 1024
#define LearningToClusterInitialLightCutSize			4
#define LearningToClusterMaximumLightCutSize			64
#define LearningToClusterLightClusteringBlockSize		64

/**
 * Options are defined in a #ifndef __KERNELCC__ block because the GPU compiler
 * receives their values through -D compiler options.
 */
#ifndef __KERNELCC__

/**
 * Initialize light-cluster Q0 values using the SG node's total power instead of the view-dependent SG node importance.
 */
#define LearningToClusterQ0UseTotalPower KERNEL_OPTION_TRUE

#endif // #ifndef __KERNELCC__

#endif
