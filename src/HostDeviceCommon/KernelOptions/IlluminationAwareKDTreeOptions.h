/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_OPTIONS_H
#define HOST_DEVICE_COMMON_ILLUMINATION_AWARE_KD_TREE_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

/**
 * Options are defined in a #ifndef __KERNELCC__ block because the GPU compiler
 * receives their values through -D compiler options.
 */
#ifndef __KERNELCC__

/**
 * Maximum depth of physical lookahead nodes created below a guiding cell.
 */
#define IlluminationAwareKDTreeMaximumLookaheadDepth 1

#endif // #ifndef __KERNELCC__

#endif
