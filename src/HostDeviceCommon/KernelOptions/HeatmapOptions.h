/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_HEATMAP_OPTIONS_H
#define HOST_DEVICE_COMMON_KERNEL_OPTIONS_HEATMAP_OPTIONS_H

#include "Device/includes/Heatmap.h"

/**
 * Heatmaps selected by debug views that visualize a normalized scalar value.
 *
 * The GPU compiler receives the value through a -D compiler option.
 */
#ifndef __KERNELCC__
#define LearningToClusterDebugModeHeatmapIndex HEATMAP_INDEX_BLUE_GREEN_RED
#define NISMLDebugModeHeatmapIndex			   HEATMAP_INDEX_BLUE_GREEN_RED
#endif // #ifndef __KERNELCC__

#endif // HOST_DEVICE_COMMON_KERNEL_OPTIONS_HEATMAP_OPTIONS_H // #ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_HEATMAP_OPTIONS_H
