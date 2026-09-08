/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MEGAKERNEL_OPTIONS_H
#define HOST_DEVICE_COMMON_MEGAKERNEL_OPTIONS_H

#define MEGAKERNEL_DEBUG_MODE_NO_DEBUG					0
#define MEGAKERNEL_DEBUG_MODE_PIXEL_CONVERGENCE_HEATMAP 1
#define MEGAKERNEL_DEBUG_MODE_PIXEL_CONVERGED_MAP		2

/**
 * Debug view for the megakernel.
 */
#ifndef __KERNELCC__
#define MegakernelDebugMode MEGAKERNEL_DEBUG_MODE_NO_DEBUG
#endif // #ifndef __KERNELCC__

#endif // #ifndef HOST_DEVICE_COMMON_MEGAKERNEL_OPTIONS_H
