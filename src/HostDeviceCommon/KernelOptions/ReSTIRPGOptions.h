/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_PG_OPTIONS_H
#define HOST_DEVICE_COMMON_RESTIR_PG_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

// This block is a security to make sure that we have everything defined otherwise this can lead
// to weird behavior because of the compiler not knowing about some macros
#ifndef KERNEL_OPTION_TRUE
#error "KERNEL_OPTION_TRUE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#else
#ifndef KERNEL_OPTION_FALSE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#endif
#endif

#define RESTIR_PG_NO_DEBUG								 0
#define RESTIR_PG_DEBUG_GRID_CELLS						 1
#define RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_DIRECTION 2
#define RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_SHARPNESS 3
#define RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_WEIGHT	 4

#ifndef __KERNELCC__

/**
 * Whether or not to enable ReSTIR PG (Path guiding).
 *
 * This is an implementation of ["ReSTIR PG: Path Guiding with Spatiotemporally Resampled Paths", Zeng et al., 2025]
 */
#define ReSTIRPGEnable KERNEL_OPTION_TRUE

#define ReSTIRPGDistributionComponentCount 1

#define ReSTIRPGHashGridCollisionResolveSteps 4

#define ReSTIRPGDebugMode RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_WEIGHT

#endif

#endif
