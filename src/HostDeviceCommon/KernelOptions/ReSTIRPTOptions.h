/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_PT_OPTIONS_H
#define HOST_DEVICE_COMMON_RESTIR_PT_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRCommonOptions.h"

// This block is a security to make sure that we have everything defined otherwise this can lead
// to weird behavior because of the compiler not knowing about some macros
#ifndef KERNEL_OPTION_TRUE
#error "KERNEL_OPTION_TRUE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#else
#ifndef KERNEL_OPTION_FALSE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#endif
#endif

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
 * See the doc of the RESTIR_MIS_WEIGHTS_TYPE macros in ReSTIRCommonOptions.h for more details on the different types of MIS weights that can be used when
 * resampling spatial neighbors.
 */
#define ReSTIR_PT_MISWeightsType RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS

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

#endif
