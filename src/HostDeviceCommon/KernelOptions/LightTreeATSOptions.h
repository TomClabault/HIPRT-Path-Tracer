/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LIGHT_TREE_ATS_OPTIONS_H
#define HOST_DEVICE_COMMON_LIGHT_TREE_ATS_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

 /**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *     - If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *             cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *             we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

/**
 * If this is true, adaptive tree splitting will be used as described in section 5.4 of
 * [Importance Sampling of Many Lights with Adaptive Tree Splitting, Conty & Kulla 2018]
 */
#define LightTreeATSDoSplitting KERNEL_OPTION_TRUE

/**
 * If splitting is enabled, how many light samples, at most, per shading point
 * is allowed
 */
#define LightTreeATSSplittingMaxLightSamples 4

/**
 * If true, the various light samples produced by splitting will all be "shaded" with visibility, which
 * may be very costly and inefficient on scenes where visibility isn't an issue.
 * 
 * If false, visibility noise won't be improved but efficiency may improve drastically depending on the scene.
 * 
 * Not using visibility does not introduce bias because the light tree splitting samples aren't really
 * shaded for real, rather, one light sample of all the split samples is chosen proportional to its
 * contribution. It is that contribution which includes visibility or not.
 */
#define LightTreeATSSplittingIncludeVisibility KERNEL_OPTION_TRUE

#endif // #ifndef __KERNELCC__

#endif
