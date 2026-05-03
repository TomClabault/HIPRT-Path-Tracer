/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_H
#define HOST_DEVICE_COMMON_KERNEL_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/GMoNOptions.h"
#include "HostDeviceCommon/KernelOptions/LightTreeATSOptions.h"
#include "HostDeviceCommon/KernelOptions/LightTreeSGOptions.h"
#include "HostDeviceCommon/KernelOptions/NEEPlusPlusOptions.h"
#include "HostDeviceCommon/KernelOptions/PrincipledBSDFKernelOptions.h"
#include "HostDeviceCommon/KernelOptions/ReGIROptions.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRDIOptions.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRGIOptions.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPGOptions.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPTOptions.h"
#include "HostDeviceCommon/KernelOptions/SSBNPermutationOptions.h"

/**
 * This file references the path tracer options that can be passed to HIPCC using the -D <macro>=<value> option.
 * These path tracer options allow "compile-time" branching to enable/disable a variety
 * of functionalities in the path tracer.
 *
 * For example, you can decide, at kernel compile-time, what envmap sampling strategy to use
 *	- "CDF + Binary search"
 *	- "Alias table"
 * by passing the "-D EnvmapSamplingStrategy=1" or "-D EnvmapSamplingStrategy=2" option string during
 * the compilation of the kernel (for "CDF" and "alias table" respectively).
 *
 * If you wish to change one of the option used by the path tracer at runtime (by interacting with
 * ImGui for example), you will have to recompile the kernel with the correct set of options
 * passed to the kernel compiler.
 *
 * The advantage of recompiling the entire kernel over branching with a simple if() condition on
 * a variable (that would be passed in RenderData for example) is that the recompiling approach
 * does not incur an additional register cost that would harm the occupancy potential of the kernel
 * (whereas registers may be allocated for the block {} of the if() conditions since the compiler
 * has no way to know which branch of the if is going to be taken at runtime).
 */

/**
 * Those are simple defines to give names to the option values.
 * This allows the use of LSS_ONE_RANDOM_LIGHT_MIS (for example) instead of a hardcoded '2'
 */
#define BSDF_NONE		0
#define BSDF_LAMBERTIAN 1
#define BSDF_OREN_NAYAR 2
#define BSDF_PRINCIPLED 3

#define NESTED_DIELECTRICS_STACK_SIZE 4

#define ESS_NO_SAMPLING	  0
#define ESS_BINARY_SEARCH 1
#define ESS_ALIAS_TABLE	  2

#define PATH_SAMPLING_BSDF		0
#define PATH_SAMPLING_RESTIR_GI 1
#define PATH_SAMPLING_RESTIR_PT 2
// This is actually a fake option just for convenience in ImGui. ReSTIR PG is useable through enabling ReSTIR GI + ReSTIRPGEnable. ReSTIR PG is not its
// own "path sampling" strategy, it has to be piggy backing on a ReSTIR path sampler
#define PATH_SAMPLING_RESTIR_PG 3

/**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

/**
 * Whether or not to use shared memory and a global buffer for BVH traversal of global rays (no maximum distance).
 *
 * This improves performance at the cost of a higher VRAM usage (because of the global buffer needed)
 */
#define UseSharedStackBVHTraversal KERNEL_OPTION_TRUE

/**
 * Size of the thread blocks for all kernels dispatched by this renderer
 */
#define KernelBlockWidthHeight 8

/**
 * Size of the thread blocks used when dispatching the kernels.
 * This value is used for allocating the shared memory stack for traversal
 */
#define KernelWorkgroupThreadCount (KernelBlockWidthHeight * KernelBlockWidthHeight)

/**
 * Size of the shared memory stack for BVH traversal of "global" rays
 * (rays that search for the closest hit with no maximum distance)
 */
#define SharedStackBVHTraversalSize 16

/**
 * The stack size for handling nested dielectrics
 */
#define NestedDielectricsStackSize NESTED_DIELECTRICS_STACK_SIZE

/**
 * If false, material textures will not be read and so the global material overrider
 * of ImGui will work properly
 */
#define UseMaterialTextures KERNEL_OPTION_TRUE

/**
 * If this is true, the base color texture of the material will always be used
 * even if UseMaterialTextures is set to false
 */
#define UseMaterialBaseColorTextureOverride KERNEL_OPTION_TRUE

/**
 * What sampling strategy to use for sampling the bounces during path tracing.
 *
 *	- PATH_SAMPLING_BSDF
 *		The classical technique: importance samples the BSDF and bounces in that direction
 *
 *	- PATH_SAMPLING_RESTIR_GI
 *		Uses ReSTIR GI for resampling a path for the pixel.
 *
 *		The implementation is based on
 *		[ReSTIR GI: Path Resampling for Real-Time Path Tracing] https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
 *		but is adapted almost full unbiasedness (full unbiasedness while resampling full path trees as in ReSTIR GI paper isn't supported
 *		by the GRIS theory. Fully unbiased path resampling with the current RIS theory can only be achieved by resampling "paths" and not full "path trees" as
 *		proposed in the ReSTIR GI paper and as implemented here)
 *
 *		The original ReSTIR GI paper indeed only is unbiased for a Lambertian BRDF
 *
 *	- PATH_SAMPLING_RESTIR_PT
 *		Implementation of [Generalized Resampled Importance Sampling: Foundations of ReSTIR, Lin et al. 2022], resampling paths and not full path trees (as in
 *ReSTIR GI).
 *
 *	- PATH_SAMPLING_RESTIR_PG
 *		Implementation of [ReSTIR PG: Path Guiding with Spatiotemporally Resampled Paths, Zeng et al. 2025]
 *
 *		Uses ReSTIR Path Guiding for learning a guiding distribution in a hash grid and sampling from that distribution for the path bounces.
 *		This option should only be selected from ImGui and not set directly here as the value
 */
#define PathSamplingStrategy PATH_SAMPLING_RESTIR_PT

/**
 * Whether or not to use a visiblity term in the target function whose PDF we're
 * approximating with RIS.
 * Only applies for pure RIS direct lighting strategy (i.e. not RIS used by ReSTIR
 * on the initial candidates pass for example)
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define RISUseVisiblityTargetFunction KERNEL_OPTION_FALSE

/**
 * Debug option which, if enabled, only outputs the sample 'render_settings.output_debug_sample_N'
 * to the framebuffer.
 *
 * Useful for debugging features that may take effect after the first sample and we only want to see what
 * the second sample (or any other sample) looks like without being accumulated with the previous samples
 */
#define DisplayOnlySampleN KERNEL_OPTION_FALSE

#endif // #ifndef __KERNELCC__

#endif
