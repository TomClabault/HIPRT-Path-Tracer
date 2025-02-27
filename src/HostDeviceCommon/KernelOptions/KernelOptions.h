/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_H
#define HOST_DEVICE_COMMON_KERNEL_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/GMoNOptions.h"
#include "HostDeviceCommon/KernelOptions/PrincipledBSDFKernelOptions.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRDIOptions.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRGIOptions.h"

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
#define MATERIAL_PACK_STRATEGY_USE_PACKED 0
#define MATERIAL_PACK_STRATEGY_USE_UNPACKED 1

#define BSDF_NONE 0
#define BSDF_LAMBERTIAN 1
#define BSDF_OREN_NAYAR 2
#define BSDF_PRINCIPLED 3

#define NESTED_DIELECTRICS_STACK_SIZE 3

#define LSS_NO_DIRECT_LIGHT_SAMPLING 0
#define LSS_UNIFORM_ONE_LIGHT 1
#define LSS_BSDF 2
#define LSS_MIS_LIGHT_BSDF 3
#define LSS_RIS_BSDF_AND_LIGHT 4
#define LSS_RESTIR_DI 5

#define ESS_NO_SAMPLING 0
#define ESS_BINARY_SEARCH 1
#define ESS_ALIAS_TABLE 2

#define PSS_BSDF 0
#define PSS_RESTIR_GI 1

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
 * If true, the BSDF ray shot for BSDF MIS during the evaluation of NEE will be reused
 * for the next bounce. 
 * 
 * There is virtually no point in disabling that option. This options i there only for
 * performance comparisons with and without reuse
 */
#define ReuseBSDFMISRay KERNEL_OPTION_FALSE

/**
 * Partial and experimental implementation of [Generate Coherent Rays Directly, Liu et al., 2024]
 * for reuse sampled directions on the first hit accross the threads of warps
 */
#define DoFirstBounceWarpDirectionReuse KERNEL_OPTION_FALSE

/**
 * Allows the overriding of the BRDF/BSDF used by the path tracer. When an override is used,
 * the material retains its properties (color, roughness, ...) but only the parameters relevant
 * to the overriden BSDF are used.
 * 
 *	- BSDF_NONE
 *		Materials will use their default BRDF/BSDF, no override
 * 
 *	- BSDF_LAMBERTIAN
 *		All materials will use a lambertian BRDF
 * 
 *	- BSDF_OREN_NAYAR
 *		All materials will use the Oren Nayar diffuse BRDF
 * 
 *	- BSDF_PRINCIPLED
 *		All materials will use the Principled BSDF
 */
#define BSDFOverride BSDF_NONE

/**
 * The stack size for handling nested dielectrics
 */
#define NestedDielectricsStackSize NESTED_DIELECTRICS_STACK_SIZE

/**
 * Whether or not to use shared memory for the nested dielectrics stack
 * 
 * This option is actually very experimental and should be KERNEL_OPTION_FALSE for
 * correct results. Incorrect results are expected (with ReSTIR GI notably) if this option
 * is KERNEL_OPTION_TRUE
 * 
 * In practice, no performance difference was observed between KERNEL_OPTION_FALSE and KERNEL_OPTION_TRUE
 */
#define NestedDielectricsStackUseSharedMemory KERNEL_OPTION_FALSE

/**
 * What direct lighting sampling strategy to use.
 * 
 * Possible values (the prefix LSS stands for "Light Sampling strategy"):
 * 
 *	- LSS_NO_DIRECT_LIGHT_SAMPLING
 *		No direct light sampling
 * 
 *	- LSS_UNIFORM_ONE_LIGHT
 *		Samples one random light in the scene without MIS. 
 *		Efficient as long as there are not too many lights in the scene and no glossy surfaces
 * 
 *  - LSS_BSDF
 *		Samples lights only using a BSDF sample
 *		Efficient as long as light sources in the scene are large
 * 
 *	- LSS_MIS_LIGHT_BSDF
 *		Samples one random light in the scene with MIS (Multiple Importance Sampling): light sample + BRDF sample
 * 
 *	- LSS_RIS_BSDF_AND_LIGHT
 *		Samples lights in the scene with Resampled Importance Sampling
 * 
 *	- LSS_RESTIR_DI
 *		Uses ReSTIR DI to sample direct lighting at the first bounce in the scene.
 *		Later bounces use the strategy given by ReSTIR_DI_LaterBouncesSamplingStrategy
 */
#define DirectLightSamplingStrategy LSS_UNIFORM_ONE_LIGHT

/**
 * Whether or not to use NEE++ features at all
 */
#define DirectLightUseNEEPlusPlus KERNEL_OPTION_FALSE

/**
 * Whether or not to use russian roulette to avoid tracing shadow rays based on the visibility
 * information of NEE++
 */
#define DirectLightUseNEEPlusPlusRR KERNEL_OPTION_TRUE

/**
 * This a debug option to visualize shadow rays discarded by the NEE++ russian roulette
 */
#define DirectLightNEEPlusPlusDisplayShadowRaysDiscarded KERNEL_OPTION_FALSE

/**
 * When using the 'DirectLightNEEPlusPlusDisplayShadowRaysDiscarded' kernel options
 * for displaying in the viewport where shadow rays were discarded, this parameter is used
 * to determine at what bounce in the scene we should display the shadow ray discarded or not
 * 
 * 0 is the first hit
 */ 
#define DirectLightNEEPlusPlusDisplayShadowRaysDiscardedBounce 0

/**
 * If this is true, light sampling with NEE (emissive geometry & envmap) will not even 
 * be attempted on perfectly smooth materials (smooth glass, smooth metals, ...)
 * 
 * This is because these materials are delta distributions and light sampling 
 * has no chance to give any contribution.
 * 
 * There is no point in disabling that option, this is basically only for 
 * performance comparisons
 */
#define DirectLightSamplingDeltaDistributionOptimization KERNEL_OPTION_TRUE

/**
 * What envmap sampling strategy to use
 * 
 * Possible values (the prefix ESS stands for "Envmap Sampling Strategy"):
 * 
 *	- ESS_NO_SAMPLING
 *		No importance sampling of the envmap
 * 
 *	- ESS_BINARY_SEARCH
 *		Importance samples a texel of the environment map proportionally to its
 *		luminance using a binary search on the CDF distributions of the envmap luminance.
 *		Good convergence.
 * 
 * - ESS_ALIAS_TABLE
 *		Importance samples a texel of the environment map proportionally to its
 *		luminance using an alias table for constant time sampling
 *		Good convergence and faster than ESS_BINARY_SEARCH
 */
#define EnvmapSamplingStrategy ESS_ALIAS_TABLE

/**
 * Whether or not to do Muliple Importance Sampling between the envmap sample and a BSDF
 * sample when importance sampling direct lighting contribution from the envmap
 */
#define EnvmapSamplingDoBSDFMIS KERNEL_OPTION_TRUE

/**
 * What sampling strategy to use for sampling the bounces during path tracing.
 * 
 *	- PSS_BSDF
 *		The classical technique: importance samples the BSDF and bounces in that direction
 * 
 *	- PSS_RESTIR_GI
 *		Uses ReSTIR GI for resampling a path for the pixel.
 * 
 *		The implementation is based on 
 *		[ReSTIR GI: Path Resampling for Real-Time Path Tracing] https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
 *		but is adapted for full unbiasedness.
 * 
 *		The original ReSTIR GI paper indeed only is unbiased for a Lambertian BRDF
 */
//#define PathSamplingStrategy PSS_BSDF
#define PathSamplingStrategy PSS_RESTIR_GI

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
 * This is a handy macro that tells us whether or not we have any other kernel option 
 * that overrides the color of the framebuffer
 */
#define ViewportColorOverriden (DirectLightNEEPlusPlusDisplayShadowRaysDiscarded)

#endif // #ifndef __KERNELCC__

#endif
