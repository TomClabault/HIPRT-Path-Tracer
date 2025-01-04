/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_H
#define HOST_DEVICE_COMMON_KERNEL_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/PrincipledBSDFKernelOptions.h"

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

#define RESTIR_DI_BIAS_CORRECTION_1_OVER_M 0
#define RESTIR_DI_BIAS_CORRECTION_1_OVER_Z 1
#define RESTIR_DI_BIAS_CORRECTION_MIS_LIKE 2
#define RESTIR_DI_BIAS_CORRECTION_MIS_GBH 3
#define RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS 4
#define RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE 5

#define RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT 0
#define RESTIR_DI_LATER_BOUNCES_BSDF 1
#define RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF 2
#define RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT 3

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
#define ReuseBSDFMISRay KERNEL_OPTION_TRUE

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
#define BSDFOverride BSDF_LAMBERTIAN

/**
 * The stack size for handling nested dielectrics
 */
#define NestedDielectricsStackSize NESTED_DIELECTRICS_STACK_SIZE

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
#define DirectLightSamplingStrategy LSS_NO_DIRECT_LIGHT_SAMPLING

/**
 * What envmap sampling strategy to use
 * 
 * Possible values (the prefix ESS stands for "Envmap Sampling Strategy"):
 * 
 *	- ESS_NO_SAMPLING
 *		No importance sampling of the envmap
 * 
 *	- ESS_BINARY_SEARCH
 *		Importance samples the environment map using a binary search on the CDF
 *		distributions of the envmap
 */
#define EnvmapSamplingStrategy ESS_NO_SAMPLING

/**
 * Whether or not to do Muliple Importance Sampling between the envmap sample and a BSDF
 * sample when importance sampling direct lighting contribution from the envmap
 */
#define EnvmapSamplingDoBSDFMIS KERNEL_OPTION_TRUE

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
 * Whether or not to use a visibility term in the target function when resampling
 * initial candidates in ReSTIR DI. *
 * 
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_InitialTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * Whether or not to use a visibility term in the target function when resampling
 * samples in ReSTIR DI. This applies to the spatial reuse pass only.
 * This option can have a good impact on quality and be worth it in terms of cost.
 *
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_SpatialTargetFunctionVisibility KERNEL_OPTION_TRUE

/**
 * Whether or not to do a visibility check at the end of the initial candidates sampling.
 * This discards reservoirs (by setting their UCW to 0.0f) whose samples are occluded.
 * This allows following ReSTIR passes (temporal and spatial) to only resample on samples
 * that are not occluded which improves quality quite a bit.
 * 
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_DoVisibilityReuse KERNEL_OPTION_TRUE

/**
 * Whether or not to use a visibility term in the MIS weights (MIS-like weights,
 * generalized balance heuristic, pairwise MIS, ...) used to remove bias when
 * resampling neighbors. An additional visibility ray will be traced for MIS-weight
 * evaluated. This effectively means for each neighbor resamples or (for each neighbor resampled)^2
 * if using the generalized balance heuristics (without pairwise-MIS)
 * 
 * To guarantee unbiasedness, this needs to be true. A small amount of energy loss
 * may be observed if this value is KERNEL_OPTION_FALSE but the performance cost of the spatial
 * reuse will be reduced noticeably
 * 
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_BiasCorrectionUseVisibility KERNEL_OPTION_TRUE

/**
 * What bias correction weights to use when resampling neighbors (temporal / spatial)
 * 
 *  - RESTIR_DI_BIAS_CORRECTION_1_OVER_M
 *		Very simple biased weights as described in the 2020 paper (Eq. 6).
 *		Those weights are biased because they do not account for cases where
 *		we resample a sample that couldn't have been produced by some neighbors. 
 *		The bias shows up as darkening, mostly at object boundaries. In GRIS vocabulary, 
 *		this type of weights can be seen as confidence weights alone c_i / sum(c_j)
 * 
 *  - RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
 *		Simple unbiased weights as described in the 2020 paper (Eq. 16 and Section 4.3)
 *		Those weights are unbiased but can have **extremely** bad variance when a neighbor being resampled
 *		has a very low target function (when the neighbor is a glossy surface for example).
 *		See Fig. 7 of the 2020 paper.
 * 
 *  - RESTIR_DI_BIAS_CORRECTION_MIS_LIKE
 *		Unbiased weights as proposed by Eq. 22 of the paper. Way better than 1/Z in terms of variance
 *		and still unbiased.
 * 
 *  - RESTIR_DI_BIAS_CORRECTION_MIS_GBH
 *		Unbiased MIS weights that use the generalized balance heuristic. Very good variance reduction but O(N^2) complexity,
		N being the number of neighbors resampled.
 *		Eq. 36 of the 2022 Generalized Resampled Importance Sampling paper.
 * 
 *	- RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
 *		Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.
 *		Section 7.1.3 of "A Gentle Introduction to ReSTIR", 2023
 */
#define ReSTIR_DI_BiasCorrectionWeights RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE

/**
 * What direct lighting sampling strategy to use for secondary bounces when ReSTIR DI is used for sampling the first bounce
 * 
 * Possible values (the prefix LSS stands for "Light Sampling strategy"):
 * 
 *	- RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT
 *		Samples one random light in the scene without MIS
 * 
 *	- RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF
 *		Samples one random light in the scene with MIS (Multiple Importance Sampling): light sample + BRDF sample
 * 
 *  - RESTIR_DI_LATER_BOUNCES_BSDF
 *		Samples a light using a BSDF sample.
 *		Efficient as long as the light sources in the scene are large.
 * 
 *	- RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT
 *		Samples lights in the scene with Resampled Importance Sampling
 */
#define ReSTIR_DI_LaterBouncesSamplingStrategy RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT

/**
 * If true, lights are presampled in a pre-process pass as described in 
 * [Rearchitecting Spatiotemporal Resampling for Production, Wyman, Panteleev, 2021] 
 * https://research.nvidia.com/publication/2021-07_rearchitecting-spatiotemporal-resampling-production.
 * 
 * This improves performance in scenes with dozens of thousands / millions of 
 * lights by avoiding cache trashing because of the memory random walk that 
 * light sampling becomes with that many lights
 */
#define ReSTIR_DI_DoLightsPresampling KERNEL_OPTION_TRUE

#endif // #ifndef __KERNELCC__

#endif
