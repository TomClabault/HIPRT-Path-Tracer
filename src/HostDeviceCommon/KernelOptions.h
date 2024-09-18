/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_KERNEL_OPTIONS_H
#define HOST_DEVICE_COMMON_KERNEL_OPTIONS_H

/**
 * This file references the path tracer options that can be passed to HIPCC using the -D <macro>=<value> option.
 * These path tracer options allow "compile-time" branching to enable/disable a variety
 * of functionalities in the path tracer.
 * 
 * For example, you can decide, at kernel compile-time, what nested dielectrics strategy to use 
 *	- "automatic" as presented in* Ray Tracing Gems 1, 2019 or 
 *	- "with priorities" as presented in Simple Nested Dielectrics in Ray Traced Images, Schmidt, 2002
 * by passing the "-D InteriorStackStrategy=0" or "-D InteriorStackStrategy=1" option string during
 * the compilation of the kernel (for "automatic" and "with priorities" respectively).
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
#define KERNEL_OPTION_FALSE 0
#define KERNEL_OPTION_TRUE 1

#define BSDF_NONE 0
#define BSDF_LAMBERTIAN 1
#define BSDF_OREN_NAYAR 2
#define BSDF_DISNEY 3

#define ISS_AUTOMATIC 0
#define ISS_WITH_PRIORITIES 1

#define LSS_NO_DIRECT_LIGHT_SAMPLING 0
#define LSS_UNIFORM_ONE_LIGHT 1
#define LSS_BSDF 2
#define LSS_MIS_LIGHT_BSDF 3
#define LSS_RIS_BSDF_AND_LIGHT 4
#define LSS_RESTIR_DI 5

#define ESS_NO_SAMPLING 0
#define ESS_BINARY_SEARCH 1

#define RESTIR_DI_BIAS_CORRECTION_1_OVER_M 0
#define RESTIR_DI_BIAS_CORRECTION_1_OVER_Z 1
#define RESTIR_DI_BIAS_CORRECTION_MIS_LIKE 2
#define RESTIR_DI_BIAS_CORRECTION_MIS_GBH 3
#define RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS 4
#define RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE 5

#define GGX_NO_VNDF 0
#define GGX_VNDF_SAMPLING 1
#define GGX_VNDF_SPHERICAL_CAPS 2
#define GGX_VNDF_BOUNDED 3

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
#define SharedStackBVHTraversalGlobalRays KERNEL_OPTION_TRUE

/**
 * Same as SharedStackBVHTraversalGlobalRays but for shadow rays. The global buffer is shared between global and shadow
 * rays so using only SharedStackBVHTraversalGlobalRays or both with also SharedStackBVHTraversalShadowRays doesn't increase
 * VRAM usage further.
 */
#define SharedStackBVHTraversalShadowRays KERNEL_OPTION_TRUE

/**
 * Size of the thread blocks used when dispatching the kernels. 
 * This value is used for allocating the shared memory stack for traversal
 */
#define SharedStackBVHTraversalBlockSize 64

 /**
  * Size of the shared memory stack for BVH traversal of "global" rays 
  * (rays that search for the closest hit with no maximum distance)
  */
#define SharedStackBVHTraversalSizeGlobalRays 16

 /**
  * Size of the shared memory stack for BVH traversal of shadow rays 
  * (rays that search for anyhit with a given maximum distance)
  */
#define SharedStackBVHTraversalSizeShadowRays 16

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
 *	- BSDF_DISNEY
 *		All materials will use the Disney BSDF
 */
#define BSDFOverride BSDF_NONE

/**
 * What nested dielectrics strategy to use.
 * 
 * Possible values (the prefix ISS stands for "Interior Stack Strategy"):
 * 
 *	- ISS_AUTOMATIC
 *		"automatic" strategy as presented in* Ray Tracing Gems 1, 2019
 * 
 *	- ISS_WITH_PRIORITIES
 *		"with priorities" as presented in Simple Nested Dielectrics in Ray Traced Images, Schmidt, 2002
 */
#define InteriorStackStrategy ISS_WITH_PRIORITIES

/**
 * What direct lighting sampling strategy to use.
 * 
 * Possible values (the prefix LSS stands for "Light Sampling strategy"):
 * 
 *	- LSS_NO_DIRECT_LIGHT_SAMPLING
 *		No direct light sampling
 * 
 *	- LSS_UNIFORM_ONE_LIGHT
 *		Samples one random light in the scene without MIS
 * 
 *	- LSS_MIS_LIGHT_BSDF
 *		Samples one random light in the scene with MIS (Multiple Importance Sampling): light sample + BRDF sample
 * 
 *	- LSS_RIS_BSDF_AND_LIGHT
 *		Samples lights in the scene with Resampled Importance Sampling using 
 *		render_settings.ris_number_of_light_candidates light candidates and
 *		render_settings.ris_number_of_bsdf_candidates BSDF candidates
 * 
 *	- LSS_RESTIR_DI
 *		Uses ReSTIR DI to sample direct lighting at the first bounce in the scene.
 * 
 *		ReSTIR DI then uses:
 *			- render_settings.ris_number_of_light_candidates & render_settings.ris_number_of_bsdf_candidates
 *				when sampling the initial candidates with RIS
 */
#define DirectLightSamplingStrategy LSS_RESTIR_DI

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
#define EnvmapSamplingStrategy ESS_BINARY_SEARCH

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
 * samples in ReSTIR DI. This applies to all passes of ReSTIR DI.
 * 
 * In the context of efficiency, there's virtually no need to set this to KERNEL_OPTION_TRUE.
 *
 * The cost of tracing yet an additional visibility ray when resampling
 * isn't worth it in terms of variance reduction. This option is basically only for
 * experimentation purposes.
 * 
 *	- KERNEL_OPTION_TRUE or KERNEL_OPTION_FALSE values are accepted. Self-explanatory
 */
#define ReSTIR_DI_TargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * If false, the light sampler PDF will not be taken into account when computing the MIS weight
 * of initial envmap sample candidates i.e. :
 *	 
 *	envmap_mis_weight = envmapPDF / (bsdfPDF + envmapPDF)
 *	 
 *	instead of
 *	
 *	envmap_mis_weight = envmapPDF / (bsdfPDF + envmapPDF + lightPDF)
 *		
 * This is technically biased (because the MIS weights now don't sum to 1) but the bias is
 * litteraly imperceptible in my experience (or I haven't found a scene that shows the bias) and this
 * saves casting a shadow ray (needed for the light sampler PDF evaluation)
 */
#define ReSTIR_DI_EnvmapSamplesMISLightSampler KERNEL_OPTION_FALSE

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
#define ReSTIR_DI_BiasCorrectionUseVisiblity KERNEL_OPTION_TRUE

/**
 * This option allows using multiple spatial reuse passes without bias (required by some weighting schemes)
 * 
 * Why is this needed?
 *
 * Picture the case where we have visibility reuse (at the end of the initial candidates sampling pass),
 * visibility term in the bias correction target function (when counting the neighbors that could
 * have produced the picked sample) and 2 spatial reuse passes.
 *
 * The first spatial reuse pass reuses from samples that were produced with visibility in mind
 * (because of the visibility reuse pass that discards occluded samples). This means that we need
 * the visibility in the target function used when counting the neighbors that could have produced
 * the picked sample otherwise we may think that our neighbor could have produced the picked
 * sample where actually it couldn't because the sample is occluded at the neighbor. We would
 * then have a Z denominator (with 1/Z weights) that is too large and we'll end up with darkening.
 *
 * Now at the end of the first spatial reuse pass, the center pixel ends up with a sample that may
 * or may not be occluded from the center's pixel point of view. We didn't include the visibility
 * in the target function when resampling the neighbors (only when counting the "correct" neighbors
 * but that's all) so we are not giving a 0 weight to occluded resampled neighbors --> it is possible
 * that we picked an occluded sample.
 *
 * In the second spatial reuse pass, we are now going to resample from our neighbors and get some
 * samples that were not generated with occlusion in mind (because the resampling target function of
 * the first spatial reuse doesn't include visibility). Yet, we are going to weight them with occlusion
 * in mind. This means that we are probably going to discard samples because of occlusion that could
 * have been generated because they are generated without occlusion test. We end up discarding too many
 * samples --> brightening bias.
 *
 * With the visibility reuse at the end of each spatial pass, we force samples at the end of each
 * spatial reuse to take visibility into account so that when we weight them with visibility testing,
 * everything goes well
 *
 * As an optimization, we also do this for the pairwise MIS because pairwise MIS evaluates the target function
 * of reservoirs at their own location. Doing the visibility reuse here ensures that a reservoir sample at its own location
 * includes visibility and so we do not need to recompute the target function of the neighbors in this case. We can just
 * reuse the target function stored in the reservoir
 *
 * The user is given the choice to remove bias using this option or not. It introduces very little bias
 * in practice (but noticeable when switching back and forth between reference image/biased image) so the
 * performance boost of not tracing rays at the end of each spatial reuse pass given the very small increase
 * in bias may be worth it
 */
#define ReSTIR_DI_SpatialReuseOutputVisibilityCheck KERNEL_OPTION_TRUE

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
 * What sampling strategy to use for the GGX NDF
 * 
 *  - GGX_NO_VNDF
 *		Not sampling the visible distribution of normals.
 *		Just classic GGX sampling
 * 
 *  - GGX_VNDF_SAMPLING
 *		Sample the distribution of visible normals as proposed
 *		in [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 * 
 *  - GGX_VNDF_SPHERICAL_CAPS
 *		Sample the distribution of visible normals using spherical
 *		caps as proposed in [Sampling Visible GGX Normals with Spherical Caps, Dupuy & Benyoub, 2023]
 * 
 *  - GGX_VNDF_BOUNDED [Not Yet Implemented]
 *		Sample the distribution of visible normals with a bounded VNDF
 *		sampling range as proposed in [Bounded VNDF Sampling for Smith-GGX Reflections, Eto & Tokuyoshi, 2023]
 *		
 */
#define GGXAnisotropicSampleFunction GGX_VNDF_SAMPLING

#endif // #ifndef __KERNELCC__

#endif
