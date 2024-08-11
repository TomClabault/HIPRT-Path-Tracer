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

#define RIS_USE_VISIBILITY_FALSE 0
#define RIS_USE_VISIBILITY_TRUE 1

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
#define LSS_RIS_BSDF_AND_LIGHT 4
#define LSS_RESTIR_DI 5
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
 *		Importance samples the environment map using a binary search on the CDF distributions of the envmap
 */
#define EnvmapSamplingStrategy ESS_BINARY_SEARCH

/**
 * Whether or not to use a visiblity term in the target function whose PDF we're approximating with RIS.
 * 
 *	- RIS_USE_VISIBILITY_TRUE 
 *		Do use a visibility term
 *
 *	- RIS_USE_VISIBILITY_FALSE
 *		Don't use a visibility term
 */
#define RISUseVisiblityTargetFunction RIS_USE_VISIBILITY_FALSE

/**
 * What sampling strategy to use for thd GGX NDF
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
 *  - GGX_VNDF_BOUNDED
 *		Sample the distribution of visible normals with a bounded VNDF
 *		sampling range as proposed in [Bounded VNDF Sampling for Smith–GGX Reflections, Eto & Tokuyoshi, 2023]
 *		
 */
#define GGXAnisotropicSampleFunction GGX_VNDF_SAMPLING
//#define GGXAnisotropicSampleFunction GGX_VNDF_SPHERICAL_CAPS

#else // #ifndef __KERNELCC__

#ifndef InteriorStackStrategy
#error "InteriorStackStrategy kernel option not defined"
#endif

#ifndef DirectLightSamplingStrategy
#error "DirectLightSamplingStrategy kernel option not defined"
#endif

#ifndef EnvmapSamplingStrategy
#error "EnvmapSamplingStrategy kernel option not defined"
#endif

#ifndef RISUseVisiblityTargetFunction
#error "RISUseVisiblityTargetFunction kernel option not defined"
#endif

#ifndef GGXAnisotropicSampleFunction
#error "GGXAnisotropicSampleFunction kernel option not defined"
#endif

#endif // #ifndef __KERNELCC__

#endif
