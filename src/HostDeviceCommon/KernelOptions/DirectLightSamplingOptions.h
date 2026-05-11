/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_DIRECT_LIGHT_SAMPLING_OPTIONS_H
#define HOST_DEVICE_COMMON_DIRECT_LIGHT_SAMPLING_OPTIONS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/KernelOptions/Common.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/KernelOptions/LightTreeATSOptions.h"

#define LSS_NO_DIRECT_LIGHT_SAMPLING 0
#define LSS_ONE_LIGHT				 1
#define LSS_BSDF					 2
#define LSS_MIS_LIGHT_BSDF			 3
#define LSS_RIS_BSDF_AND_LIGHT		 4
#define LSS_RISLTC					 5
#define LSS_LTC_SHADING				 6
#define LSS_RESTIR_DI				 7

#define LSS_BASE_UNIFORM		0
#define LSS_BASE_POWER			1
#define LSS_BASE_LIGHT_TREE_ATS 2
#define LSS_BASE_LIGHT_TREE_SG	3
#define LSS_BASE_REGIR			4

#define TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA		   0
#define TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE		   1
#define TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE 2

#define TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_TURK_1990	0
#define TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_HEITZ_2019 1

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
 * If the length of the normal is less than that, the triangle is going to be rejected
 * from light sampling.
 *
 * This is to avoid having NaN troubles with degenerate triangles (that have no surface mostly
 * and so a normal of length 0)
 */
#define TriangleSamplingNormalLengthRejectionThreshold 1.0e-9f

/**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

/**
 * How to sample lights in the scene.
 * This directly affects the 'DirectLightNEEEstimator' that estimates NEE
 *
 *	- LSS_BASE_UNIFORM
 *		Lights are sampled uniformly
 *
 *	- LSS_BASE_POWER
 *		Lights are sampled proportionally to their power
 *
 *	- LSS_BASE_LIGHT_TREE_ATS
 *		Implementation of [Importance Sampling of Many Lights with Adaptive Tree Splitting, Conty & Kulla, 2018]
 *
 *	- LSS_BASE_LIGHT_TREE_SG
 *		Implementation of [Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting, Tokuyoshi et al., 2024]
 *
 *	- LSS_BASE_REGIR
 *		Uses ReGIR to sample lights
 *		Very custom and advanced implementation of [Rendering many lights with grid-based reservoirs, Boksansky, 2021] +
 *		Disney's cache points: [Cache Points For Production-Scale Occlusion-Aware Many-Lights Sampling And Volumetric Scattering, Li et al. 2024]
 *
 *      Blog post explaining the details of this ReGIR implementation: https://tomclabault.github.io/blog/2025/regir/
 */
#define DirectLightSamplingStrategy LSS_BASE_POWER

/**
 * What direct lighting sampling strategy to use.
 *
 * Possible values (the prefix LSS stands for "Light Sampling strategy"):
 *
 *	- LSS_NO_DIRECT_LIGHT_SAMPLING
 *		No direct light sampling. Emission is only gathered if rays happen to bounce into the lights.
 *
 *	- LSS_ONE_LIGHT
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
 * 	- LSS_RISLTC
 *		Samples lights in the scene with RISLTC (from [Combining Resampled Importance and Projected Solid Angle
 *		Samplings for Many Area Light Rendering, Shash et. al 2023])
 *
 *	- LSS_RESTIR_DI
 *		Uses ReSTIR DI to sample direct lighting at the first bounce in the scene.
 *		Later bounces use the strategy given by ReSTIR_DI_LaterBouncesSamplingStrategy
 *
 *	- LSS_LTC_SHADING
 *		Uses Linearly Transformed Cosines to analytically shade lights. This is biased
 *		as shadowing is not taken into account. Not all BSDF lobe configurations are supported.
 */
#if PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT
// ReSTIR PT is forcing RIS
#define DirectLightNEEEstimator LSS_RIS_BSDF_AND_LIGHT
#else
#define DirectLightNEEEstimator LSS_BSDF
#endif

/**
 * What sampling strategy to use to sample points on triangles (most relevant
 * for sampling points on emissive triangles for light sampling)
 *
 * - TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA
 *		Most basic sampling method, fastest but has the highest variance. Does not
 *		take the shading point into consideration at all
 *
 * - TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE
 *		Slower than uniformly sampling the area but has lower variance.
 *		Takes the geometry term into account but not the cosine term at the shading point
 *
 * - TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE
 *		Slower than sampling according to solid angle but has even lower
 *		variance. Takes the cosine term at the shading point into account
 *		on top of the geometry term.
 */
#define TrianglePointSamplingStrategy TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA

/**
 * If true, the LTC-based method from [BRDF Importance Sampling for Polygonal Lights, Peters 2021] will be used
 * for sampling a point on emissive triangle. This has for effect of taking the BRDF into account
 * when sampling the point on the triangle, massively increasing the quality of the sampling on glossy surfaces.
 *
 * This only applies if the TrianglePointSamplingStrategy is set to
 * TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE or TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE.
 *
 * Warning: LTC-based sampling may be slightly biased due to the BRDF approximation error of LTCs.
 */
#define TrianglePointSamplingStrategySolidAngleUseLTC KERNEL_OPTION_FALSE

/**
 * How to randomly sample a point on a triangle
 *
 *	- TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_TURK_1990
 *		Common way of warping from a square to a triangle using square roots:
 *		V = (1.0f - sqrt(u1)) * V1 + sqrt(u1) * (s2 * V2 + (1.0f - s2) * V3)
 *
 *	- TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_HEITZ_2019
 *		Implementation of [A Low-Distortion Map Between Triangle and Square, Heitz, 2019]
 *		It is faster than Turk method's and better perserves the stratification of the random
 *		number samplers
 */
#define TrianglePointSamplingUniformAreaStrategy TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_HEITZ_2019

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
 * Whether or not to allow backfacing lights during NEE evaluation.
 *
 * For most scenes, this is going to have no impact on visuals as lights are generally
 * watertight meshes, meaning that backfacing emissive triangles of those meshes are not visible from
 * the outside. There will thus be no visual difference but a non negligeable boost in
 * performance/sampling quality as backfacing lights will not be sampled anymore (depending on the sampling strategy)
 */
#define DirectLightSamplingAllowBackfacingLights KERNEL_OPTION_FALSE

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
#define EnvmapSamplingDoBSDFMIS KERNEL_OPTION_FALSE

/**
 * Whether or not to do bilinear filtering when sampling the envmap.
 *
 * This is mostly useful when the camera is looking straigth at the envmap and we don't
 * have camera ray jittering on: in this case, bilinear filtering will hide the
 * pixelated look of the envmap.
 */
#define EnvmapSamplingDoBilinearFiltering KERNEL_OPTION_FALSE

#endif // #ifndef __KERNELCC__

#ifdef LightTreeATSDoSplitting
// Some kernels are not meant to be compiled with kernel compiler options
// so this function below will not compile for those kernels because they don't
// have LightTreeATSDoSplitting defined for example. So we're guarding that function
// if #ifdef to avoid compilation issues.

template <int lightSamplingStrategy>
HIPRT_DEVICE constexpr int DirectLightSampleCount()
{
	if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS && LightTreeATSDoSplitting == KERNEL_OPTION_TRUE)
		// ATS Light tree with splitting is the only strategy that supports multiple light samples per path vertex
		return LightTreeATSSplittingMaxLightSamples;
	else
		// Other strategies just return 1 light sample per path vertex
		return 1;
}

template <int lightSamplingStrategy>
HIPRT_DEVICE constexpr int DirectLightIntegrationFactor()
{
	if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS && LightTreeATSDoSplitting == KERNEL_OPTION_TRUE)
		// ATS Light tree with splitting is essentially not a MC integrator since the returned
		// light samples are disjoint thanks to the splitting.
		//
		// So need not average the 4 (if splitting max light samples is 4) NEE samples together for example
		// but just sum them up. So we're returning 1 here such that the division by the integration factor
		// does not average the 4 samples.
		return 1;
	else
		return DirectLightSampleCount<lightSamplingStrategy>();
}

#define DirectLightNEEEstimatorHasBSDFSampling                                                                                                                 \
	(DirectLightNEEEstimator == LSS_BSDF || DirectLightNEEEstimator == LSS_MIS_LIGHT_BSDF || DirectLightNEEEstimator == LSS_RIS_BSDF_AND_LIGHT ||              \
	 DirectLightNEEEstimator == LSS_RISLTC)

#endif

#endif
