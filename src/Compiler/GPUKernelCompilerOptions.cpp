/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernel.h"
#include "Compiler/GPUKernelCompilerOptions.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "Utils/Utils.h"

#include <cassert>

/**
 * Defining the strings that go with the option so that they can be passed to the shader compiler
 * with the -D<string>=<value> option.
 * 
 * The strings used here must match the ones used in KernelOptions.h
 */
const std::string GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL = "UseSharedStackBVHTraversal";
const std::string GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE = "SharedStackBVHTraversalSize";
const std::string GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_BLOCK_SIZE = "KernelWorkgroupThreadCount";
const std::string GPUKernelCompilerOptions::REUSE_BSDF_MIS_RAY = "ReuseBSDFMISRay";
const std::string GPUKernelCompilerOptions::DO_FIRST_BOUNCE_WARP_DIRECTION_REUSE = "DoFirstBounceWarpDirectionReuse";

const std::string GPUKernelCompilerOptions::BSDF_OVERRIDE = "BSDFOverride";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE = "PrincipledBSDFDiffuseLobe";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION = "PrincipledBSDFDoEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENFORCE_ENERGY_CONSERVATION = "PrincipledBSDFEnforceStrongEnergyConservation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION = "PrincipledBSDFDoGlassEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION = "PrincipledBSDFDoClearcoatEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION = "PrincipledBSDFDoMetallicEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION = "PrincipledBSDFDoMetallicFresnelEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION = "PrincipledBSDFDoSpecularEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION = "PrincipledBSDFDeltaDistributionEvaluationOptimization";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL = "PrincipledBSDFSampleGlossyBasedOnFresnel";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL = "PrincipledBSDFSampleCoatBasedOnFresnel";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION = "PrincipledBSDFDoMicrofacetRegularization";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION = "PrincipledBSDFDoMicrofacetRegularizationConsistentParameterization";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC= "PrincipledBSDFMicrofacetRegularizationDiffusionHeuristic";
const std::string GPUKernelCompilerOptions::GGX_SAMPLE_FUNCTION = "PrincipledBSDFAnisotropicGGXSampleFunction";
const std::string GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION = "NestedDielectricsStackSize";

const std::string GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY = "TrianglePointSamplingStrategy";

const std::string GPUKernelCompilerOptions::REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY = "ReGIR_GridFillLightSamplingBaseStrategy";
const std::string GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_VISIBILITY = "ReGIR_GridFillTargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM = "ReGIR_GridFillTargetFunctionCosineTerm";
const std::string GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM_LIGHT_SOURCE = "ReGIR_GridFillTargetFunctionCosineTermLightSource";
const std::string GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_TARGET_FUNCTION_VISIBILITY = "ReGIR_ShadingResamplingTargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_INCLUDE_BSDF = "ReGIR_ShadingResamplingIncludeBSDF";
const std::string GPUKernelCompilerOptions::REGIR_DO_VISIBILITY_REUSE = "ReGIR_DoVisibilityReuse";
const std::string GPUKernelCompilerOptions::REGIR_FALLBACK_LIGHT_SAMPLING_STRATEGY = "ReGIR_FallbackLightSamplingStrategy";
const std::string GPUKernelCompilerOptions::REGIR_DEBUG_MODE = "ReGIR_DebugMode";

const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY = "DirectLightSamplingStrategy";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY = "DirectLightSamplingBaseStrategy";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS = "DirectLightUseNEEPlusPlus";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE = "DirectLightUseNEEPlusPlusRR";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED = "DirectLightNEEPlusPlusDisplayShadowRaysDiscarded";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED_BOUNCE = "DirectLightNEEPlusPlusDisplayShadowRaysDiscardedBounce";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION = "DirectLightSamplingDeltaDistributionOptimization";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS = "DirectLightSamplingAllowBackfacingLights";
const std::string GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION = "RISUseVisiblityTargetFunction";
const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY = "EnvmapSamplingStrategy";
const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS = "EnvmapSamplingDoBSDFMIS";
const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING = "EnvmapSamplingDoBilinearFiltering";

const std::string GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY = "PathSamplingStrategy";

const std::string GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY = "ReSTIR_DI_InitialTargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY = "ReSTIR_DI_SpatialTargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE = "ReSTIR_DI_DoVisibilityReuse";
const std::string GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY = "ReSTIR_DI_BiasCorrectionUseVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS = "ReSTIR_DI_BiasCorrectionWeights";
const std::string GPUKernelCompilerOptions::RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY = "ReSTIR_DI_LaterBouncesSamplingStrategy";
const std::string GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING = "ReSTIR_DI_DoLightPresampling";
const std::string GPUKernelCompilerOptions::RESTIR_DI_LIGHT_PRESAMPLING_STRATEGY = "ReSTIR_DI_LightPresamplingStrategy";
const std::string GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT = "ReSTIR_DI_SpatialDirectionalReuseBitCount";
const std::string GPUKernelCompilerOptions::RESTIR_DI_DO_OPTIMAL_VISIBILITY_SAMPLING = "ReSTIR_DI_DoOptimalVisibilitySampling";

const std::string GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY = "ReSTIR_GI_SpatialTargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT = "ReSTIR_GI_SpatialDirectionalReuseBitCount";
const std::string GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_USE_VISIBILITY = "ReSTIR_GI_BiasCorrectionUseVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_WEIGHTS = "ReSTIR_GI_BiasCorrectionWeights";
const std::string GPUKernelCompilerOptions::RESTIR_GI_DO_OPTIMAL_VISIBILITY_SAMPLING = "ReSTIR_GI_DoOptimalVisibilitySampling";

const std::string GPUKernelCompilerOptions::GMON_M_SETS_COUNT = "GMoNMSetsCount";

const std::unordered_set<std::string> GPUKernelCompilerOptions::ALL_MACROS_NAMES = {
	GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
	GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE,
	GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_BLOCK_SIZE,
	GPUKernelCompilerOptions::REUSE_BSDF_MIS_RAY,
	GPUKernelCompilerOptions::DO_FIRST_BOUNCE_WARP_DIRECTION_REUSE,

	GPUKernelCompilerOptions::BSDF_OVERRIDE,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENFORCE_ENERGY_CONSERVATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC ,
	GPUKernelCompilerOptions::GGX_SAMPLE_FUNCTION,
	GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION,

	GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY,

	GPUKernelCompilerOptions::REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY,
	GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM,
	GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM_LIGHT_SOURCE,
	GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_INCLUDE_BSDF,
	GPUKernelCompilerOptions::REGIR_DO_VISIBILITY_REUSE,
	GPUKernelCompilerOptions::REGIR_FALLBACK_LIGHT_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::REGIR_DEBUG_MODE,

	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY,
	GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS,
	GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE,
	GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED,
	GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED_BOUNCE,
	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION,
	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS,
	GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION,
	GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS,
	GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING,

	GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY,

	GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE,
	GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS,
	GPUKernelCompilerOptions::RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING,
	GPUKernelCompilerOptions::RESTIR_DI_LIGHT_PRESAMPLING_STRATEGY,
	GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT,
	GPUKernelCompilerOptions::RESTIR_DI_DO_OPTIMAL_VISIBILITY_SAMPLING,

	GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT,
	GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_USE_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_WEIGHTS,
	GPUKernelCompilerOptions::RESTIR_GI_DO_OPTIMAL_VISIBILITY_SAMPLING,

	GPUKernelCompilerOptions::GMON_M_SETS_COUNT,
};

GPUKernelCompilerOptions::GPUKernelCompilerOptions()
{
	// Mandatory options that every kernel must have so we're
	// adding them here with their default values
	m_options_macro_map[GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL] = std::make_shared<int>(UseSharedStackBVHTraversal);
	m_options_macro_map[GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE] = std::make_shared<int>(SharedStackBVHTraversalSize);
	m_options_macro_map[GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_BLOCK_SIZE] = std::make_shared<int>(KernelWorkgroupThreadCount);
	m_options_macro_map[GPUKernelCompilerOptions::REUSE_BSDF_MIS_RAY] = std::make_shared<int>(ReuseBSDFMISRay);
	m_options_macro_map[GPUKernelCompilerOptions::DO_FIRST_BOUNCE_WARP_DIRECTION_REUSE] = std::make_shared<int>(DoFirstBounceWarpDirectionReuse);

	m_options_macro_map[GPUKernelCompilerOptions::BSDF_OVERRIDE] = std::make_shared<int>(BSDFOverride);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE] = std::make_shared<int>(PrincipledBSDFDiffuseLobe);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENFORCE_ENERGY_CONSERVATION] = std::make_shared<int>(PrincipledBSDFEnforceStrongEnergyConservation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoGlassEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoClearcoatEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoMetallicEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoMetallicFresnelEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoSpecularEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION] = std::make_shared<int>(PrincipledBSDFDeltaDistributionEvaluationOptimization);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL] = std::make_shared<int>(PrincipledBSDFSampleGlossyBasedOnFresnel);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL] = std::make_shared<int>(PrincipledBSDFSampleCoatBasedOnFresnel);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION] = std::make_shared<int>(PrincipledBSDFDoMicrofacetRegularization);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION] = std::make_shared<int>(PrincipledBSDFDoMicrofacetRegularizationConsistentParameterization);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC] = std::make_shared<int>(PrincipledBSDFMicrofacetRegularizationDiffusionHeuristic);
	m_options_macro_map[GPUKernelCompilerOptions::GGX_SAMPLE_FUNCTION] = std::make_shared<int>(PrincipledBSDFAnisotropicGGXSampleFunction);
	m_options_macro_map[GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION] = std::make_shared<int>(NestedDielectricsStackSize);

	m_options_macro_map[GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY] = std::make_shared<int>(TrianglePointSamplingStrategy);

	m_options_macro_map[GPUKernelCompilerOptions::REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY] = std::make_shared<int>(ReGIR_GridFillLightSamplingBaseStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_VISIBILITY] = std::make_shared<int>(ReGIR_GridFillTargetFunctionVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM] = std::make_shared<int>(ReGIR_GridFillTargetFunctionCosineTerm);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM_LIGHT_SOURCE] = std::make_shared<int>(ReGIR_GridFillTargetFunctionCosineTermLightSource);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_TARGET_FUNCTION_VISIBILITY] = std::make_shared<int>(ReGIR_ShadingResamplingTargetFunctionVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_INCLUDE_BSDF] = std::make_shared<int>(ReGIR_ShadingResamplingIncludeBSDF);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_DO_VISIBILITY_REUSE] = std::make_shared<int>(ReGIR_DoVisibilityReuse);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_FALLBACK_LIGHT_SAMPLING_STRATEGY] = std::make_shared<int>(ReGIR_FallbackLightSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::REGIR_DEBUG_MODE] = std::make_shared<int>(ReGIR_DebugMode);

	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY] = std::make_shared<int>(DirectLightSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BASE_STRATEGY] = std::make_shared<int>(DirectLightSamplingBaseStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS] = std::make_shared<int>(DirectLightUseNEEPlusPlus);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE] = std::make_shared<int>(DirectLightUseNEEPlusPlusRR);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED] = std::make_shared<int>(DirectLightNEEPlusPlusDisplayShadowRaysDiscarded);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED_BOUNCE] = std::make_shared<int>(DirectLightNEEPlusPlusDisplayShadowRaysDiscardedBounce);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION] = std::make_shared<int>(DirectLightSamplingDeltaDistributionOptimization);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS] = std::make_shared<int>(DirectLightSamplingAllowBackfacingLights);
	m_options_macro_map[GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION] = std::make_shared<int>(RISUseVisiblityTargetFunction);

	m_options_macro_map[GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY] = std::make_shared<int>(EnvmapSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS] = std::make_shared<int>(EnvmapSamplingDoBSDFMIS);
	m_options_macro_map[GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING] = std::make_shared<int>(EnvmapSamplingDoBilinearFiltering);

	m_options_macro_map[GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY] = std::make_shared<int>(PathSamplingStrategy);

	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY] = std::make_shared<int>(ReSTIR_DI_InitialTargetFunctionVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY] = std::make_shared<int>(ReSTIR_DI_SpatialTargetFunctionVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE] = std::make_shared<int>(ReSTIR_DI_DoVisibilityReuse);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY] = std::make_shared<int>(ReSTIR_DI_BiasCorrectionUseVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS] = std::make_shared<int>(ReSTIR_DI_BiasCorrectionWeights);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY] = std::make_shared<int>(ReSTIR_DI_LaterBouncesSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHT_PRESAMPLING] = std::make_shared<int>(ReSTIR_DI_DoLightPresampling);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_LIGHT_PRESAMPLING_STRATEGY] = std::make_shared<int>(ReSTIR_DI_LightPresamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT] = std::make_shared<int>(ReSTIR_DI_SpatialDirectionalReuseBitCount);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_DI_DO_OPTIMAL_VISIBILITY_SAMPLING] = std::make_shared<int>(ReSTIR_DI_DoOptimalVisibilitySampling);

	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY] = std::make_shared<int>(ReSTIR_GI_SpatialTargetFunctionVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT] = std::make_shared<int>(ReSTIR_GI_SpatialDirectionalReuseBitCount);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_USE_VISIBILITY] = std::make_shared<int>(ReSTIR_GI_BiasCorrectionUseVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_WEIGHTS] = std::make_shared<int>(ReSTIR_GI_BiasCorrectionWeights);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_DO_OPTIMAL_VISIBILITY_SAMPLING] = std::make_shared<int>(ReSTIR_GI_DoOptimalVisibilitySampling);

	m_options_macro_map[GPUKernelCompilerOptions::GMON_M_SETS_COUNT] = std::make_shared<int>(GMoNMSetsCount);
	
	// Making sure we didn't forget to fill the ALL_MACROS_NAMES vector with all the options that exist
	if (GPUKernelCompilerOptions::ALL_MACROS_NAMES.size() != m_options_macro_map.size())
		Utils::debugbreak();
}

GPUKernelCompilerOptions::GPUKernelCompilerOptions(const GPUKernelCompilerOptions& other)
{
	*this = other.deep_copy();
}

GPUKernelCompilerOptions& GPUKernelCompilerOptions::operator=(const GPUKernelCompilerOptions& other)
{
	m_options_macro_map = other.m_options_macro_map;
	m_custom_macro_map = other.m_custom_macro_map;

	return *this;
}

GPUKernelCompilerOptions GPUKernelCompilerOptions::deep_copy() const
{
	GPUKernelCompilerOptions out;
	out.clear();

	for (auto& pair : m_options_macro_map)
		// Creating new shared ptr for the copy
		out.m_options_macro_map[pair.first] = std::make_shared<int>(*pair.second);

	for (auto& pair : m_custom_macro_map)
		// Creating new shared ptr for the copy
		out.m_custom_macro_map[pair.first] = std::make_shared<int>(*pair.second);

	return out;
}

std::vector<std::string> GPUKernelCompilerOptions::get_all_macros_as_std_vector_string()
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_options_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(*macro_key_value.second));

	for (auto macro_key_value : m_custom_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(*macro_key_value.second));

	return macros;
}

std::vector<std::string> GPUKernelCompilerOptions::get_relevant_macros_as_std_vector_string(const GPUKernel* kernel) const
{
	std::vector<std::string> macros;

	// Looping on all the options macros and checking if the kernel uses that option macro,
	// only adding the macro to the returned vector if the kernel uses that option macro
	for (auto macro_key_value : m_options_macro_map)
		if (kernel->uses_macro(macro_key_value.first))
			macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(*macro_key_value.second));

	// Adding all the custom macros without conditions
	for (auto macro_key_value : m_custom_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(*macro_key_value.second));

	std::vector<std::string> additional_macros = kernel->get_additional_compiler_macros();
	for (const std::string& additional_macro : additional_macros)
		macros.push_back(additional_macro);

	return macros;
}

void GPUKernelCompilerOptions::set_macro_value(const std::string& name, int value)
{
	if (ALL_MACROS_NAMES.find(name) != ALL_MACROS_NAMES.end())
	{
		if (m_options_macro_map.find(name) != m_options_macro_map.end())
			// If you could find the name in the options-macro, settings its value
			*m_options_macro_map[name] = value;
		else
			// Otherwise, creating it
			m_options_macro_map[name] = std::make_shared<int>(value);
	}
	else
	{
		// Otherwise, this is a user defined macro, putting it in the custom macro map
		if (m_custom_macro_map.find(name) != m_custom_macro_map.end())
			// Updating the macro's alue if it already exists
			*m_custom_macro_map[name] = value;
		else
			// Creating it otherwise
			m_custom_macro_map[name] = std::make_shared<int>(value);
	}
}

void GPUKernelCompilerOptions::remove_macro(const std::string& name)
{
	// Only removing from the custom macro map because we cannot remove the options-macro
	m_custom_macro_map.erase(name);
}

bool GPUKernelCompilerOptions::has_macro(const std::string& name)
{
	// Only checking the custom macro map because we cannot remove the options-macro so it makes
	// no sense to check whether this instance has the macro "InteriorStackStrategy"
	// for example, it will always be yes
	return m_custom_macro_map.find(name) != m_custom_macro_map.end();
}

int GPUKernelCompilerOptions::get_macro_value(const std::string& name) const
{
	auto find = m_options_macro_map.find(name);

	if (find == m_options_macro_map.end())
	{
		// Wasn't found in the options-macro, trying in the custom macros
		auto find_custom = m_custom_macro_map.find(name);
		if (find_custom == m_custom_macro_map.end())
			return std::numeric_limits<int>::min();
		else
			return *find_custom->second;
	}
	else
		return *find->second;
}

const std::shared_ptr<int> GPUKernelCompilerOptions::get_pointer_to_macro_value(const std::string& name) const
{
	auto find = m_options_macro_map.find(name);

	if (find == m_options_macro_map.end())
	{
		// Wasn't found in the options-macro, trying in the custom macros
		auto find_custom = m_custom_macro_map.find(name);
		if (find_custom == m_custom_macro_map.end())
			return nullptr;
		else
			return find_custom->second;
	}
	else
		return find->second;
}

int* GPUKernelCompilerOptions::get_raw_pointer_to_macro_value(const std::string& name)
{
	std::shared_ptr<int> pointer = get_pointer_to_macro_value(name);
	if (pointer != nullptr)
		return pointer.get();

	return nullptr;
}

void GPUKernelCompilerOptions::set_pointer_to_macro(const std::string& name, std::shared_ptr<int> pointer_to_value)
{
	auto find = m_options_macro_map.find(name);

	if (find == m_options_macro_map.end())
		// Wasn't found in the options-macro, adding/setting it in the custom macro map
		m_custom_macro_map[name] = pointer_to_value;
	else
		m_options_macro_map[name] = pointer_to_value;
}

const std::unordered_map<std::string, std::shared_ptr<int>>& GPUKernelCompilerOptions::get_options_macro_map() const
{
	return m_options_macro_map;
}

const std::unordered_map<std::string, std::shared_ptr<int>>& GPUKernelCompilerOptions::get_custom_macro_map() const
{
	return m_custom_macro_map;
}

void GPUKernelCompilerOptions::clear()
{
	m_custom_macro_map.clear();
	m_options_macro_map.clear();
}

void GPUKernelCompilerOptions::apply_onto(GPUKernelCompilerOptions& other)
{
	for (auto& pair : m_options_macro_map)
	{
		if (other.m_options_macro_map.find(pair.first) == other.m_options_macro_map.end())
			// The option doesn't exist, we need to create the shared ptr
			other.m_options_macro_map[pair.first] = std::make_shared<int>(*pair.second);
		else
			// No need to create a shared ptr, we can just copy the value
			*other.m_options_macro_map[pair.first] = *pair.second;
	}

	for (auto& pair : m_custom_macro_map)
	{
		if (other.m_custom_macro_map.find(pair.first) == other.m_custom_macro_map.end())
			// The option doesn't exist, we need to create the shared ptr
			other.m_custom_macro_map[pair.first] = std::make_shared<int>(*pair.second);
		else
			// No need to create a shared ptr, we can just copy the value
			*other.m_custom_macro_map[pair.first] = *pair.second;
	}
}
