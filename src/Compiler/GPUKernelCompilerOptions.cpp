/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/KernelOptions/HeatmapOptions.h"
#include "HostDeviceCommon/KernelOptions/IlluminationAwareKDTreeLearningToClusterOptions.h"
#include "HostDeviceCommon/KernelOptions/MegakernelOptions.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"
#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "Utils/Utils.h"

#include <cassert>

/**
 * Defining the strings that go with the option so that they can be passed to the shader compiler
 * with the -D<string>=<value> option.
 *
 * The strings used here must match the ones used in KernelOptions.h
 */

// clang-format off
const std::string GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL = "UseSharedStackBVHTraversal";
const std::string GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE = "SharedStackBVHTraversalSize";
const std::string GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_BLOCK_SIZE = "KernelWorkgroupThreadCount";

const std::string GPUKernelCompilerOptions::BSDF_MODEL = "BSDF_MODEL";
const std::string GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION = "KERNEL_MATERIAL_SPECIALIZATION";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE = "PrincipledBSDFDiffuseLobe";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_ANISOTROPIC_GGX_SAMPLE_FUNCTION = "PrincipledBSDFAnisotropicGGXSampleFunction";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_METALLIC_SAMPLE_COSINE_WEIGHTED = "PrincipledBSDFMetallicSampleCosineWeighted";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION = "PrincipledBSDFDoEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENERGY_COMPENSATION_MODE = "PrincipledBSDFEnergyCompensationMode";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_BOUNCES = "PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_VARIABLE_BOUNCES = "PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceVariableBounces";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_DO_RUSSIAN_ROULETTE = "PrincipledBSDFMultipleScatteringCuiDoRussianRoulette";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION = "PrincipledBSDFDoGlassEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION = "PrincipledBSDFDoClearcoatEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION = "PrincipledBSDFDoMetallicEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION = "PrincipledBSDFDoMetallicFresnelEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION = "PrincipledBSDFDoSpecularEnergyCompensation";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION = "PrincipledBSDFDeltaDistributionEvaluationOptimization";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL = "PrincipledBSDFSampleGlossyBasedOnFresnel";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL = "PrincipledBSDFSampleCoatBasedOnFresnel";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_DIFFUSE_LUMINANCE = "PrincipledBSDFSampleDiffuseLuminance";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION = "PrincipledBSDFDoMicrofacetRegularization";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION = "PrincipledBSDFDoMicrofacetRegularizationConsistentParameterization";
const std::string GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC = "PrincipledBSDFMicrofacetRegularizationDiffusionHeuristic";
const std::string GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION = "NestedDielectricsStackSize";

const std::string GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY = "TrianglePointSamplingStrategy";
const std::string GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE_USE_LTC = "TrianglePointSamplingStrategySolidAngleUseLTC";
const std::string GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_STRATEGY = "TrianglePointSamplingUniformAreaStrategy";

const std::string GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES = "UseMaterialTextures";
const std::string GPUKernelCompilerOptions::USE_MATERIAL_BASE_COLOR_TEXTURE_OVERRIDE = "UseMaterialBaseColorTextureOverride";

const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR = "DirectLightNEEEstimator";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY = "DirectLightSamplingStrategy";
const std::string GPUKernelCompilerOptions::WAVEFRONT_DEFER_NEE_SHADOW_TRAVERSAL = "WavefrontDeferNEEShadowTraversal";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS = "DirectLightUseNEEPlusPlus";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_GRID_PREPOPULATE_LIGHT_SAMPLING_STRATEGY = "NEEPlusPlusGridPrepopulateLightSamplingStrategy";
const std::string GPUKernelCompilerOptions::NEE_PLUS_PLUS_LINEAR_PROBING_STEPS = "NEEPlusPlus_LinearProbingSteps";
const std::string GPUKernelCompilerOptions::NEE_PLUS_PLUS_DEBUG_MODE = "NEEPlusPlusDebugMode";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION = "DirectLightSamplingDeltaDistributionOptimization";
const std::string GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS = "DirectLightSamplingAllowBackfacingLights";
const std::string GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION = "RISUseVisiblityTargetFunction";

const std::string GPUKernelCompilerOptions::LIGHT_TREE_ATS_IMPORTANCE_FUNCTION_USE_ORIENTATION = "LightTreeATSImportanceFunctionUseOrientation";
const std::string GPUKernelCompilerOptions::LIGHT_TREE_ATS_DO_SPLITTING = "LightTreeATSDoSplitting";
const std::string GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES = "LightTreeATSSplittingMaxLightSamples";

const std::string GPUKernelCompilerOptions::LIGHT_TREE_ATS_SG_DO_SPECULAR_IMPORTANCE = "LightTreeSGDoSpecularImportance";

const std::string GPUKernelCompilerOptions::LIGHT_TREE_SG_DO_SPLITTING = "LightTreeSGDoSplitting";
const std::string GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_NEW_SPLITTING_MODEL = "LightTreeSGUseNewSplittingModel";
const std::string GPUKernelCompilerOptions::LIGHT_TREE_SG_NEW_SPLITTING_MODEL_ALWAYS_SPLIT_FIRST_CANDIDATE = "LightTreeSGNewSplittingModelAlwaysSplitFirstCandidate";
const std::string GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES = "LightTreeSGSplittingMaxLightSamples";
const std::string GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_TREE_CUT = "LightTreeSGUseTreeCut";

const std::string GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT = "IlluminationAwareKDTreeMaximumLookaheadLevelCount";
const std::string GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_MAXIMUM_LIGHT_CUT_SIZE = "LearningToClusterMaximumLightCutSize";
const std::string GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_Q0_USE_TOTAL_POWER = "LearningToClusterQ0UseTotalPower";
const std::string GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_DEBUG_MODE = "LearningToClusterDebugMode";
const std::string GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE = "IlluminationAwareKDTreeDebugMode";
const std::string GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_DEBUG_REPRESENTATIVE_POINTS = "IlluminationAwareKDTreeDebugRepresentativePoints";
const std::string GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_DEBUG_MODE_HEATMAP_INDEX = "LearningToClusterDebugModeHeatmapIndex";

const std::string GPUKernelCompilerOptions::NISML_USE_SG_IMPORTANCES_KD_TREE_CACHES = "NISMLUseSGImportancesKDTreeCaches";
const std::string GPUKernelCompilerOptions::NISML_DEBUG_MODE = "NISMLDebugMode";
const std::string GPUKernelCompilerOptions::NISML_DEBUG_MODE_HEATMAP_INDEX = "NISMLDebugModeHeatmapIndex";

const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY = "EnvmapSamplingStrategy";
const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS = "EnvmapSamplingDoBSDFMIS";
const std::string GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING = "EnvmapSamplingDoBilinearFiltering";

const std::string GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY = "PathSamplingStrategy";

const std::string GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY = "ReSTIR_GI_SpatialTargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_USE_VISIBILITY = "ReSTIR_GI_MISWeightsUseVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_TYPE = "ReSTIR_GI_MISWeightsType";
const std::string GPUKernelCompilerOptions::RESTIR_GI_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED = "ReSTIR_GI_DebugViewShadeOnlyInitialCandidatesEnabled";

const std::string GPUKernelCompilerOptions::RESTIR_PT_SPATIAL_TARGET_FUNCTION_VISIBILITY = "ReSTIR_PT_SpatialTargetFunctionVisibility";
const std::string GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE = "ReSTIR_PT_MISWeightsType";
const std::string GPUKernelCompilerOptions::RESTIR_PT_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED = "ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled";

const std::string GPUKernelCompilerOptions::RESTIR_PG_ENABLE = "ReSTIRPGEnable";
const std::string GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT = "ReSTIRPGDistributionComponentCount";
const std::string GPUKernelCompilerOptions::RESTIR_PG_HASH_GRID_COLLISION_RESOLVE_STEPS = "ReSTIRPGHashGridCollisionResolveSteps";
const std::string GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE = "ReSTIRPGDebugMode";

const std::string GPUKernelCompilerOptions::GMON_M_SETS_COUNT = "GMoNMSetsCount";

const std::string GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE = "SSBNPermutationBlockSize";
const std::string GPUKernelCompilerOptions::SSBN_PERMUTATION_ENABLED = "SSBNPermutationEnabled";
const std::string GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_HASH_GRID = "SSBNPermutationDebugHashGrid";
const std::string GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_SEEDS = "SSBNPermutationDebugSeeds";

// clang-format on

const std::unordered_set<std::string> GPUKernelCompilerOptions::ALL_MACROS_NAMES = {
	GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
	GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE,
	GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_BLOCK_SIZE,

	GPUKernelCompilerOptions::BSDF_MODEL,
	GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_ANISOTROPIC_GGX_SAMPLE_FUNCTION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_METALLIC_SAMPLE_COSINE_WEIGHTED,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENERGY_COMPENSATION_MODE,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_BOUNCES,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_VARIABLE_BOUNCES,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_DO_RUSSIAN_ROULETTE,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_DIFFUSE_LUMINANCE,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION,
	GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC,
	GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION,

	GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE_USE_LTC,
	GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_STRATEGY,

	GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES,
	GPUKernelCompilerOptions::USE_MATERIAL_BASE_COLOR_TEXTURE_OVERRIDE,

	GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR,
	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::WAVEFRONT_DEFER_NEE_SHADOW_TRAVERSAL,
	GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS,
	GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_GRID_PREPOPULATE_LIGHT_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::NEE_PLUS_PLUS_LINEAR_PROBING_STEPS,
	GPUKernelCompilerOptions::NEE_PLUS_PLUS_DEBUG_MODE,
	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION,
	GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS,
	GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION,

	GPUKernelCompilerOptions::LIGHT_TREE_ATS_IMPORTANCE_FUNCTION_USE_ORIENTATION,
	GPUKernelCompilerOptions::LIGHT_TREE_ATS_DO_SPLITTING,
	GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES,

	GPUKernelCompilerOptions::LIGHT_TREE_ATS_SG_DO_SPECULAR_IMPORTANCE,

	GPUKernelCompilerOptions::LIGHT_TREE_SG_DO_SPLITTING,
	GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_NEW_SPLITTING_MODEL,
	GPUKernelCompilerOptions::LIGHT_TREE_SG_NEW_SPLITTING_MODEL_ALWAYS_SPLIT_FIRST_CANDIDATE,
	GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES,
	GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_TREE_CUT,

	GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT,
	GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_MAXIMUM_LIGHT_CUT_SIZE,
	GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_Q0_USE_TOTAL_POWER,
	GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_DEBUG_MODE,
	GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE,
	GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_DEBUG_REPRESENTATIVE_POINTS,
	GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_DEBUG_MODE_HEATMAP_INDEX,

	GPUKernelCompilerOptions::NISML_USE_SG_IMPORTANCES_KD_TREE_CACHES,
	GPUKernelCompilerOptions::NISML_DEBUG_MODE,
	GPUKernelCompilerOptions::NISML_DEBUG_MODE_HEATMAP_INDEX,

	GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY,
	GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS,
	GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING,

	GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY,

	GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_USE_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_TYPE,
	GPUKernelCompilerOptions::RESTIR_GI_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED,

	GPUKernelCompilerOptions::RESTIR_PT_SPATIAL_TARGET_FUNCTION_VISIBILITY,
	GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE,
	GPUKernelCompilerOptions::RESTIR_PT_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED,

	GPUKernelCompilerOptions::RESTIR_PG_ENABLE,
	GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT,
	GPUKernelCompilerOptions::RESTIR_PG_HASH_GRID_COLLISION_RESOLVE_STEPS,
	GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE,

	GPUKernelCompilerOptions::GMON_M_SETS_COUNT,

	GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE,
	GPUKernelCompilerOptions::SSBN_PERMUTATION_ENABLED,
	GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_HASH_GRID,
	GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_SEEDS,
};

GPUKernelCompilerOptions::GPUKernelCompilerOptions()
{
	// Mandatory options that every kernel must have so we're
	// adding them here with their default values

	// clang-format off
	m_options_macro_map[GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL] = std::make_shared<int>(UseSharedStackBVHTraversal);
	m_options_macro_map[GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE] = std::make_shared<int>(SharedStackBVHTraversalSize);
	m_options_macro_map[GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_BLOCK_SIZE] = std::make_shared<int>(KernelWorkgroupThreadCount);

	m_options_macro_map[GPUKernelCompilerOptions::BSDF_MODEL] = std::make_shared<int>(BSDF_PRINCIPLED);
	m_options_macro_map[GPUKernelCompilerOptions::KERNEL_MATERIAL_SPECIALIZATION_OPTION] =
		std::make_shared<int>(KernelMaterialSpecializationAll);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE] = std::make_shared<int>(PrincipledBSDFDiffuseLobe);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_ANISOTROPIC_GGX_SAMPLE_FUNCTION] = std::make_shared<int>(PrincipledBSDFAnisotropicGGXSampleFunction);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_METALLIC_SAMPLE_COSINE_WEIGHTED] = std::make_shared<int>(PrincipledBSDFMetallicSampleCosineWeighted);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENERGY_COMPENSATION_MODE] = std::make_shared<int>(PrincipledBSDFEnergyCompensationMode);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_BOUNCES] = std::make_shared<int>(PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceBounces);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_VARIABLE_BOUNCES] = std::make_shared<int>(PrincipledBSDFMultipleScatteringCuiMaxMicrosurfaceVariableBounces);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_DO_RUSSIAN_ROULETTE] = std::make_shared<int>(PrincipledBSDFMultipleScatteringCuiDoRussianRoulette);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoGlassEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoClearcoatEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoMetallicEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoMetallicFresnelEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION] = std::make_shared<int>(PrincipledBSDFDoSpecularEnergyCompensation);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION] = std::make_shared<int>(PrincipledBSDFDeltaDistributionEvaluationOptimization);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL] = std::make_shared<int>(PrincipledBSDFSampleGlossyBasedOnFresnel);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL] = std::make_shared<int>(PrincipledBSDFSampleCoatBasedOnFresnel);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_DIFFUSE_LUMINANCE] = std::make_shared<int>(PrincipledBSDFSampleDiffuseLuminance);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION] = std::make_shared<int>(PrincipledBSDFDoMicrofacetRegularization);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION] = std::make_shared<int>(PrincipledBSDFDoMicrofacetRegularizationConsistentParameterization);
	m_options_macro_map[GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC] = std::make_shared<int>(PrincipledBSDFMicrofacetRegularizationDiffusionHeuristic);
	m_options_macro_map[GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION] = std::make_shared<int>(NestedDielectricsStackSize);

	m_options_macro_map[GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY] = std::make_shared<int>(TrianglePointSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE_USE_LTC] = std::make_shared<int>(TrianglePointSamplingStrategySolidAngleUseLTC);
	m_options_macro_map[GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_STRATEGY] = std::make_shared<int>(TrianglePointSamplingUniformAreaStrategy);

	m_options_macro_map[GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES] = std::make_shared<int>(UseMaterialTextures);
	m_options_macro_map[GPUKernelCompilerOptions::USE_MATERIAL_BASE_COLOR_TEXTURE_OVERRIDE] = std::make_shared<int>(UseMaterialBaseColorTextureOverride);

	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR] = std::make_shared<int>(DirectLightNEEEstimator);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY] = std::make_shared<int>(DirectLightSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::WAVEFRONT_DEFER_NEE_SHADOW_TRAVERSAL] = std::make_shared<int>(WavefrontDeferNEEShadowTraversal);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS] = std::make_shared<int>(DirectLightUseNEEPlusPlus);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_GRID_PREPOPULATE_LIGHT_SAMPLING_STRATEGY] = std::make_shared<int>(NEEPlusPlusGridPrepopulateLightSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::NEE_PLUS_PLUS_LINEAR_PROBING_STEPS] = std::make_shared<int>(NEEPlusPlus_LinearProbingSteps);
	m_options_macro_map[GPUKernelCompilerOptions::NEE_PLUS_PLUS_DEBUG_MODE] = std::make_shared<int>(NEEPlusPlusDebugMode);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION] = std::make_shared<int>(DirectLightSamplingDeltaDistributionOptimization);
	m_options_macro_map[GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS] = std::make_shared<int>(DirectLightSamplingAllowBackfacingLights);
	m_options_macro_map[GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION] = std::make_shared<int>(RISUseVisiblityTargetFunction);

	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_ATS_IMPORTANCE_FUNCTION_USE_ORIENTATION] = std::make_shared<int>(LightTreeATSImportanceFunctionUseOrientation);
	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_ATS_DO_SPLITTING] = std::make_shared<int>(LightTreeATSDoSplitting);
	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES] = std::make_shared<int>(LightTreeATSSplittingMaxLightSamples);

	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_ATS_SG_DO_SPECULAR_IMPORTANCE] = std::make_shared<int>(LightTreeSGDoSpecularImportance);

	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_SG_DO_SPLITTING] = std::make_shared<int>(LightTreeSGDoSplitting);
	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_NEW_SPLITTING_MODEL] = std::make_shared<int>(LightTreeSGUseNewSplittingModel);
	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_SG_NEW_SPLITTING_MODEL_ALWAYS_SPLIT_FIRST_CANDIDATE] = std::make_shared<int>(LightTreeSGNewSplittingModelAlwaysSplitFirstCandidate);
	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES] = std::make_shared<int>(LightTreeSGSplittingMaxLightSamples);
	m_options_macro_map[GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_TREE_CUT] = std::make_shared<int>(LightTreeSGUseTreeCut);

	m_options_macro_map[GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT] =
		std::make_shared<int>(IlluminationAwareKDTreeMaximumLookaheadLevelCount);
	m_options_macro_map[GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_MAXIMUM_LIGHT_CUT_SIZE] =
		std::make_shared<int>(LearningToClusterMaximumLightCutSize);
	m_options_macro_map[GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_Q0_USE_TOTAL_POWER] =
		std::make_shared<int>(LearningToClusterQ0UseTotalPower);
	m_options_macro_map[GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_DEBUG_MODE] = std::make_shared<int>(LearningToClusterDebugMode);
	m_options_macro_map[GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE] = std::make_shared<int>(IlluminationAwareKDTreeDebugMode);
	m_options_macro_map[GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_DEBUG_REPRESENTATIVE_POINTS] =
		std::make_shared<int>(IlluminationAwareKDTreeDebugRepresentativePoints);
	m_options_macro_map[GPUKernelCompilerOptions::LEARNING_TO_CLUSTER_DEBUG_MODE_HEATMAP_INDEX] =
		std::make_shared<int>(LearningToClusterDebugModeHeatmapIndex);

	m_options_macro_map[GPUKernelCompilerOptions::NISML_USE_SG_IMPORTANCES_KD_TREE_CACHES] = std::make_shared<int>(NISMLUseSGImportancesKDTreeCaches);
	m_options_macro_map[GPUKernelCompilerOptions::NISML_DEBUG_MODE] = std::make_shared<int>(NISMLDebugMode);
	m_options_macro_map[GPUKernelCompilerOptions::NISML_DEBUG_MODE_HEATMAP_INDEX] = std::make_shared<int>(NISMLDebugModeHeatmapIndex);

	m_options_macro_map[GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY] = std::make_shared<int>(EnvmapSamplingStrategy);
	m_options_macro_map[GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS] = std::make_shared<int>(EnvmapSamplingDoBSDFMIS);
	m_options_macro_map[GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING] = std::make_shared<int>(EnvmapSamplingDoBilinearFiltering);

	m_options_macro_map[GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY] = std::make_shared<int>(PathSamplingStrategy);

	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY] = std::make_shared<int>(ReSTIR_GI_SpatialTargetFunctionVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_USE_VISIBILITY] = std::make_shared<int>(ReSTIR_GI_MISWeightsUseVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_TYPE] = std::make_shared<int>(ReSTIR_GI_MISWeightsType);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_GI_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED] = std::make_shared<int>(ReSTIR_GI_DebugViewShadeOnlyInitialCandidatesEnabled);

	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_PT_SPATIAL_TARGET_FUNCTION_VISIBILITY] = std::make_shared<int>(ReSTIR_PT_SpatialTargetFunctionVisibility);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE] = std::make_shared<int>(ReSTIR_PT_MISWeightsType);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_PT_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED] = std::make_shared<int>(ReSTIR_PT_DebugViewShadeOnlyInitialCandidatesEnabled);

	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_PG_ENABLE] = std::make_shared<int>(ReSTIRPGEnable);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT] = std::make_shared<int>(ReSTIRPGDistributionComponentCount);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_PG_HASH_GRID_COLLISION_RESOLVE_STEPS] = std::make_shared<int>(ReSTIRPGHashGridCollisionResolveSteps);
	m_options_macro_map[GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE] = std::make_shared<int>(ReSTIRPGDebugMode);

	m_options_macro_map[GPUKernelCompilerOptions::GMON_M_SETS_COUNT] = std::make_shared<int>(GMoNMSetsCount);

	m_options_macro_map[GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE] = std::make_shared<int>(SSBNPermutationBlockSize);
	m_options_macro_map[GPUKernelCompilerOptions::SSBN_PERMUTATION_ENABLED] = std::make_shared<int>(SSBNPermutationEnabled);
	m_options_macro_map[GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_HASH_GRID] = std::make_shared<int>(SSBNPermutationDebugHashGrid);
	m_options_macro_map[GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_SEEDS] = std::make_shared<int>(SSBNPermutationDebugSeeds);
	// clang-format on

	// Making sure we didn't forget to fill the ALL_MACROS_NAMES vector with all the options that exist
	if (GPUKernelCompilerOptions::ALL_MACROS_NAMES.size() != m_options_macro_map.size())
		Debug::debugbreak();
}

GPUKernelCompilerOptions::GPUKernelCompilerOptions(const GPUKernelCompilerOptions& other)
{
	*this = other.deep_copy();
}

GPUKernelCompilerOptions& GPUKernelCompilerOptions::operator=(const GPUKernelCompilerOptions& other)
{
	m_options_macro_map		  = other.m_options_macro_map;
	m_custom_macro_map		  = other.m_custom_macro_map;
	m_custom_string_macro_map = other.m_custom_string_macro_map;

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

	for (auto& pair : m_custom_string_macro_map)
		// Creating new shared ptr for the copy
		out.m_custom_string_macro_map[pair.first] = std::make_pair(std::make_shared<std::string>(*pair.second.first), pair.second.second);

	return out;
}

std::vector<std::string> GPUKernelCompilerOptions::get_all_macros_as_std_vector_string()
{
	std::vector<std::string> macros;

	for (auto macro_key_value : m_options_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(*macro_key_value.second));

	for (auto macro_key_value : m_custom_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + std::to_string(*macro_key_value.second));

	for (auto macro_key_value : m_custom_string_macro_map)
		macros.push_back("-D " + macro_key_value.first + "=" + (macro_key_value.second.second ? "\"" : "") + (*macro_key_value.second.first) +
						 (macro_key_value.second.second ? "\"" : ""));

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

void GPUKernelCompilerOptions::set_macro_value_independently(const std::string& name, int value)
{
	std::shared_ptr<int> independent_value = std::make_shared<int>(value);

	if (ALL_MACROS_NAMES.find(name) != ALL_MACROS_NAMES.end())
		m_options_macro_map[name] = independent_value;
	else
		m_custom_macro_map[name] = independent_value;
}

void GPUKernelCompilerOptions::set_string_macro_value(const std::string& name, const std::string& value, bool with_quotes)
{
	// User defined string macro, putting it in the custom string macro map
	if (m_custom_string_macro_map.find(name) != m_custom_string_macro_map.end())
	{
		// Updating the macro's value if it already exists
		*m_custom_string_macro_map[name].first = value;
		m_custom_string_macro_map[name].second = with_quotes;
	}
	else
		// Creating it otherwise
		m_custom_string_macro_map[name] = std::make_pair(std::make_shared<std::string>(value), with_quotes);
}

void GPUKernelCompilerOptions::remove_macro(const std::string& name)
{
	// Only removing from the custom macro map because we cannot remove the options-macro
	m_custom_macro_map.erase(name);
	m_custom_string_macro_map.erase(name);
}

bool GPUKernelCompilerOptions::has_macro(const std::string& name)
{

	// Only checking the custom macro map because we cannot remove the options-macro so it makes
	// no sense to check whether this instance has the macro "InteriorStackStrategy"
	// for example, it will always be yes
	return m_custom_macro_map.find(name) != m_custom_macro_map.end() || m_custom_string_macro_map.find(name) != m_custom_string_macro_map.end();
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

std::string GPUKernelCompilerOptions::get_string_macro_value(const std::string& name) const
{
	auto find = m_custom_string_macro_map.find(name);
	if (find == m_custom_string_macro_map.end())

		return "";
	else

		return *find->second.first;
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

const std::shared_ptr<std::string> GPUKernelCompilerOptions::get_pointer_to_string_macro_value(const std::string& name) const
{
	auto find = m_custom_string_macro_map.find(name);
	if (find == m_custom_string_macro_map.end())

		return nullptr;
	else

		return find->second.first;
}

int* GPUKernelCompilerOptions::get_raw_pointer_to_macro_value(const std::string& name)
{
	std::shared_ptr<int> pointer = get_pointer_to_macro_value(name);
	if (pointer != nullptr)

		return pointer.get();

	return nullptr;
}

std::string* GPUKernelCompilerOptions::get_raw_pointer_to_string_macro_value(const std::string& name)
{
	std::shared_ptr<std::string> pointer = get_pointer_to_string_macro_value(name);
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

void GPUKernelCompilerOptions::set_pointer_to_string_macro(const std::string& name, std::shared_ptr<std::string> pointer_to_value)
{
	// Adding/setting it in the custom string macro map because the options-macro cannot contain string macros
	m_custom_string_macro_map[name].first = pointer_to_value;
}

const std::map<std::string, std::shared_ptr<int>>& GPUKernelCompilerOptions::get_options_macro_map() const
{

	return m_options_macro_map;
}

const std::map<std::string, std::shared_ptr<int>>& GPUKernelCompilerOptions::get_custom_macro_map() const
{

	return m_custom_macro_map;
}

const std::map<std::string, std::pair<std::shared_ptr<std::string>, bool>>& GPUKernelCompilerOptions::get_custom_string_macro_map() const
{

	return m_custom_string_macro_map;
}

void GPUKernelCompilerOptions::clear()
{
	m_options_macro_map.clear();
	m_custom_macro_map.clear();
	m_custom_string_macro_map.clear();
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

	for (auto& pair : m_custom_string_macro_map)
	{
		if (other.m_custom_string_macro_map.find(pair.first) == other.m_custom_string_macro_map.end())
			// The option doesn't exist, we need to create the shared ptr
			other.m_custom_string_macro_map[pair.first] = std::make_pair(std::make_shared<std::string>(*pair.second.first), pair.second.second);
		else
			// No need to create a shared ptr, we can just copy the value
			*other.m_custom_string_macro_map[pair.first].first = *pair.second.first;
	}
}
