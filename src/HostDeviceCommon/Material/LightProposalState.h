/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_LIGHT_PROPOSAL_STATE_H
#define HOST_DEVICE_COMMON_MATERIAL_LIGHT_PROPOSAL_STATE_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"

struct LightProposalInputs
{
	float base_color_luminance;
	float roughness;
	float ior;
	float metallic;
	float specular;
	float coat;
	float coat_roughness;
	float coat_ior;
	float specular_transmission;
	float diffuse_transmission;
	float anisotropy;
	float coat_anisotropy;
};

struct LightTreeSGProposalState
{
	float sg_specular_weight;
	float alpha_x;
	float alpha_y;
};

struct LTCProposalState
{
	float base_color_luminance;
	float roughness;
	float ior;
	float metallic;
	float specular;
	float coat;
	float coat_roughness;
	float coat_ior;
};

HIPRT_HOST_DEVICE inline float get_ltc_base_color_luminance(const LTCProposalState& proposal_state)
{
	return proposal_state.base_color_luminance;
}

HIPRT_HOST_DEVICE inline float get_ltc_base_color_luminance(const DeviceUnpackedPrincipledFullMaterial& material)
{
	return material.base_color.luminance();
}

template <bool has_sg_state, bool has_ltc_state>
struct LightProposalStateComponents
{
};

template <>
struct LightProposalStateComponents<true, false> : LightTreeSGProposalState
{
};

template <>
struct LightProposalStateComponents<false, true> : LTCProposalState
{
};

template <>
struct LightProposalStateComponents<true, true> : LightTreeSGProposalState, LTCProposalState
{
};

template <int light_sampling_strategy, int triangle_point_sampling_strategy, BSDFModel model>
struct LightProposalStateFor
	: LightProposalStateComponents<light_sampling_strategy == LSS_BASE_LIGHT_TREE_SG || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator),
								   DirectLightNEEEstimator == LSS_RISLTC ||
									   (TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE &&
										(triangle_point_sampling_strategy == TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE ||
										 triangle_point_sampling_strategy == TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE))>
{
	static_assert(model == BSDFModel::Lambertian || model == BSDFModel::OrenNayar || model == BSDFModel::Principled);
	static constexpr bool has_sg_state = light_sampling_strategy == LSS_BASE_LIGHT_TREE_SG || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator);
	static constexpr bool has_ltc_state =
		DirectLightNEEEstimator == LSS_RISLTC || (TrianglePointSamplingStrategySolidAngleUseLTC == KERNEL_OPTION_TRUE &&
												  (triangle_point_sampling_strategy == TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE ||
												   triangle_point_sampling_strategy == TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE));
};

HIPRT_HOST_DEVICE inline void get_sg_specular_importance_parameters(const LightProposalInputs& proposal_inputs,
																	float& sg_specular_weight,
																	float& alpha_x,
																	float& alpha_y)
{
	float material_specular_weight = (1.0f - proposal_inputs.metallic) *
									 (1.0f - proposal_inputs.specular_transmission * (1.0f - proposal_inputs.diffuse_transmission)) * proposal_inputs.specular;

	float specular_lobes_sum = proposal_inputs.coat + proposal_inputs.metallic + material_specular_weight;
	sg_specular_weight		 = hippt::max(proposal_inputs.coat, hippt::max(proposal_inputs.metallic, material_specular_weight));
	float sg_roughness		 = hippt::max(MaterialConstants::ROUGHNESS_CLAMP,
										  (proposal_inputs.coat * proposal_inputs.coat_roughness + proposal_inputs.metallic * proposal_inputs.roughness +
									   material_specular_weight * proposal_inputs.roughness) /
											  specular_lobes_sum);
	float sg_anisotropy		 = (proposal_inputs.coat * proposal_inputs.coat_anisotropy + proposal_inputs.metallic * proposal_inputs.anisotropy +
							material_specular_weight * proposal_inputs.anisotropy) /
						  specular_lobes_sum;

	MaterialUtils::get_alphas(sg_roughness, sg_anisotropy, alpha_x, alpha_y);
}

HIPRT_HOST_DEVICE inline LightTreeSGProposalState make_light_tree_sg_proposal_state(const LightProposalInputs& proposal_inputs)
{
	LightTreeSGProposalState proposal_state{};
	get_sg_specular_importance_parameters(proposal_inputs, proposal_state.sg_specular_weight, proposal_state.alpha_x, proposal_state.alpha_y);
	return proposal_state;
}

template <int light_sampling_strategy, int triangle_point_sampling_strategy, BSDFModel model>
HIPRT_HOST_DEVICE LightProposalStateFor<light_sampling_strategy, triangle_point_sampling_strategy, model> make_light_proposal_state(
	const LightProposalInputs& proposal_inputs)
{
	LightProposalStateFor<light_sampling_strategy, triangle_point_sampling_strategy, model> proposal_state{};

	if constexpr (LightProposalStateFor<light_sampling_strategy, triangle_point_sampling_strategy, model>::has_sg_state)
		get_sg_specular_importance_parameters(proposal_inputs, proposal_state.sg_specular_weight, proposal_state.alpha_x, proposal_state.alpha_y);

	if constexpr (LightProposalStateFor<light_sampling_strategy, triangle_point_sampling_strategy, model>::has_ltc_state)
	{
		proposal_state.base_color_luminance = proposal_inputs.base_color_luminance;
		proposal_state.roughness			= proposal_inputs.roughness;
		proposal_state.ior					= proposal_inputs.ior;
		proposal_state.metallic				= proposal_inputs.metallic;
		proposal_state.specular				= proposal_inputs.specular;
		proposal_state.coat					= proposal_inputs.coat;
		proposal_state.coat_roughness		= proposal_inputs.coat_roughness;
		proposal_state.coat_ior				= proposal_inputs.coat_ior;
	}

	return proposal_state;
}

HIPRT_HOST_DEVICE inline LightProposalInputs make_light_proposal_inputs(const DeviceUnpackedPrincipledFullMaterial& material)
{
	LightProposalInputs proposal_inputs{};
	proposal_inputs.base_color_luminance  = material.base_color.luminance();
	proposal_inputs.roughness			  = material.roughness;
	proposal_inputs.ior					  = material.ior;
	proposal_inputs.metallic			  = material.metallic;
	proposal_inputs.specular			  = material.specular;
	proposal_inputs.coat				  = material.coat;
	proposal_inputs.coat_roughness		  = material.coat_roughness;
	proposal_inputs.coat_ior			  = material.coat_ior;
	proposal_inputs.specular_transmission = material.specular_transmission;
	proposal_inputs.diffuse_transmission  = material.diffuse_transmission;
	proposal_inputs.anisotropy			  = material.anisotropy;
	proposal_inputs.coat_anisotropy		  = material.coat_anisotropy;
	return proposal_inputs;
}

#endif // #ifndef HOST_DEVICE_COMMON_MATERIAL_LIGHT_PROPOSAL_STATE_H
