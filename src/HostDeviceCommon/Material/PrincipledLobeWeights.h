/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_PRINCIPLED_LOBE_WEIGHTS_H
#define HOST_DEVICE_COMMON_PRINCIPLED_LOBE_WEIGHTS_H

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/Maths/Math.h"
#include "HostDeviceCommon/Material/PrincipledLobeSupport.h"

struct PrincipledLobeUserWeights
{
	float coat					  = 0.0f;
	float sheen					  = 0.0f;
	float metallic				  = 0.0f;
	float second_roughness_weight = 0.0f;
	float retro_reflection		  = 0.0f;
	float specular				  = 0.0f;
	float specular_transmission	  = 0.0f;
	float diffuse_transmission	  = 0.0f;
};

enum ResolvedMaterialUserControlValidityBits : unsigned int
{
	ResolvedMaterialUserControlRoughness			= 1u << 0,
	ResolvedMaterialUserControlMetallic				= 1u << 1,
	ResolvedMaterialUserControlSpecular				= 1u << 2,
	ResolvedMaterialUserControlCoat					= 1u << 3,
	ResolvedMaterialUserControlSheen				= 1u << 4,
	ResolvedMaterialUserControlSpecularTransmission = 1u << 5
};

struct ResolvedMaterialUserControlsCache
{
	float roughness				= 0.0f;
	float metallic				= 0.0f;
	float specular				= 0.0f;
	float coat					= 0.0f;
	float sheen					= 0.0f;
	float specular_transmission = 0.0f;
	unsigned int validity_mask	= 0u;
};

HIPRT_DEVICE inline PrincipledLobeWeights compute_principled_lobe_weights(const PrincipledLobeUserWeights& user_weights, bool outside_object)
{
	PrincipledLobeWeights weights;

	weights.coat  = user_weights.coat * outside_object;
	weights.sheen = user_weights.sheen * outside_object;

	weights.metallic_first	= user_weights.metallic * outside_object;
	weights.metallic_second = user_weights.metallic * outside_object;
	weights.metallic_first	= hippt::lerp(weights.metallic_first, 0.0f, user_weights.second_roughness_weight);
	weights.metallic_second = hippt::lerp(0.0f, weights.metallic_second, user_weights.second_roughness_weight);

	float retro_reflection	 = user_weights.retro_reflection * user_weights.metallic;
	weights.retro_reflection = weights.metallic_first * retro_reflection * outside_object;
	weights.metallic_first	 = hippt::lerp(weights.metallic_first, 0.0f, retro_reflection);

	weights.glass				 = !outside_object ? (1.0f - user_weights.diffuse_transmission)
												   : (1.0f - user_weights.metallic) * (1.0f - user_weights.diffuse_transmission) * user_weights.specular_transmission;
	weights.diffuse_transmission = !outside_object ? user_weights.diffuse_transmission : (1.0f - user_weights.metallic) * user_weights.diffuse_transmission;

	weights.specular = (1.0f - user_weights.metallic) * (1.0f - user_weights.specular_transmission * (1.0f - user_weights.diffuse_transmission)) *
					   user_weights.specular * outside_object;
	weights.diffuse =
		(1.0f - user_weights.metallic) * (1.0f - user_weights.specular_transmission) * (1.0f - user_weights.diffuse_transmission) * outside_object;

	return weights;
}

#endif // #ifndef HOST_DEVICE_COMMON_PRINCIPLED_LOBE_WEIGHTS_H
