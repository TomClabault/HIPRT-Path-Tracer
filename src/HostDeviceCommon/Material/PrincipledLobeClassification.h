/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_PRINCIPLED_LOBE_CLASSIFICATION_H
#define HOST_DEVICE_COMMON_PRINCIPLED_LOBE_CLASSIFICATION_H

#include "HostDeviceCommon/Material/PrincipledLobeWeights.h"

struct PrincipledMaterialClassificationInputs
{
	PrincipledLobeUserWeights lobe_user_weights;
	float thin_film_strength				= 0.0f;
	float dispersion_scale					= 0.0f;
	bool thin_walled						= false;
	bool enforce_strong_energy_conservation = false;
};

HIPRT_DEVICE inline KernelMaterialSpecialization classify_principled_material(const PrincipledMaterialClassificationInputs& inputs, bool outside_object)
{
	const PrincipledLobeUserWeights& user_weights = inputs.lobe_user_weights;

	if (user_weights.coat != 0.0f || user_weights.sheen != 0.0f || user_weights.diffuse_transmission != 0.0f || inputs.thin_film_strength != 0.0f ||
		inputs.dispersion_scale != 0.0f || user_weights.second_roughness_weight != 0.0f || user_weights.retro_reflection != 0.0f || inputs.thin_walled ||
		inputs.enforce_strong_energy_conservation)
		return KernelMaterialSpecializationAll;

	PrincipledLobeWeights weights = compute_principled_lobe_weights(user_weights, outside_object);
	unsigned int support_mask	  = get_principled_lobe_support_mask(weights);

	if (outside_object && user_weights.metallic == 0.0f && user_weights.specular_transmission == 0.0f && user_weights.specular == 0.0f &&
		support_mask == PrincipledLobeSupportDiffuse)
		return KernelMaterialSpecializationDiffuse;

	if (user_weights.metallic == 0.0f && user_weights.specular_transmission == 1.0f && support_mask == PrincipledLobeSupportGlass)
		return KernelMaterialSpecializationGlass;

	if (outside_object && user_weights.metallic == 1.0f && support_mask == PrincipledLobeSupportMetallicFirst)
		return KernelMaterialSpecializationSingleMetallic;

	unsigned int specular_diffuse_support = PrincipledLobeSupportSpecular | PrincipledLobeSupportDiffuse;
	if (outside_object && user_weights.metallic == 0.0f && user_weights.specular_transmission == 0.0f && user_weights.specular > 0.0f &&
		support_mask == specular_diffuse_support)
		return KernelMaterialSpecializationSpecularDiffuse;

	return KernelMaterialSpecializationAll;
}

#endif // #ifndef HOST_DEVICE_COMMON_PRINCIPLED_LOBE_CLASSIFICATION_H
