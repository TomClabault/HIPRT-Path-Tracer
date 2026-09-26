/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_PRINCIPLED_LOBE_SUPPORT_H
#define HOST_DEVICE_COMMON_MATERIAL_PRINCIPLED_LOBE_SUPPORT_H

struct PrincipledLobeWeights
{
	float coat				   = 0.0f;
	float sheen				   = 0.0f;
	float metallic_first	   = 0.0f;
	float metallic_second	   = 0.0f;
	float retro_reflection	   = 0.0f;
	float specular			   = 0.0f;
	float diffuse			   = 0.0f;
	float glass				   = 0.0f;
	float diffuse_transmission = 0.0f;
};

enum PrincipledLobeSupportBits : unsigned int
{
	PrincipledLobeSupportCoat				 = 1u << 0,
	PrincipledLobeSupportSheen				 = 1u << 1,
	PrincipledLobeSupportMetallicFirst		 = 1u << 2,
	PrincipledLobeSupportMetallicSecond		 = 1u << 3,
	PrincipledLobeSupportRetroReflection	 = 1u << 4,
	PrincipledLobeSupportSpecular			 = 1u << 5,
	PrincipledLobeSupportDiffuse			 = 1u << 6,
	PrincipledLobeSupportGlass				 = 1u << 7,
	PrincipledLobeSupportDiffuseTransmission = 1u << 8
};

HIPRT_DEVICE inline unsigned int get_principled_lobe_support_mask(const PrincipledLobeWeights& weights)
{
	unsigned int support_mask = 0u;

	if (weights.coat > 0.0f)
		support_mask |= PrincipledLobeSupportCoat;
	if (weights.sheen > 0.0f)
		support_mask |= PrincipledLobeSupportSheen;
	if (weights.metallic_first > 0.0f)
		support_mask |= PrincipledLobeSupportMetallicFirst;
	if (weights.metallic_second > 0.0f)
		support_mask |= PrincipledLobeSupportMetallicSecond;
	if (weights.retro_reflection > 0.0f)
		support_mask |= PrincipledLobeSupportRetroReflection;
	if (weights.specular > 0.0f)
		support_mask |= PrincipledLobeSupportSpecular;
	if (weights.diffuse > 0.0f)
		support_mask |= PrincipledLobeSupportDiffuse;
	if (weights.glass > 0.0f)
		support_mask |= PrincipledLobeSupportGlass;
	if (weights.diffuse_transmission > 0.0f)
		support_mask |= PrincipledLobeSupportDiffuseTransmission;

	return support_mask;
}

#endif // #ifndef HOST_DEVICE_COMMON_MATERIAL_PRINCIPLED_LOBE_SUPPORT_H
