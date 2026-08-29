/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_UTILS_H
#define HOST_DEVICE_COMMON_MATERIAL_UTILS_H

#include "HostDeviceCommon/KernelOptions/KernelOptions.h"
#include "HostDeviceCommon/Material/MaterialConstants.h"

struct MaterialUtils
{
	HIPRT_HOST_DEVICE static void get_oren_nayar_AB(float sigma, float& out_oren_A, float& out_oren_B)
	{
		float sigma2 = sigma * sigma;
		out_oren_A	 = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
		out_oren_B	 = 0.45f * sigma2 / (sigma2 + 0.09f);
	}

	HIPRT_HOST_DEVICE static void get_alphas(float roughness, float anisotropy, float& out_alpha_x, float& out_alpha_y)
	{
		float aspect = hippt::sqrt(1.0f - 0.9f * anisotropy);
		out_alpha_x	 = roughness * roughness / aspect;
		out_alpha_y	 = roughness * roughness * aspect;

		// Open PBR 2025
		/*out_alpha_x = roughness * roughness * hippt::sqrt(2.0f / (1.0f + hippt::square(1.0f - anisotropy)));
		out_alpha_y = (1.0f - anisotropy) * out_alpha_x;*/

		out_alpha_x = hippt::clamp(0.0f, 1.0f, out_alpha_x);
		out_alpha_y = hippt::clamp(0.0f, 1.0f, out_alpha_y);
	}

	HIPRT_HOST_DEVICE static float get_thin_walled_roughness(bool thin_walled, float base_roughness, float relative_eta)
	{
		if (!thin_walled)
			return base_roughness;

		/*
		 * Roughness remapping so that a thin walled interface matches better a
		 * properly modeled double interface model. Said otherwise: roughness remapping
		 * so that the thin walled approximation matches the non thin walled physically correct equivalent
		 *
		 * Reference:
		 * [Revisiting Physically Based Shading at Imageworks, Christopher Kulla & Alejandro Conty, 2017]
		 *
		 * https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
		 */
		float remapped = base_roughness * sqrt(3.7f * (relative_eta - 1.0f) * hippt::square(relative_eta - 0.5f) / hippt::pow_3(relative_eta));

		// Remapped roughness starts going above 1.0f starting at relative eta around 1.9f
		// and ends up at 1.39f at relative eta 3.5f
		//
		// Because we don't expect the user to input higher IOR values than that,
		// we remap that remapped roughness from [0.0f, 1.39f] to [0.0f, 1.0f]
		// and if the user inputs higher IOR values than 3.5f, we clamp to 1.0f roughness
		// anyways
		return hippt::clamp(0.0f, 1.0f, remapped / 1.39f);
	}

	HIPRT_HOST_DEVICE static bool is_perfectly_smooth(float roughness, float roughness_threshold = MaterialConstants::PERFECTLY_SMOOTH_ROUGHNESS_THRESHOLD)
	{
		return roughness <= roughness_threshold;
	}

	/**
	 * Whether or not it makes sense to even try light sampling with NEE on that material
	 *
	 * Perfectly smooth materials for example cannot do light sampling because no given light
	 * direction is going to align with the delta distribution peak of the BRDF so we can save
	 * some performance by not even attempting light sampling in the first place
	 */
	HIPRT_HOST_DEVICE static bool can_do_light_sampling(float material_roughness,
														float material_metallic,
														float material_specular_transmission,
														float material_coat,
														float material_coat_roughness,
														float material_second_roughness,
														float material_second_roughness_weight,
														float roughness_threshold)
	{
#if DirectLightSamplingDeltaDistributionOptimization == KERNEL_OPTION_FALSE
		return true;
#elif PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_TRUE // #if DirectLightSamplingDeltaDistributionOptimization == KERNEL_OPTION_FALSE
		// If we have BSDF regularization, everything can do light sampling now
		return true;
#endif // #if DirectLightSamplingDeltaDistributionOptimization == KERNEL_OPTION_FALSE

#if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR
		// We can always do light sampling on these BSDFs
		return true;
#endif // #if BSDFOverride == BSDF_LAMBERTIAN || BSDFOverride == BSDF_OREN_NAYAR

		bool smooth_base_layer = MaterialUtils::is_perfectly_smooth(material_roughness, roughness_threshold) &&
								 (material_metallic == 1.0f || material_specular_transmission == 1.0f);
		bool smooth_coat = material_coat == 0.0f || (material_coat > 0.0f && MaterialUtils::is_perfectly_smooth(material_coat_roughness, roughness_threshold));
		bool second_roughness_smooth =
								MaterialUtils::is_perfectly_smooth(material_second_roughness, roughness_threshold) || material_second_roughness_weight == 0.0f;
		if (smooth_base_layer && smooth_coat && second_roughness_smooth)
			// Everything is smooth, cannot do light sampling
			return false;

		return true;
	}

	HIPRT_DEVICE static bool use_base_color_texture(unsigned short int base_color_texture_index)
	{
		return base_color_texture_index == MaterialConstants::NO_TEXTURE ||
			   (UseMaterialTextures == KERNEL_OPTION_FALSE && UseMaterialBaseColorTextureOverride == KERNEL_OPTION_FALSE);
	}

	HIPRT_DEVICE static bool use_roughness_texture(unsigned short int roughness_texture_index, unsigned short int roughness_metallic_texture_index)
	{
		return UseMaterialTextures == KERNEL_OPTION_FALSE ||
			   (roughness_texture_index == MaterialConstants::NO_TEXTURE && roughness_metallic_texture_index == MaterialConstants::NO_TEXTURE);
	}

	HIPRT_DEVICE static bool use_metallic_texture(unsigned short int metallic_texture_index, unsigned short int roughness_metallic_texture_index)
	{
		return UseMaterialTextures == KERNEL_OPTION_FALSE ||
			   (metallic_texture_index == MaterialConstants::NO_TEXTURE && roughness_metallic_texture_index == MaterialConstants::NO_TEXTURE);
	}

	HIPRT_DEVICE static bool use_anisotropy_texture(unsigned short int anisotropy_texture_index)
	{
		return anisotropy_texture_index == MaterialConstants::NO_TEXTURE || UseMaterialTextures == KERNEL_OPTION_FALSE;
	}

	HIPRT_DEVICE static bool use_specular_texture(unsigned short int specular_texture_index)
	{
		return specular_texture_index == MaterialConstants::NO_TEXTURE || UseMaterialTextures == KERNEL_OPTION_FALSE;
	}

	HIPRT_DEVICE static bool use_coat_texture(unsigned short int coat_texture_index)
	{
		return coat_texture_index == MaterialConstants::NO_TEXTURE || UseMaterialTextures == KERNEL_OPTION_FALSE;
	}

	HIPRT_DEVICE static bool use_sheen_texture(unsigned short int sheen_texture_index)
	{
		return sheen_texture_index == MaterialConstants::NO_TEXTURE || UseMaterialTextures == KERNEL_OPTION_FALSE;
	}

	HIPRT_DEVICE static bool use_specular_transmission_texture(unsigned short int specular_transmission_texture_index)
	{
		return specular_transmission_texture_index == MaterialConstants::NO_TEXTURE || UseMaterialTextures == KERNEL_OPTION_FALSE;
	}
};

#endif // #ifndef HOST_DEVICE_COMMON_MATERIAL_UTILS_H
