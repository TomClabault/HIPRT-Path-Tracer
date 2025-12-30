/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTC_READ_PARAMS_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_LTCS_LTC_READ_PARAMS_H

#include "Device/includes/LightSampling/LTCs/LTCLobe.h"
#include "Device/includes/Texture.h"

 /**
 * This function returns the LTC parameters for the given cos_theta_v and material.
 * The coefficients M00, M02, M11 and M20 are stored in the R, G, B and A
 * components of the returned ColorRGBA32F respectively.
 *
 * This function takes a random number generator and outputs a PDF because
 * the material be consisting of multiple lobes, in which case
 * this function will return the LTC matrix of one of the lobes randomly selected.
 *
 * The PDF should be taken into account in the sampling routine using these LTC parameters.
 * The PDF parameter can be passed nullptr if not needed
 */
HIPRT_DEVICE ColorRGBA32F read_ltc_params(void* ltcs_data_param_pointer, float cos_theta_v,
	const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	float roughness = 1.0f;

	//switch (ltc_lobe)
	//{
	//case DIFFUSE_LOBE:
	//	return ColorRGBA32F(1.0f, 0.0f, 1.0f, 0.0f); // Identity matrix for the diffuse lobe
	//	break;

	//case METALLIC_LOBE:
	//case SPECULAR_LOBE:
	//	roughness = material.roughness;
	//	break;

	//case COAT_LOBE:
	//	roughness = material.coat_roughness;
	//	break;
	//}

	const void* texture_ptr = nullptr;
#ifdef __KERNELCC__
	texture_ptr = &ltcs_data_param_pointer;
#else
	texture_ptr = ltcs_data_param_pointer;
#endif

#ifdef __KERNELCC__
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, roughness * roughness);
#else
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, 1.0f - roughness * roughness);
#endif

	// R, G, B and A components represent respectively:
	// m00, m02, m11 and m20 of the LTC matrix
	//
	// The matrix is thus:
	// 	{ m00  0   m02 }
	//  { 0    m11  0  }
	//  { m20  0    1  }
	ColorRGBA32F ltc_params = sample_texture_rgba_32bits(texture_ptr, 0, /* is_srgb */ false, uv, /* flip UV-Y */ false);

	return ltc_params;
}

HIPRT_DEVICE float read_ltc_amplitude(void* ltcs_data_amplitude_texture, float cos_theta_v,
	const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	float roughness = 1.0f;
	//switch (ltc_lobe)
	//{
	//case DIFFUSE_LOBE:
	//	return 1.0f; // Amplitude is 1 for the diffuse lobe
	//	break;

	//case METALLIC_LOBE:
	//case SPECULAR_LOBE:
	//	roughness = material.roughness;
	//	break;

	//case COAT_LOBE:
	//	roughness = material.coat_roughness;
	//	break;
	//}

	const void* texture_ptr = nullptr;

#ifdef __KERNELCC__
	texture_ptr = &ltcs_data_amplitude_texture;
#else
	texture_ptr = ltcs_data_amplitude_texture;
#endif

#ifdef __KERNELCC__
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, roughness * roughness);
#else
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, 1.0f - roughness * roughness);
#endif

	return sample_texture_rgba_32bits(texture_ptr, 0, /* is_srgb */ false, uv, /* flip UV-Y */ false).r;
}

HIPRT_DEVICE float read_ltc_fresnel(void* ltcs_data_fresnel_texture, float cos_theta_v,
	const DeviceUnpackedEffectiveMaterial& material, LTCLobe ltc_lobe)
{
	float roughness = 1.0f;
	//switch (ltc_lobe)
	//{
	//case DIFFUSE_LOBE:
	//	roughness = 1.0f; // Amplitude is 1 for the diffuse lobe
	//	break;

	//case METALLIC_LOBE:
	//case SPECULAR_LOBE:
	//	roughness = material.roughness;
	//	break;

	//case COAT_LOBE:
	//	roughness = material.coat_roughness;
	//	break;
	//}

	const void* texture_ptr = nullptr;

#ifdef __KERNELCC__
	texture_ptr = &ltcs_data_fresnel_texture;
#else
	texture_ptr = ltcs_data_fresnel_texture;
#endif

#ifdef __KERNELCC__
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, roughness * roughness);
#else
	float2 uv = make_float2(acos(cos_theta_v) / hippt::M_PI_TWO, 1.0f - roughness * roughness);
#endif

	return sample_texture_rgba_32bits(texture_ptr, 0, /* is_srgb */ false, uv, /* flip UV-Y */ false).r;
}

#endif
