/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LTCS_DATA_H
#define HOST_DEVICE_COMMON_LTCS_DATA_H

struct LTCsData
{
	// 32x32 texture containing the precomputed parameters of the LTC
	// fitted to approximate the SSGX sheen volumetric layer.
	// See SheenLTCFittedParameters.h
	void* sheen_zeltner_texture_ltc_params = nullptr;

	// 64x64 texture containing the precomputed parameters of the LTC
	// fitted to approximate the GGX specular + Lambertian diffuse BRDF
	void* GGX_conductor_ltc_params		   = nullptr;
	void* GGX_conductor_ltc_amplitude_data = nullptr;
	// This buffer contains a LUT (theta_NoV, roughness * roughness) for fD
	// as presented in [LTC Fresnel Approximation, Stephen Hill, SIGGRAPH 2016]
	void* GGX_conductor_ltc_fresnel_data = nullptr;

	// If the material lobe has a roughness below this cutoff, then this lobe
	// won't be included in the LTC sampling.
	//
	// For example, for a diffuse / specular material with a specular roughness of 1.0f
	// and a roughness cutoff of 0.5f, the specular lobe will never be sampled by LTCs, only
	// the diffuse lobe. This is to help with LTC sampling overhead on lobes that wouldn't really
	// benefit from it
	float specular_ltc_maximum_roughness = 0.7f;
};

#endif
