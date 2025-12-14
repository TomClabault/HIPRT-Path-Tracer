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
	void* GGX_conductor_ltc_params = nullptr;
	void* GGX_conductor_ltc_amplitude_data = nullptr;
};

#endif
