/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_BSDFS_DATA_H
#define HOST_DEVICE_COMMON_BSDFS_DATA_H

#include "HostDeviceCommon/LTCsData.h"
#include "HostDeviceCommon/MicrofacetRegularizationSettings.h"

/**
 * What masking-shadowing term to use with the GGX NDF.
 *
 * 'HeightCorrelated' is a little be more precise and
 * corect than 'HeightUncorrelated' so it should basically
 * always be preferred.
 *
 * This is basically only for experimentation purposes
 */
enum GGXMaskingShadowingFlavor
{
	HeightCorrelated,
	HeightUncorrelated
};

struct BRDFsData
{
	bool white_furnace_mode					   = false;
	bool white_furnace_mode_turn_off_emissives = true;

	LTCsData ltcs_data;

	// 2D texture for the precomputed directional albedo
	// for the GGX BRDFs used in the principled BSDF for energy compensation
	// of conductors
	void* GGX_conductor_directional_albedo = nullptr;

	// 3D texture for the precomputed directional albedo of the base layer
	// of the principled BSDF (specular GGX layer + diffuse below)
	void* glossy_dielectric_directional_albedo = nullptr;

	// 3D texture (cos_theta_o, roughness, relative_eta) for the precomputed
	// directional albedo used for energy compensation of glass objects when
	// entering a medium
	void* GGX_glass_directional_albedo = nullptr;
	// Table when leaving a medium
	void* GGX_glass_inverse_directional_albedo = nullptr;

	// Table for energy compesantion of thin walled glass
	// Fetching into this table should use the base roughness
	// of the material i.e. **not** the remapped thin-walled roughness
	void* GGX_thin_glass_directional_albedo = nullptr;

	// Whether or not to use the texture unit's hardware texel interpolation
	// when fetching the LUTs. It's faster but less precise.
	bool use_hardware_tex_interpolation = false;

	GGXMaskingShadowingFlavor GGX_masking_shadowing = GGXMaskingShadowingFlavor::HeightCorrelated;

	float energy_compensation_roughness_threshold = 0.15f;

	// If the roughness of the metallic lobe of the Principled BSDF is higher or equal to this threshold, the metallic lobe will be sampled using
	// cosine-weighted hemisphere sampling instead of GGX VNDF sampling. This massively improves variance on very rough conductor lobes (> 0.7 roughness)
	// and also saves on performance because cosine weighted hemisphere sampling is much faster to compute than GGX VNDF sampling.
	//
	// This option is only useful if PrincipledBSDFMetallicSampleCosineWeighted is KERNEL_OPTION_TRUE
	float metallic_sample_cosine_weighted_roughness_threshold = 0.75f;

	// If using the invariance Cui et al. method GGX for multiple scattering in the microsurface energy compensation, this is after how many bounces in the
	// microsurface we start applying russian roulette to potentially end the random walk. Higher values for this parameter means that more bounces will be
	// computed without russian roulette --> lower variance but higher cost
	//
	// A value of 1 means that russian roulette will be applied starting at the first bounce in the microsurface, which means that russian roulette will always
	// be applied.
	int multiple_scattering_cui_2023_min_bounce_russian_roulette = 3;

	MicrofacetRegularizationSettings microfacet_regularization;
};

#endif
