/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_BSDFS_DATA_H
#define HOST_DEVICE_COMMON_BSDFS_DATA_H

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
	bool white_furnace_mode = false;
	bool white_furnace_mode_turn_off_emissives = true;

	// 32x32 texture containing the precomputed parameters of the LTC
	// fitted to approximate the SSGX sheen volumetric layer.
	// See SheenLTCFittedParameters.h
	void* sheen_ltc_parameters_texture = nullptr;

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
	void* GGX_glass_directional_albedo_inverse = nullptr;

	// Table for energy compesantion of thin walled glass
	// Fetching into this table should use the base roughness
	// of the material i.e. **not** the remapped thin-walled roughness
	void* GGX_thin_glass_directional_albedo = nullptr;

	// Whether or not to use the texture unit's hardware texel interpolation
	// when fetching the LUTs. It's faster but less precise.
	bool use_hardware_tex_interpolation = false;

	GGXMaskingShadowingFlavor GGX_masking_shadowing = GGXMaskingShadowingFlavor::HeightUncorrelated;

	float energy_compensation_roughness_threshold = 0.0f;

	// After hom many bounces to stop doing energy compensation to save performance?
	// 
	// For example, 0 means that energy compensation will only be done on the first hit and
	// not later
	//
	// -1 to disable
	int glass_energy_compensation_max_bounce = -1;
	int metal_energy_compensation_max_bounce = -1;
	int clearcoat_energy_compensation_max_bounce = -1;
	int glossy_base_energy_compensation_max_bounce = -1;

	MicrofacetRegularizationSettings microfacet_regularization;
};

#endif
