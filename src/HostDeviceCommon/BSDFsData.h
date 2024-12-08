/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_BSDFS_DATA_H
#define HOST_DEVICE_COMMON_BSDFS_DATA_H

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
	// 32x32 texture containing the precomputed parameters of the LTC
	// fitted to approximate the SSGX sheen volumetric layer.
	// See SheenLTCFittedParameters.h
	void* sheen_ltc_parameters_texture = nullptr;

	// 2D texture for the precomputed directional albedo
	// for the GGX BRDFs used in the principled BSDF for energy conservation
	// of conductors
	void* GGX_Ess = nullptr;

	// 3D texture for the precomputed directional albedo of the base layer
	// of the principled BSDF (specular GGX layer + diffuse below)
	void* glossy_dielectric_Ess = nullptr;

	// 3D texture (cos_theta_o, roughness, relative_eta) for the precomputed
	// directional albedo used for energy conservation of glass objects when
	// entering a medium
	void* GGX_Ess_glass = nullptr;
	// Table when leaving a medium
	void* GGX_Ess_glass_inverse = nullptr;

	// Whether or not to use the texture unit's hardware texel interpolation
	// when fetching the LUTs. It's faster but less precise.
	bool use_hardware_tex_interpolation = false;

	GGXMaskingShadowingFlavor GGX_masking_shadowing = GGXMaskingShadowingFlavor::HeightCorrelated;

	// For on-the-fly monte carlo integration of the directional albedo of the clearcoat layer 
	// (so basically the whole BSDF because the clearcoat is the very top layer)
	// 
	// How many samples to evaluate the integral with. This is done for each pixel for each material
	// that has clearcoat energy compensation so this may be very expensive.
	//
	// More samples = better energy compensation. Too few samples leads to energy-gains/losses
	int clearcoat_energy_compensation_samples = 12;
};

#endif
