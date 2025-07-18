/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_BAKER_CONSTANTS_H
#define GPU_BAKER_CONSTANTS_H

#ifndef __KERNELCC__
#include <string>
#endif

#include "HostDeviceCommon/BSDFsData.h"

struct GPUBakerConstants
{
	static const int GGX_CONDUCTOR_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O = 128;
	static const int GGX_CONDUCTOR_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS = 128;

	static const int GGX_FRESNEL_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O = 256;
	static const int GGX_FRESNEL_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS = 256;
	static const int GGX_FRESNEL_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR = 256;

	static const int GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O = 256;
	static const int GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS = 16;
	static const int GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR = 128;

	static const int GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O = 32;
	static const int GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS = 32;
	static const int GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR = 96;

	static const int GLOSSY_DIELECTRIC_TEXTURE_SIZE_COS_THETA_O = 128;
	static const int GLOSSY_DIELECTRIC_TEXTURE_SIZE_ROUGHNESS = 64;
	static const int GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR = 128;

	// Arbitrary number to limit how much computation we do per bake kernel launch.
	// This is to avoid driver timeouts
	static const int COMPUTE_ELEMENT_PER_BAKE_KERNEL_LAUNCH = 100000000;

#ifndef __KERNELCC__
	// Not using these on the GPU since they are std::string types: unavailable on the GPU
	// and besides, we don't these paths on the GPU, only the texture sizes
	static std::string get_GGX_conductor_directional_albedo_texture_filename(GGXMaskingShadowingFlavor masking_shadowing_term, 
		int texture_size_cos_theta = GPUBakerConstants::GGX_CONDUCTOR_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O,
		int texture_size_roughness = GPUBakerConstants::GGX_CONDUCTOR_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS)
	{
		std::string flavor_string = masking_shadowing_term == GGXMaskingShadowingFlavor::HeightCorrelated ? "Correlated_" : "Uncorrelated_";

		return "GGX_Conductor_" + flavor_string + std::to_string(texture_size_cos_theta) + "x" + std::to_string(texture_size_roughness) + ".hdr";
	}

	static std::string get_GGX_fresnel_directional_albedo_texture_filename(GGXMaskingShadowingFlavor masking_shadowing_term, 
		int texture_size_cos_theta = GPUBakerConstants::GGX_FRESNEL_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O,
		int texture_size_roughness = GPUBakerConstants::GGX_FRESNEL_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS,
		int texture_size_ior = GPUBakerConstants::GGX_FRESNEL_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR)
	{
		std::string flavor_string = masking_shadowing_term == GGXMaskingShadowingFlavor::HeightCorrelated ? "Correlated_" : "Uncorrelated_";

		return "GGX_Fresnel_" + flavor_string + std::to_string(texture_size_cos_theta) + "x" + std::to_string(texture_size_roughness) + "x" + std::to_string(texture_size_ior) + ".hdr";
	}

	static std::string get_glossy_dielectric_directional_albedo_texture_filename(GGXMaskingShadowingFlavor masking_shadowing_term, 
		int texture_size_cos_theta = GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_COS_THETA_O,
		int texture_size_roughness = GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_ROUGHNESS,
		int texture_size_ior = GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR)
	{
		std::string flavor_string = masking_shadowing_term == GGXMaskingShadowingFlavor::HeightCorrelated ? "Correlated_" : "Uncorrelated_";

		return "Glossy_Ess_" + flavor_string + std::to_string(texture_size_cos_theta) + "x" + std::to_string(texture_size_roughness) + "x" + std::to_string(texture_size_ior) + ".hdr";
	}

	static std::string get_GGX_glass_directional_albedo_texture_filename(GGXMaskingShadowingFlavor masking_shadowing_term, 
		int texture_size_cos_theta = GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O,
		int texture_size_roughness = GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS,
		int texture_size_ior = GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR)
	{
		std::string flavor_string = masking_shadowing_term == GGXMaskingShadowingFlavor::HeightCorrelated ? "Correlated_" : "Uncorrelated_";

		return "GGX_Glass_Ess_" + flavor_string + std::to_string(texture_size_cos_theta) + "x" + std::to_string(texture_size_roughness) + "x" + std::to_string(texture_size_ior) + ".hdr";
	}

	static std::string get_GGX_thin_glass_directional_albedo_texture_filename(GGXMaskingShadowingFlavor masking_shadowing_term, 
		int texture_size_cos_theta = GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O,
		int texture_size_roughness = GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS,
		int texture_size_ior = GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR)
	{
		std::string flavor_string = masking_shadowing_term == GGXMaskingShadowingFlavor::HeightCorrelated ? "Correlated_" : "Uncorrelated_";

		return "GGX_Thin_Glass_Ess_" + flavor_string + std::to_string(texture_size_cos_theta) + "x" + std::to_string(texture_size_roughness) + "x" + std::to_string(texture_size_ior) + ".hdr";
	}

	static std::string get_GGX_glass_directional_albedo_inv_texture_filename(GGXMaskingShadowingFlavor masking_shadowing_term, 
		int texture_size_cos_theta = GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O, 
		int texture_size_roughness = GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS, 
		int texture_size_ior = GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR)
	{
		std::string flavor_string = masking_shadowing_term == GGXMaskingShadowingFlavor::HeightCorrelated ? "Correlated_" : "Uncorrelated_";
		
		return "inv_GGX_Glass_Ess_" + flavor_string + std::to_string(texture_size_cos_theta) + "x" + std::to_string(texture_size_roughness) + "x" + std::to_string(texture_size_ior) + ".hdr";
	}
#endif
};

#endif
