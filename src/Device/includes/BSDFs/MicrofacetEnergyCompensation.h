/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_BSDF_MICROFACET_ENERGY_COMPENSATION_H
#define DEVICE_BSDF_MICROFACET_ENERGY_COMPENSATION_H

#include "Device/includes/Fresnel.h"
#include "Device/includes/Texture.h"

#include "Device/includes/SanityCheck.h"
#include "HostDeviceCommon/RenderData.h"
 // To be able to access GPUBakerConstants::GGX_DIRECTIONAL_ALBEDO_TEXTURE_SIZE && GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE
#include "Renderer/Baker/GPUBakerConstants.h"

 /**
  * References:
  * [1] [Practical multiple scattering compensation for microfacet models, Turquin, 2019]
  * [2] [Revisiting Physically Based Shading at Imageworks, Kulla & Conty, SIGGRAPH 2017]
  * [3] [Dassault Enterprise PBR 2025 Specification]
  * [4] [Google - Physically Based Rendering in Filament]
  * [5] [MaterialX codebase on Github]
  * [6] [Blender's Cycles codebase on Github]
  */

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F get_GGX_energy_compensation_conductors(const HIPRTRenderData& render_data, const ColorRGB32F& F0, float material_roughness, bool material_do_energy_compensation, const float3& local_view_direction, int current_bounce)
{
	bool max_bounce_reached = current_bounce > render_data.bsdfs_data.metal_energy_compensation_max_bounce && render_data.bsdfs_data.metal_energy_compensation_max_bounce > -1;
	bool smooth_enough = material_roughness <= render_data.bsdfs_data.energy_compensation_roughness_threshold;
	bool invalid_view_direction = local_view_direction.z < 0.0f;
	if (!material_do_energy_compensation || smooth_enough || max_bounce_reached || invalid_view_direction)
		return ColorRGB32F(1.0f);

    const void* GGX_directional_albedo_texture_pointer = nullptr;
#ifdef __KERNELCC__
    GGX_directional_albedo_texture_pointer = &render_data.bsdfs_data.GGX_conductor_directional_albedo;
#else
    GGX_directional_albedo_texture_pointer = render_data.bsdfs_data.GGX_conductor_directional_albedo;
#endif

    // Reading the precomputed directional albedo from the texture
	float2 uv = make_float2(hippt::max(0.0f, local_view_direction.z), material_roughness);

	// Flipping the Y manually (and that's why we pass 'false' in the sample call that follow)
	// because that GGX energy compensation texture is created with a clamp address mode, not wrap
	// and we have to do the Y-flipping manually when not sampling in wrap mode
	uv.y = 1.0f - uv.y;
	float Ess = sample_texture_rgb_32bits(GGX_directional_albedo_texture_pointer, 0, /* is_srgb */ false, uv, /* flip UV-Y */ false).r;

    // Computing kms, [Practical multiple scattering compensation for microfacet models, Turquin, 2019], Eq. 10
    float kms = (1.0f - Ess) / Ess;

#if PrincipledBSDFDoMetallicFresnelEnergyCompensation == KERNEL_OPTION_TRUE
    // [Practical multiple scattering compensation for microfacet models, Turquin, 2019], Eq. 15
    ColorRGB32F fresnel_compensation_term = F0;
#else
    // 1.0f F so that the fresnel compensation has no effect
    ColorRGB32F fresnel_compensation_term = ColorRGB32F(1.0f);
#endif
    // Computing the compensation term and multiplying by the single scattering non-energy conserving base GGX BRDF,
    // Eq. 9
    return ColorRGB32F(1.0f) + kms * fresnel_compensation_term;
}

/**
 * References:
 * [1] [Practical multiple scattering compensation for microfacet models, Turquin, 2019] [Main implementation]
 * [2] [Revisiting Physically Based Shading at Imageworks, Kulla & Conty, SIGGRAPH 2017]
 * [3] [Dassault Enterprise PBR 2025 Specification]
 * [4] [Google - Physically Based Rendering in Filament]
 * [5] [MaterialX codebase on Github]
 * [6] [Blender's Cycles codebase on Github]
 *
 * The energy compensation LUT for GGX Glass materials is computed by remapping cos_theta
 * with cos_theta^2.5
 *
 * However cos_theta^2.5 still results in energy gains at grazing angles so we're going to bias
 * the exponent used for fetching in the table here.
 *
 * This means that we store in the LUT during the precomputation but we're going to fetch
 * from the LUT with an exponent higher than 2.5f to try and force-remove energy gains
 *
 * The "ideal" exponent depends primarily on roughness so I've fined tuned some parameters
 * here to try and get the best white furnace tests
 *
 *
 * --------------------
 * If you're reading this code for a reference implementation, read what follows:
 * In the end, what we're doing here is to fix the unwanted energy gains that we have with
 * the base implementation as proposed in
 * [Practical multiple scattering compensation for microfacet models, Turquin, 2019].
 *
 * I don't think that these energy gains are supposed to happen, they are not mentioned
 * anywhere in the papers. And the papers use 32x32x32 tables. We use 256x16x192. And
 * we still have issues. I'm lead to believe that the issue is elsewhere in the codebase
 * but oh well... I can't find where this is coming from so we're fixing the broken code
 * instead of fixing the root of the issue which probably isn't what you should do if you're
 * reading this
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_glass_energy_compensation_get_correction_exponent(float roughness, float relative_eta)
{
    if (hippt::is_zero(roughness) || hippt::abs(1.0f - relative_eta) < 1.0e-3f)
        // No correction for these, returning the original 2.5f that is used in the LUT
        return 2.5f;

	float lower_relative_eta_bound = 1.01f;
	float lower_correction = 2.5f;
	if (relative_eta > 1.01f && relative_eta <= 1.02f)
	{
		lower_relative_eta_bound = 1.01f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = 2.5f;
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.3f, 2.4f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.4f, 2.45f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.45f, 2.4665f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.4665f, 2.52f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(2.52f, 2.55f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = 2.55f;
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(2.55f, 2.585f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(2.585f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 1.02f && relative_eta <= 1.03f)
	{
		lower_relative_eta_bound = 1.02f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = 2.5f;
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.3f, 2.4f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.4f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 2.51f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.51f, 2.54f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(2.54f, 2.565f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = hippt::lerp(2.565f, 2.57f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(2.57f, 2.59f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(2.59f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 1.03f && relative_eta <= 1.1f)
	{
		lower_relative_eta_bound = 1.03f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = 2.5f;
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.3f, 2.4f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.4f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 2.51f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.51f, 2.544f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(2.544f, 2.565f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = hippt::lerp(2.565f, 2.58f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(2.58f, 2.6f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(2.6f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 1.1f && relative_eta <= 1.2f)
	{
		lower_relative_eta_bound = 1.1f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = 2.5f;
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 2.54f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.54f, 2.575f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(2.575f, 2.61f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = hippt::lerp(2.61f, 2.63f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(2.63f, 2.6f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(2.6f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 1.2f && relative_eta <= 1.4f)
	{
		lower_relative_eta_bound = 1.2f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = hippt::lerp(2.5f, 1.8f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(1.8f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 2.55f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.55f, 2.65f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(2.65f, 2.675f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = hippt::lerp(2.675f, 2.7f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(2.7f, 2.675f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(2.675f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 1.4f && relative_eta <= 1.5f)
	{
		lower_relative_eta_bound = 1.4f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = hippt::lerp(2.5f, 1.8f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(1.8f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 2.7f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.7f, 2.875f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(2.875f, 2.925f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = hippt::lerp(2.925f, 2.95f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(2.95f, 2.8f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(2.8f, 2.55f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 1.5f && relative_eta <= 2.0f)
	{
		lower_relative_eta_bound = 1.5f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = hippt::lerp(2.5f, 1.6f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(1.6f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 2.7f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.7f, 2.95f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(2.95f, 3.1f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = 3.1f;
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(3.1f, 3.05f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(3.05f, 2.57f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 2.0f && relative_eta <= 2.4f)
	{
		lower_relative_eta_bound = 2.0f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = hippt::lerp(2.5f, 1.5f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(1.5f, 2.2f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.2f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 2.75f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(2.75f, 3.5f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(3.5f, 4.85f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = hippt::lerp(4.85f, 6.0f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(6.0f, 7.0f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(7.0f, 2.57f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta > 2.4f && relative_eta <= 3.0f)
	{
		lower_relative_eta_bound = 2.4f;

		if (roughness <= 0.0f)
			lower_correction = 2.5f;
		else if (roughness <= 0.1f)
			lower_correction = hippt::lerp(2.5f, 1.5f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			lower_correction = hippt::lerp(1.5f, 2.0f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			lower_correction = hippt::lerp(2.0f, 2.44f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			lower_correction = hippt::lerp(2.44f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			lower_correction = hippt::lerp(2.475f, 3.0f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			lower_correction = hippt::lerp(3.0f, 3.8f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			lower_correction = hippt::lerp(3.8f, 7.0f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			lower_correction = hippt::lerp(7.0f, 10.0f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			lower_correction = hippt::lerp(10.0f, 12.0f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			lower_correction = hippt::lerp(12.0f, 3.9f, (roughness - 0.9f) / 0.1f);
	}

	float higher_relative_eta_bound = 1.01f;
	float higher_correction = 2.5f;
	if (relative_eta <= 1.01f)
	{
		higher_relative_eta_bound = 1.01f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = 2.5f;
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.3f, 2.4f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.4f, 2.45f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.45f, 2.4665f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.4665f, 2.52f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(2.52f, 2.55f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = 2.55f;
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(2.55f, 2.585f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(2.585f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 1.02f)
	{
		higher_relative_eta_bound = 1.02f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = 2.5f;
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.3f, 2.4f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.4f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.51f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.51f, 2.54f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(2.54f, 2.565f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(2.565f, 2.57f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(2.57f, 2.59f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(2.59f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 1.03f)
	{
		higher_relative_eta_bound = 1.03f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = 2.5f;
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.3f, 2.4f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.4f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.51f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.51f, 2.544f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(2.544f, 2.565f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(2.565f, 2.58f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(2.58f, 2.6f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(2.6f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 1.1f)
	{
		higher_relative_eta_bound = 1.1f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = 2.5f;
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(2.5f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.54f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.54f, 2.575f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(2.575f, 2.61f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(2.61f, 2.63f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(2.63f, 2.6f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(2.6f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 1.2f)
	{
		higher_relative_eta_bound = 1.2f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = hippt::lerp(2.5f, 1.8f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(1.8f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.55f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.55f, 2.65f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(2.65f, 2.675f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(2.675f, 2.7f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(2.7f, 2.675f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(2.675f, 2.5f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 1.4f)
	{
		higher_relative_eta_bound = 1.4f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = hippt::lerp(2.5f, 1.8f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(1.8f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.7f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.7f, 2.875f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(2.875f, 2.925f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(2.925f, 2.95f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(2.95f, 2.8f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(2.8f, 2.55f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 1.5f)
	{
		higher_relative_eta_bound = 1.5f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = hippt::lerp(2.5f, 1.6f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(1.6f, 2.3f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.3f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.7f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.7f, 2.95f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(2.95f, 3.1f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = 3.1f;
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(3.1f, 3.05f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(3.05f, 2.57f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 2.0f)
	{
		higher_relative_eta_bound = 2.0f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = hippt::lerp(2.5f, 1.5f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(1.5f, 2.2f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.2f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.75f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.75f, 3.5f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(3.5f, 4.85f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(4.85f, 6.0f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(6.0f, 7.0f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(7.0f, 2.57f, (roughness - 0.9f) / 0.1f);
	}
	else if (relative_eta <= 2.4f)
	{
		higher_relative_eta_bound = 2.4f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = hippt::lerp(2.5f, 1.5f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(1.5f, 2.0f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(2.0f, 2.44f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.44f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 3.0f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(3.0f, 3.8f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(3.8f, 7.0f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(7.0f, 10.0f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(10.0f, 12.0f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(12.0f, 3.9f, (roughness - 0.9f) / 0.1f);
	}
	else
	{
		higher_relative_eta_bound = 3.0f;

		if (roughness <= 0.0f)
			higher_correction = 2.5f;
		else if (roughness <= 0.1f)
			higher_correction = hippt::lerp(2.5f, 1.5f, (roughness - 0.0f) / 0.1f);
		else if (roughness <= 0.2f)
			higher_correction = hippt::lerp(1.5f, 1.7f, (roughness - 0.1f) / 0.1f);
		else if (roughness <= 0.3f)
			higher_correction = hippt::lerp(1.7f, 2.38f, (roughness - 0.2f) / 0.1f);
		else if (roughness <= 0.4f)
			higher_correction = hippt::lerp(2.38f, 2.475f, (roughness - 0.3f) / 0.1f);
		else if (roughness <= 0.5f)
			higher_correction = hippt::lerp(2.475f, 2.9f, (roughness - 0.4f) / 0.1f);
		else if (roughness <= 0.6f)
			higher_correction = hippt::lerp(2.9f, 3.8f, (roughness - 0.5f) / 0.1f);
		else if (roughness <= 0.7f)
			higher_correction = hippt::lerp(3.8f, 7.5f, (roughness - 0.6f) / 0.1f);
		else if (roughness <= 0.8f)
			higher_correction = hippt::lerp(7.5f, 12.0f, (roughness - 0.7f) / 0.1f);
		else if (roughness <= 0.9f)
			higher_correction = hippt::lerp(12.0f, 13.75f, (roughness - 0.8f) / 0.1f);
		else if (roughness <= 1.0f)
			higher_correction = hippt::lerp(13.75f, 2.5f, (roughness - 0.9f) / 0.1f);
	}

	if (higher_relative_eta_bound == lower_relative_eta_bound)
		// Arbitrarily returning the lower correction 
		return lower_correction;

	return hippt::lerp(lower_correction, higher_correction, (relative_eta - lower_relative_eta_bound) / (higher_relative_eta_bound - lower_relative_eta_bound));
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_GGX_energy_compensation_dielectrics(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, float custom_roughness, bool inside_object, float eta_t, float eta_i, float relative_eta, float NoV, int current_bounce)
{
	bool smooth_enough = custom_roughness <= render_data.bsdfs_data.energy_compensation_roughness_threshold;
	bool max_bounce_reached = current_bounce > render_data.bsdfs_data.glass_energy_compensation_max_bounce && render_data.bsdfs_data.glass_energy_compensation_max_bounce > -1;
	if (!material.do_glass_energy_compensation || smooth_enough || max_bounce_reached)
		return 1.0f;

	float compensation_term = 1.0f;

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoGlassEnergyCompensation == KERNEL_OPTION_TRUE
	// Not doing energy compensation if the thin-film is fully present
	// See the // TODO FIX THIS HORROR below
	//
	// Also not doing compensation if we already have full compensation on the material
	// because the energy compensation of the glass lobe here is then redundant
	bool bsdf_already_compensated = material.enforce_strong_energy_conservation && PrincipledBSDFEnforceStrongEnergyConservation == KERNEL_OPTION_TRUE;
	if (material.thin_film < 1.0f && !bsdf_already_compensated)
	{
		float relative_eta_for_correction = inside_object ? 1.0f / relative_eta : relative_eta;
		float exponent_correction = 2.5f;
		if (!material.thin_walled)
			exponent_correction = GGX_glass_energy_compensation_get_correction_exponent(custom_roughness, relative_eta_for_correction);

		// We're storing cos_theta_o^2.5 in the LUT so we're retrieving it with pow(1.0f / 2.5f) i.e.
		// sqrt 2.5
		//
		// We're using a "correction exponent" to forcefully get rid of energy gains at grazing angles due
		// to float precision issues: storing in the LUT with cos_theta^2.5 but fetching with pow(1.0f / 2.6f)
		// for example (instead of fetching with pow(1.0f / 2.5f)) darkens the overall appearance and helps remove
		// energy gains
		float view_direction_tex_fetch = powf(hippt::max(1.0e-3f, NoV), 1.0f / exponent_correction);

		float F0 = F0_from_eta(eta_t, eta_i);
		// sqrt(sqrt()) of F0 here because we're storing F0^4 in the LUT
		float F0_remapped = sqrt(sqrt(F0));

		float3 uvw = make_float3(view_direction_tex_fetch, custom_roughness, F0_remapped);
		if (material.thin_walled)
		{
			void* texture = render_data.bsdfs_data.GGX_thin_glass_directional_albedo;
			int3 dims = make_int3(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);

			compensation_term = sample_texture_3D_rgb_32bits(texture, dims, uvw, render_data.bsdfs_data.use_hardware_tex_interpolation).r;
		}
		else
		{
			void* texture = inside_object ? render_data.bsdfs_data.GGX_glass_directional_albedo_inverse : render_data.bsdfs_data.GGX_glass_directional_albedo;
			int3 dims = make_int3(GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GGX_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR);

			compensation_term = sample_texture_3D_rgb_32bits(texture, dims, uvw, render_data.bsdfs_data.use_hardware_tex_interpolation).r;
		}

		// TODO FIX THIS HORROR
		// This is here because directional albedo for the glass BSDF is tabulated with the standard non-colored Fresnel
		// This means that the precomputed table is incompatible with the thin-film interference fresnel
		// 
		// And as a matter of fact, using the energy compensation term (precomputed for the traditional fresnel)
		// with thin-film interference Fresnel results in noticeable energy gains at grazing angles at high roughnesses
		//
		// Blender Cycles doesn't have that issue but I don't understand yet how they avoid it.
		//
		// The quick and disgusting solution here is just to disable energy compensation as the thin-film
		// weight gets stronger. Energy compensation is fully disabled when the thin-film weight is 1.0f
		//
		// Because the error is stronger at high roughnesses than at low roughnesses, we can include the roughness
		// in the lerp such that we use less and less the energy compensation term as the roughness increases
		compensation_term = hippt::lerp(compensation_term, 1.0f, material.thin_film * custom_roughness);
	}
#endif

	return compensation_term;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_GGX_energy_compensation_dielectrics(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, bool inside_object, float eta_t, float eta_i, float relative_eta, float NoV, int current_bounce)
{
	return get_GGX_energy_compensation_dielectrics(render_data, material, material.roughness, inside_object, eta_t, eta_i, relative_eta, NoV, current_bounce);
}

#endif
