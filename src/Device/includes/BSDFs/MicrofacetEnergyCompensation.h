/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_BSDF_MICROFACET_ENERGY_COMPENSATION_H
#define DEVICE_BSDF_MICROFACET_ENERGY_COMPENSATION_H

#include "HostDeviceCommon/RenderData.h"
 // To be able to access GPUBakerConstants::GGX_ESS_TEXTURE_SIZE && GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE
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
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F get_GGX_energy_compensation_conductors(const HIPRTRenderData& render_data, const ColorRGB32F& F0, float material_roughness, const float3& local_view_direction)
{
    const void* GGX_Ess_texture_pointer = nullptr;
#ifdef __KERNELCC__
    GGX_Ess_texture_pointer = &render_data.bsdfs_data.GGX_Ess;
#else
    GGX_Ess_texture_pointer = render_data.bsdfs_data.GGX_Ess;
#endif

    // Reading the precomputed directional albedo from the texture
    int2 dims = make_int2(GPUBakerConstants::GGX_ESS_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GGX_ESS_TEXTURE_SIZE_ROUGHNESS);
    float Ess = sample_texture_rgb_32bits(GGX_Ess_texture_pointer, 0, dims, false, make_float2(hippt::max(0.0f, local_view_direction.z), material_roughness)).r;

    // Computing kms, [Practical multiple scattering compensation for microfacet models, Turquin, 2019], Eq. 10
    float kms = (1.0f - Ess) / Ess;

#if PrincipledBSDFGGXUseMultipleScatteringDoFresnel == KERNEL_OPTION_TRUE
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
 * The energy conservation LUT for GGX Glass materials is computed by remapping cos_theta
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
HIPRT_HOST_DEVICE HIPRT_INLINE float GGX_glass_energy_conservation_get_correction_exponent(const HIPRTRenderData& render_data, float roughness, float relative_eta)
{
    float exponent_correction = 2.5f;
    if (hippt::is_zero(roughness) || hippt::abs(1.0f - relative_eta) < 1.0e-3f)
        // No correction for these
        return exponent_correction;

    /*if (relative_eta <= 1.01f)
        exponent_correction = 2.509375f;
    else if (relative_eta <= 1.02f)
        exponent_correction = 2.53f;
    else if (relative_eta <= 1.03f)
        exponent_correction = 2.575f;
    else if (relative_eta <= 1.1f)
        exponent_correction = 2.68f;
    else if (relative_eta <= 1.2f)
        exponent_correction = 2.68f;
    else if (relative_eta <= 1.4f)
        exponent_correction = 2.68f;
    else if (relative_eta <= 1.5f)
        exponent_correction = 2.68f;
    else if (relative_eta <= 2.0f)
        exponent_correction = 2.68f;
    else if (relative_eta <= 2.4f)
        exponent_correction = 2.68f;
    else if (relative_eta <= 3.0f)
        exponent_correction = 2.68f;
    else
        exponent_correction = 2.8f;

    return exponent_correction;*/

    if (roughness > 0.0f && roughness <= 0.5f)
    {
        if (relative_eta <= 1.01f)
            exponent_correction = 2.509375f;
        else if (relative_eta <= 1.03f)
            exponent_correction = 2.53f;
        else if (relative_eta <= 1.1f)
            exponent_correction = 2.575f;
        else if (relative_eta <= 1.3f)
            exponent_correction = 2.68f;
        else if (relative_eta <= 1.5f)
            exponent_correction = 2.75f;
        else
            exponent_correction = 2.8f;
    }
    else if (roughness >= 0.5f)
    {
        float correction_rough_0_5;
        float correction_rough_1_0;

        if (relative_eta <= 1.01f)
            correction_rough_0_5 = render_data.bsdfs_data.correction_1_01;
        else if (relative_eta <= 1.03f)
            correction_rough_0_5 = render_data.bsdfs_data.correction_1_03;
        else if (relative_eta <= 1.1f)
            correction_rough_0_5 = render_data.bsdfs_data.correction_1_1;
        else if (relative_eta <= 1.3f)
            correction_rough_0_5 = render_data.bsdfs_data.correction_1_3;
        else if (relative_eta <= 1.5f)
            correction_rough_0_5 = render_data.bsdfs_data.correction_1_5;
        else if (relative_eta <= 2.0f)
            correction_rough_0_5 = render_data.bsdfs_data.correction_2_0;
        else
            correction_rough_0_5 = render_data.bsdfs_data.correction_other;

        if (relative_eta <= 1.01f)
            correction_rough_1_0 = 2.509375f;
        else if (relative_eta <= 1.03f)
            correction_rough_1_0 = 2.5f;
        else if (relative_eta <= 1.1f)
            correction_rough_1_0 = 2.5225f;
        else if (relative_eta <= 1.3f)
            correction_rough_1_0 = 2.545f;
        else if (relative_eta <= 1.5f)
            correction_rough_1_0 = 2.59f;
        else if (relative_eta <= 2.0f)
            correction_rough_1_0 = 2.8f;
        else
            correction_rough_1_0 = 3.0f;

        exponent_correction = hippt::lerp(correction_rough_0_5, correction_rough_1_0, (roughness - 0.5f) * 2.0f * (roughness - 0.5f) * 2.0f);
    }

    return exponent_correction;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float get_GGX_energy_compensation_dielectrics(const HIPRTRenderData& render_data, const SimplifiedRendererMaterial& material, const RayVolumeState& ray_volume_state, float eta_t, float eta_i, float relative_eta, float NoV)
{
    float compensation_term = 1.0f;
#if PrincipledBSDFGGXUseMultipleScattering == KERNEL_OPTION_TRUE
    // Not doing energy compensation if the thin-film is fully present
    // See the // TODO FIX THIS HORROR below
    //
    // Also not doing compensation if we already have full compensation on the material
    // because the energy conservation of the glass lobe here is then redundant
    bool bsdf_already_compensated = material.enforce_strong_energy_conservation && PrincipledBSDFEnforceStrongEnergyConservation == KERNEL_OPTION_TRUE;
    if (material.thin_film < 1.0f && !bsdf_already_compensated)
    {
        bool inside_object = ray_volume_state.inside_material;
        float relative_eta_for_correction = inside_object ? 1.0f / relative_eta : relative_eta;
        float exponent_correction = GGX_glass_energy_conservation_get_correction_exponent(render_data, material.roughness, relative_eta_for_correction);

        // We're storing cos_theta_o^2.5 in the LUT so we're retrieving it with pow(1.0f / 2.5f) i.e.
        // sqrt 2.5
        //
        // We're using a "correction exponent" to forcefully get rid of energy gains at grazing angles due
        // to float precision issues: storing in the LUT with cos_theta^2.5 but fetching with pow(1.0f / 2.6f)
        // for example darkens to overall appearance and helps remove energy gains
        float view_direction_tex_fetch = powf(hippt::max(0.0f, NoV), 1.0f / exponent_correction);

        float F0 = F0_from_eta(eta_t, eta_i);
        // sqrt(sqrt()) of F0 here because we're storing F0^4 in the LUT
        float F0_remapped = sqrt(sqrt(F0));

        float3 uvw = make_float3(view_direction_tex_fetch, material.roughness, F0_remapped);

        void* texture = inside_object ? render_data.bsdfs_data.GGX_Ess_glass_inverse : render_data.bsdfs_data.GGX_Ess_glass;
        int3 dims = make_int3(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR);
        compensation_term = sample_texture_3D_rgb_32bits(texture, dims, uvw, render_data.bsdfs_data.use_hardware_tex_interpolation).r;

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
        compensation_term = hippt::lerp(compensation_term, 1.0f, material.thin_film * material.roughness);
    }
#endif

    return compensation_term;
}

#endif
