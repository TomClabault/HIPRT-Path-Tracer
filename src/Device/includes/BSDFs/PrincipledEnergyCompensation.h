/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_PRINCIPLED_ENERGY_COMPENSATION_H
#define DEVICE_PRINCIPLED_ENERGY_COMPENSATION_H

#include "Device/includes/BSDFs/BSDFContext.h"

#include "HostDeviceCommon/Color.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float principled_specular_relative_ior(const DeviceUnpackedEffectiveMaterial& material, float incident_medium_ior);

HIPRT_HOST_DEVICE HIPRT_INLINE float get_principled_energy_compensation_glossy_base(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, float incident_medium_ior, float NoV, int current_bounce)
{
    bool energy_compensation_disabled = !material.do_specular_energy_compensation;
    bool roughness_low_enough = material.roughness < render_data.bsdfs_data.energy_compensation_roughness_threshold;
    // If all we have for the glossy base is the diffuse layer (i.e. no specular
    // layer because the specular weight is low, then we don't need energy compensation)
    bool no_specular_layer = material.specular < 1.0e-3f;
    bool max_bounce_reached = current_bounce > render_data.bsdfs_data.glossy_base_energy_compensation_max_bounce && render_data.bsdfs_data.glossy_base_energy_compensation_max_bounce > -1;
    bool invalid_view_direction = NoV < 0.0f;
    if (energy_compensation_disabled || roughness_low_enough || no_specular_layer || max_bounce_reached || invalid_view_direction)
        return 1.0f;

    float ms_compensation = 1.0f;

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoSpecularEnergyCompensation == KERNEL_OPTION_TRUE
    int3 texture_dims = make_int3(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);

    float ior = material.ior;
    float relative_ior = principled_specular_relative_ior(material, incident_medium_ior);
    if (hippt::abs(relative_ior - 1.0f) < 1.0e-3f)
        // If the relative ior is very close to 1.0f,
        // adding some offset to avoid singularities at 1.0f which cause
        // fireflies
        relative_ior += 1.0e-3f;

    // We're storing cos_theta_o^2.5 in the LUT so we're retrieving with
    // root 2.5
    float view_dir_remapped = pow(NoV, 1.0f / 2.5f);
    // sqrt(sqrt(F0)) here because we're storing F0^4 in the LUT
    float F0_remapped = sqrt(sqrt(F0_from_eta_t_and_relative_ior(ior, relative_ior)));

    float3 uvw = make_float3(view_dir_remapped, material.roughness, F0_remapped);
    float multiple_scattering_compensation = sample_texture_3D_rgb_32bits(render_data.bsdfs_data.glossy_dielectric_directional_albedo, texture_dims, uvw, render_data.bsdfs_data.use_hardware_tex_interpolation).r;

    // Applying the compensation term for energy preservation
    // If material.specular == 1, then we want the full energy compensation
    // If material.specular == 0, then we only have the diffuse lobe and so we
    // need no energy compensation at all and so we just divide by 1 to basically do nothing
    ms_compensation = hippt::lerp(1.0f, multiple_scattering_compensation, material.specular);
    // Multi scatter compensation is not tabulated to take thin film interference into account.
    // That's because thin film interference completely modifies the fresnel term and the
    // tabulated multi scatter compensation only accounts for the usual dielectric fresnel
    // 
    // So we're progressively disabling ms compensation on the glossy base as the thin-film 
    // is more and more pronounced
    ms_compensation = hippt::lerp(ms_compensation, 1.0f, material.thin_film);
#endif

    return ms_compensation;
}

/**
 * This function gives an approximation of the energy lost by the clearcoat layer
 * by assuming that whatever is under the clearcoat is lambertian (which *may*
 * obviously be a very rough approximation, depending on what's the BSDF below the clearcoat)
 * 
 * This basically treats the clearcoat layer exactly the same as a specular/diffuse
 * (just like the "glossy base" of the principled BSDF) and so that's why we're using the LUTs
 * of the glossy base
 * 
 * The approximation can be harsh but in reasonable scenarios (where we're clearcoating something 
 * quite diffuse: which is usually the case because WHO CLEARCOATS A MIRROR?), it's actually quite good and 
 * it's way better than nothing and cheap compared to the full on-the-fly integration that we
 * would have to do otherwise (or full interlayer-multiple-scattering simulation)
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float get_principled_energy_compensation_clearcoat_lobe(const HIPRTRenderData& render_data, const DeviceUnpackedEffectiveMaterial& material, float incident_medium_ior, float NoV, int current_bounce)
{
    bool energy_compensation_disabled = !material.do_specular_energy_compensation;
    // If we don't have a clearcoat, let's not compensate energy
    bool no_coat_layer = material.coat < 1.0e-3f;
    bool max_bounce_reached = current_bounce > render_data.bsdfs_data.clearcoat_energy_compensation_max_bounce && render_data.bsdfs_data.clearcoat_energy_compensation_max_bounce > -1;
    bool invalid_view_direction = NoV < 0.0f;
    if (energy_compensation_disabled || no_coat_layer || max_bounce_reached || invalid_view_direction)
        return 1.0f;

    float ms_compensation = 1.0f;

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoClearcoatEnergyCompensation == KERNEL_OPTION_TRUE
    int3 texture_dims = make_int3(GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_COS_THETA_O, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_ROUGHNESS, GPUBakerConstants::GLOSSY_DIELECTRIC_TEXTURE_SIZE_IOR);

    if (hippt::abs(material.coat_ior / incident_medium_ior - 1.0f) < 1.0e-3f)
        // If the relative ior is very close to 1.0f,
        // adding some offset to avoid singularities which cause
        // fireflies
        incident_medium_ior += 1.0e-3f;

    // We're storing cos_theta_o^2.5 in the LUT so we're retrieving with
    // root 2.5
    float view_dir_remapped = pow(NoV, 1.0f / 2.5f);
    // sqrt(sqrt(F0)) here because we're storing F0^4 in the LUT
    float F0_remapped = sqrt(sqrt(F0_from_eta(material.coat_ior, incident_medium_ior)));

    float3 uvw = make_float3(view_dir_remapped, material.coat_roughness, F0_remapped);
    float multiple_scattering_compensation = sample_texture_3D_rgb_32bits(render_data.bsdfs_data.glossy_dielectric_directional_albedo, texture_dims, uvw, render_data.bsdfs_data.use_hardware_tex_interpolation).r;

    // Applying the compensation term for energy preservation
    // If material.coat == 1, then we want the full energy compensation
    // If material.coat == 0, then we only have the diffuse lobe and so we
    // need no energy compensation at all and so we just divide by 1 to basically do nothing
    //
    // We're also disabling the compensation when the clearcoat is on top of a glass
    // transmission lobe because the approximation here falls apart and can gain quite a bit
    // of energy.
    ms_compensation = hippt::lerp(1.0f, multiple_scattering_compensation, material.coat * (1.0f - material.specular_transmission));
    // Multi scatter compensation is not tabulated to take thin film interference into account.
    // That's because thin film interference completely modifies the fresnel term and the
    // tabulated multi scatter compensation only accounts for the usual dielectric fresnel
    // 
    // So we're progressively disabling ms compensation on the glossy base as the thin-film 
    // is more and more pronounced
    ms_compensation = hippt::lerp(ms_compensation, 1.0f, material.thin_film);
#endif

    return ms_compensation;
}

#endif
