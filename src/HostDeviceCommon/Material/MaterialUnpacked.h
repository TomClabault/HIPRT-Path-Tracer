/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_UNPACKED_H
#define HOST_DEVICE_COMMON_MATERIAL_UNPACKED_H

#include "HostDeviceCommon/Material/MaterialPacked.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"

 /**
  * Packed material for use in the shaders
  */
struct DeviceUnpackedEffectiveMaterial
{
    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r)
            || !hippt::is_zero(emission.g)
            || !hippt::is_zero(emission.b)
            || emissive_texture_used;
    }

    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
    ColorRGB32F base_color = ColorRGB32F(1.0f);

    float roughness = 0.3f;
    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian

    // Parameters for Adobe 2023 F82-tint model
    float metallic = 0.0f;
    float metallic_F90_falloff_exponent = 5.0f;
    // F0 is not here as it uses the 'base_color' of the material
    ColorRGB32F metallic_F82 = ColorRGB32F(1.0f);
    ColorRGB32F metallic_F90 = ColorRGB32F(1.0f);
    float anisotropy = 0.0f;
    float anisotropy_rotation = 0.0f;
    float second_roughness_weight = 0.0f;
    float second_roughness = 0.5f;

    // Specular intensity
    float specular = 1.0f;
    // Specular tint intensity.
    // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
    float specular_tint = 1.0f;
    ColorRGB32F specular_color = ColorRGB32F(1.0f);
    // Same as coat darkening but for total internal reflection inside the specular layer
    // that sits on top of the diffuse base
    //
    // Disabled by default for artistic "expectations"
    float specular_darkening = 0.0f;

    float coat = 0.0f;
    ColorRGB32F coat_medium_absorption = ColorRGB32F{ 1.0f, 1.0f, 1.0f };
    // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
    // that will happen inside the coat
    float coat_medium_thickness = 5.0f;
    float coat_roughness = 0.0f;
    // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
    // i.e. the specular/metallic/transmission layers.
    // 
    // The option is however given here to artistically disable
    // that behavior by using coat roughening = 0.0f.
    float coat_roughening = 1.0f;
    // Because of the total internal reflection that can happen inside the coat layer (i.e.
    // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
    // clearcoat will appear will increased saturation.
    float coat_darkening = 1.0f;
    float coat_anisotropy = 0.0f;
    float coat_anisotropy_rotation = 0.0f;
    float coat_ior = 1.5f;

    float sheen = 0.0f; // Sheen strength
    float sheen_roughness = 0.5f;
    ColorRGB32F sheen_color = ColorRGB32F(1.0f);

    float ior = 1.40f;
    float specular_transmission = 0.0f;
    // At what distance is the light absorbed to the given absorption_color
    float absorption_at_distance = 1.0f;
    // Color of the light absorption when traveling through the medium
    ColorRGB32F absorption_color = ColorRGB32F(1.0f);
    float dispersion_scale = 0.0f;
    float dispersion_abbe_number = 20.0f;

    float thin_film = 0.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_hue_shift_degrees = 0.0f;
    float thin_film_base_ior_override = 1.0f;

    // 1.0f makes the material completely opaque
    // 0.0f completely transparent (becomes invisible)
    float alpha_opacity = 1.0f;

    int energy_preservation_monte_carlo_samples = 12;

    // Nested dielectric parameter
    unsigned char dielectric_priority = 0;
    
    /**
     * The booleans are moved to the end of the structure to avoid too much structure packing
     */

    // Whether or not to do energy compensation of the metallic layer
    // for that material
    bool do_metallic_energy_compensation = true;
    // Whether or not to do energy compensation of the specular/diffuse layer
    // for that material
    bool do_specular_energy_compensation = true;
    // Whether or not to do energy compensation of the clearcoat layer
    // for that material
    bool do_coat_energy_compensation = true;
    bool thin_walled = false;
    // Whether or not to do energy compensation of the glass layer
    // for that material
    bool do_glass_energy_compensation = true;
    bool thin_film_do_ior_override = false;

    // If true, 'energy_preservation_monte_carlo_samples' will be used
    // to compute the directional albedo of this material.
    // This computed directional albedo is then used to ensure perfect energy conservation
    // and preservation. 
    // 
    // This is however very expensive.
    // This is usually only needed on clearcoated materials (but even then, the energy loss due to the absence of multiple scattering between
    // the clearcoat layer and the BSDF below may be acceptable).
    // 
    // Non-clearcoated materials can already ensure perfect (modulo implementation quality) energy 
    // conservation/preservation with the precomputed LUTs [Turquin, 2019]. 
    // 
    // See PrincipledBSDFDoEnergyCompensation in this codebase.
    bool enforce_strong_energy_conservation = false;
    bool emissive_texture_used = false;
};

struct DeviceUnpackedTexturedMaterial : public DeviceUnpackedEffectiveMaterial
{
    int normal_map_texture_index = MaterialUtils::NO_TEXTURE;

    int emission_texture_index = MaterialUtils::NO_TEXTURE;
    int base_color_texture_index = MaterialUtils::NO_TEXTURE;

    // If not MaterialUtils::NO_TEXTURE, there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness
    int roughness_metallic_texture_index = MaterialUtils::NO_TEXTURE;
    int roughness_texture_index = MaterialUtils::NO_TEXTURE;
    int metallic_texture_index = MaterialUtils::NO_TEXTURE;
    int anisotropic_texture_index = MaterialUtils::NO_TEXTURE;

    int specular_texture_index = MaterialUtils::NO_TEXTURE;
    int coat_texture_index = MaterialUtils::NO_TEXTURE;
    int sheen_texture_index = MaterialUtils::NO_TEXTURE;
    int specular_transmission_texture_index = MaterialUtils::NO_TEXTURE;
};

#endif
