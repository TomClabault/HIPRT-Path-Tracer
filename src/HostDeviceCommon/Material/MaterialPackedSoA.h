/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_PACKED_SOA_H
#define HOST_DEVICE_COMMON_MATERIAL_PACKED_SOA_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialPacked.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"

struct DevicePackedEffectiveMaterialSoA
{
    HIPRT_HOST_DEVICE ColorRGB32F get_emission(int material_index) const { return this->emission[material_index]; }
    HIPRT_HOST_DEVICE bool get_emissive_texture_used(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::PACKED_EMISSIVE_TEXTURE_USED>(); }

    HIPRT_HOST_DEVICE ColorRGB32F get_base_color(int material_index) const { return base_color_roughness[material_index].get_color(); }

    HIPRT_HOST_DEVICE float get_roughness(int material_index) const { return base_color_roughness[material_index].get_float(); }
    HIPRT_HOST_DEVICE float get_oren_nayar_sigma(int material_index) const { return this->oren_nayar_sigma[material_index]; }

    HIPRT_HOST_DEVICE float get_metallic(int material_index) const { return metallic_F90_and_metallic[material_index].get_float(); }
    HIPRT_HOST_DEVICE float get_metallic_F90_falloff_exponent(int material_index) const { return this->metallic_F90_falloff_exponent[material_index]; }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F82(int material_index) const { return metallic_F82_packed_and_diffuse_transmission[material_index].get_color(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F90(int material_index) const { return metallic_F90_and_metallic[material_index].get_color(); }
    HIPRT_HOST_DEVICE float get_anisotropy(int material_index) const { return anisotropy_and_rotation_and_second_roughness[material_index].get_float<DevicePackedEffectiveMaterial::PackedAnisotropyGroupIndices::PACKED_ANISOTROPY>(); }
    HIPRT_HOST_DEVICE float get_anisotropy_rotation(int material_index) const { return anisotropy_and_rotation_and_second_roughness[material_index].get_float<DevicePackedEffectiveMaterial::PackedAnisotropyGroupIndices::PACKED_ANISOTROPY_ROTATION>(); }
    HIPRT_HOST_DEVICE float get_second_roughness_weight(int material_index) const { return anisotropy_and_rotation_and_second_roughness[material_index].get_float<DevicePackedEffectiveMaterial::PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS_WEIGHT>(); }
    HIPRT_HOST_DEVICE float get_second_roughness(int material_index) const { return anisotropy_and_rotation_and_second_roughness[material_index].get_float<DevicePackedEffectiveMaterial::PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE bool get_do_metallic_energy_compensation(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::METALLIC_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_specular(int material_index) const { return specular_and_darkening_and_coat_roughness[material_index].get_float<DevicePackedEffectiveMaterial::PackedSpecularGroupIndices::PACKED_SPECULAR>(); }
    HIPRT_HOST_DEVICE float get_specular_tint(int material_index) const { return specular_color_and_tint_factor[material_index].get_float(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_specular_color(int material_index) const { return specular_color_and_tint_factor[material_index].get_color(); }
    HIPRT_HOST_DEVICE float get_specular_darkening(int material_index) const { return specular_and_darkening_and_coat_roughness[material_index].get_float<DevicePackedEffectiveMaterial::PackedSpecularGroupIndices::PACKED_SPECULAR_DARKENING>(); }
    HIPRT_HOST_DEVICE bool get_do_specular_energy_compensation(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::SPECULAR_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_coat(int material_index) const { return coat_and_medium_absorption[material_index].get_float(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_coat_medium_absorption(int material_index) const { return coat_and_medium_absorption[material_index].get_color(); }
    HIPRT_HOST_DEVICE float get_coat_medium_thickness(int material_index) const { return this->coat_medium_thickness[material_index]; }
    HIPRT_HOST_DEVICE float get_coat_roughness(int material_index) const { return specular_and_darkening_and_coat_roughness[material_index].get_float<DevicePackedEffectiveMaterial::PackedSpecularGroupIndices::PACKED_COAT_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE float get_coat_roughening(int material_index) const { return coat_roughening_darkening_anisotropy_and_rotation[material_index].get_float<DevicePackedEffectiveMaterial::PackedCoatGroupIndices::PACKED_COAT_ROUGHENING>(); }
    HIPRT_HOST_DEVICE float get_coat_darkening(int material_index) const { return coat_roughening_darkening_anisotropy_and_rotation[material_index].get_float<DevicePackedEffectiveMaterial::PackedCoatGroupIndices::PACKED_COAT_DARKENING>(); }
    HIPRT_HOST_DEVICE float get_coat_anisotropy(int material_index) const { return coat_roughening_darkening_anisotropy_and_rotation[material_index].get_float<DevicePackedEffectiveMaterial::PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY>(); }
    HIPRT_HOST_DEVICE float get_coat_anisotropy_rotation(int material_index) const { return coat_roughening_darkening_anisotropy_and_rotation[material_index].get_float<DevicePackedEffectiveMaterial::PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY_ROTATION>(); }
    HIPRT_HOST_DEVICE float get_coat_ior(int material_index) const { return this->coat_ior[material_index]; }
    HIPRT_HOST_DEVICE bool get_do_coat_energy_compensation(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::CLEARCOAT_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_sheen(int material_index) const { return sheen_and_color[material_index].get_float(); }
    HIPRT_HOST_DEVICE float get_sheen_roughness(int material_index) const { return sheen_roughness_transmission_dispersion_thin_film[material_index].get_float<DevicePackedEffectiveMaterial::PackedSheenRoughnessGroupIndices::PACKED_SHEEN_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_sheen_color(int material_index) const { return sheen_and_color[material_index].get_color(); }

    HIPRT_HOST_DEVICE float get_ior(int material_index) const { return this->ior[material_index]; }
    HIPRT_HOST_DEVICE float get_specular_transmission(int material_index) const { return sheen_roughness_transmission_dispersion_thin_film[material_index].get_float<DevicePackedEffectiveMaterial::PackedSheenRoughnessGroupIndices::PACKED_SPECULAR_TRANSMISSION>(); }
    HIPRT_HOST_DEVICE float get_diffuse_transmission(int material_index) const { return metallic_F82_packed_and_diffuse_transmission[material_index].get_float(); }
    HIPRT_HOST_DEVICE float get_absorption_at_distance(int material_index) const { return this->absorption_at_distance[material_index]; }
    HIPRT_HOST_DEVICE ColorRGB32F get_absorption_color(int material_index) const { return absorption_color_packed[material_index].get_color(); }
    HIPRT_HOST_DEVICE float get_dispersion_scale(int material_index) const { return sheen_roughness_transmission_dispersion_thin_film[material_index].get_float<DevicePackedEffectiveMaterial::PackedSheenRoughnessGroupIndices::PACKED_DISPERSION_SCALE>(); }
    HIPRT_HOST_DEVICE float get_dispersion_abbe_number(int material_index) const { return this->dispersion_abbe_number[material_index]; }
    HIPRT_HOST_DEVICE bool get_thin_walled(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::PACKED_THIN_WALLED >(); }
    HIPRT_HOST_DEVICE bool get_do_glass_energy_compensation(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::GLASS_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_thin_film(int material_index) const { return sheen_roughness_transmission_dispersion_thin_film[material_index].get_float<DevicePackedEffectiveMaterial::PackedSheenRoughnessGroupIndices::PACKED_THIN_FILM>(); }
    HIPRT_HOST_DEVICE float get_thin_film_ior(int material_index) const { return this->thin_film_ior[material_index]; }
    HIPRT_HOST_DEVICE float get_thin_film_thickness(int material_index) const { return this->thin_film_thickness[material_index]; }
    HIPRT_HOST_DEVICE float get_thin_film_kappa_3(int material_index) const { return this->thin_film_kappa_3[material_index]; }
    HIPRT_HOST_DEVICE float get_thin_film_hue_shift_degrees(int material_index) const { return alpha_thin_film_hue_dielectric_priority[material_index].get_float<DevicePackedEffectiveMaterial::PackedAlphaOpacityGroupIndices::PACKED_THIN_FILM_HUE_SHIFT>(); }
    HIPRT_HOST_DEVICE float get_thin_film_base_ior_override(int material_index) const { return this->thin_film_base_ior_override[material_index]; }
    HIPRT_HOST_DEVICE bool get_thin_film_do_ior_override(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::PACKED_THIN_FILM_DO_IOR_OVERRIDE>(); }

    HIPRT_HOST_DEVICE float get_alpha_opacity(int material_index) const { return alpha_thin_film_hue_dielectric_priority[material_index].get_float<DevicePackedEffectiveMaterial::PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(); }
    HIPRT_HOST_DEVICE unsigned char get_dielectric_priority(int material_index) const { return alpha_thin_film_hue_dielectric_priority[material_index].get_uchar<DevicePackedEffectiveMaterial::PackedAlphaOpacityGroupIndices::PACKED_DIELECTRIC_PRIORITY>(); }

    HIPRT_HOST_DEVICE unsigned char get_energy_preservation_monte_carlo_samples(int material_index) const { return alpha_thin_film_hue_dielectric_priority[material_index].get_uchar<DevicePackedEffectiveMaterial::PackedAlphaOpacityGroupIndices::PACKED_ENERGY_PRESERVATION_SAMPLES>(); }
    HIPRT_HOST_DEVICE bool get_enforce_strong_energy_conservation(int material_index) const { return flags[material_index].get_bool<DevicePackedEffectiveMaterial::PackedFlagsIndices::PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION>(); }

    // Packed flags of the material:
    //  - thin_walled
    //      Is the material thin walled? i.e. it doesn't have an interior and light doesn't
    //      bend as it goes through
    // 
    //  - emissive_texture_used
    //      Does the material use an emissive texture?
    // 
    //  - thin_film_do_ior_override
    //      Whether or not to override the IORs used for the base material on top of which
    //      the thin film sits.
    // 
    //  - enforce_strong_energy_conservation
    //      If true, 'energy_preservation_monte_carlo_samples' will be used
    //      to compute the directional albedo of this material.
    //      This computed directional albedo is then used to ensure perfect energy conservation
    //      and preservation. 
    // 
    //      This is however very expensive.
    //      This is usually only needed on clearcoated materials (but even then, the energy loss due to the absence of multiple scattering between
    //      the clearcoat layer and the BSDF below may be acceptable).
    // 
    //      Non-clearcoated materials can already ensure perfect (modulo implementation quality) energy 
    //      conservation/preservation with the precomputed LUTs [Turquin, 2019]. 
    // 
    //      See PrincipledBSDFDoEnergyCompensation in this codebase.
    //
    //      Values from the 'PackedFlagsIndices' enum should be used
    //      to retrieve/set from the packed flags
    UChar8BoolsPacked* flags = nullptr;

    // Full range emission
    ColorRGB32F* emission = nullptr;

    // Base color RGB 3x8 bits + roughness uchar [float in [0,1] packed in 8 bit]
    ColorRGB24bFloat0_1Packed* base_color_roughness = nullptr;

    float* oren_nayar_sigma = nullptr;

    // Parameters for Adobe 2023 F82-tint model
    // Packs the SDR F90 color and the metalness parameter
    ColorRGB24bFloat0_1Packed* metallic_F90_and_metallic;
    ColorRGB24bFloat0_1Packed* metallic_F82_packed_and_diffuse_transmission;
    float* metallic_F90_falloff_exponent = nullptr;

    Float4xPacked* anisotropy_and_rotation_and_second_roughness = nullptr;

    // Packed specular color and the intensity of the tint
    // 
    // Specular tint intensity: Specular will be white if 0.0f and will be 'specular_color' if 1.0f
    ColorRGB24bFloat0_1Packed* specular_color_and_tint_factor = nullptr;

    // Packed:
    //  - specular_darkening
    //      Same as coat darkening but for total internal reflection inside the specular layer
    //      that sits on top of the diffuse base
    //
    //      Disabled by default for artistic "expectations"
    //
    //  - Specular
    //      Specular intensity
    //
    //  - Coat roughness
    //      Roughness of the coat 
    // TODO: PACKED 1 FLOAT IS UNUSED IN HERE
    Float4xPacked* specular_and_darkening_and_coat_roughness = nullptr;
    float* coat_medium_thickness = nullptr;

    // Packed:
    //  - Coat
    //      Intensity of the coat. 0.0f disables the coating
    //
    //  - Coat medium absorption color 
    ColorRGB24bFloat0_1Packed* coat_and_medium_absorption = nullptr;

    // Packed:
    //  - Coat roughening
    //      Physical accuracy requires that a rough clearcoat also roughens what's underneath it
    //      i.e. the specular/metallic/transmission layers.
    // 
    //      The option is however given here to artistically disable
    //      that behavior by using coat roughening = 0.0f.
    //
    //  - Coat darkening
    //      Because of the total internal reflection that can happen inside the coat layer (i.e.
    //      light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
    //      clearcoat will appear will increased saturation.
    //     
    //  - Coat anisotropy
    //  - Coat anisotropy rotation
    Float4xPacked* coat_roughening_darkening_anisotropy_and_rotation = nullptr;
    float* coat_ior = nullptr;

    // Packed:
    //  - Sheen intensity. 0.0f disables the sheen effect
    //
    //  - Sheen color
    ColorRGB24bFloat0_1Packed* sheen_and_color = nullptr;

    // IOR of the base material
    float* ior = nullptr;

    // Packed:
    //  - Absorption color
    //      Color of the light absorption when traveling through the medium
    // TODO: PACKED FLOAT IS UNUSED IN HERE
    ColorRGB24bFloat0_1Packed* absorption_color_packed = nullptr;
    float* absorption_at_distance = nullptr;

    // Packed:
    //  - Sheen roughness
    // 
    //  - Specular transmission
    //      How much light is transmitted through the material. This essentially controls the glass lobe
    //
    //  - Dispersion scale
    //      Intensity of the dispersion effect in glass objects
    //
    //  - Thin film
    //      Intensity of the thin-film effect
    Float4xPacked* sheen_roughness_transmission_dispersion_thin_film = nullptr;

    float* dispersion_abbe_number = nullptr;
    float* thin_film_ior = nullptr;
    float* thin_film_thickness = nullptr;
    float* thin_film_kappa_3 = nullptr;
    float* thin_film_base_ior_override = nullptr;

    // Packed:
    //  - Alpha opacity
    //      1.0f makes the material completely opaque
    //      0.0f completely transparent (becomes invisible)
    //
    //  - Thin film hue shift in degrees
    // 
    //  - Dielectric priority
    //      Nested dielectric with priority parameter
    //
    //  - Energy preservation samples
    //      How many samples will be computed for the integration of the directional
    //      when the strong energy preservation/conservation of the material is enabled
    Float2xUChar2xPacked* alpha_thin_film_hue_dielectric_priority = nullptr;
};

struct DevicePackedTexturedMaterialSoA : public DevicePackedEffectiveMaterialSoA
{
    HIPRT_HOST_DEVICE unsigned short int get_normal_map_texture_index(int material_index) const { return normal_map_emission_index[material_index].get_value<DevicePackedTexturedMaterial::NormalMapEmissionIndices::NORMAL_MAP_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_emission_texture_index(int material_index) const { return normal_map_emission_index[material_index].get_value<DevicePackedTexturedMaterial::NormalMapEmissionIndices::EMISSION_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_base_color_texture_index(int material_index) const { return base_color_roughness_metallic_index[material_index].get_value<DevicePackedTexturedMaterial::BaseColorRoughnessMetallicIndices::BASE_COLOR_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_metallic_texture_index(int material_index) const { return base_color_roughness_metallic_index[material_index].get_value<DevicePackedTexturedMaterial::BaseColorRoughnessMetallicIndices::ROUGHNESS_METALLIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_texture_index(int material_index) const { return roughness_and_metallic_index[material_index].get_value<DevicePackedTexturedMaterial::RoughnessAndMetallicIndices::ROUGHNESS_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_metallic_texture_index(int material_index) const { return roughness_and_metallic_index[material_index].get_value<DevicePackedTexturedMaterial::RoughnessAndMetallicIndices::METALLIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_anisotropic_texture_index(int material_index) const { return roughness_and_metallic_index[material_index].get_value<DevicePackedTexturedMaterial::AnisotropicSpecularIndices::ANISOTROPIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_specular_texture_index(int material_index) const { return anisotropic_specular_index[material_index].get_value<DevicePackedTexturedMaterial::AnisotropicSpecularIndices::SPECULAR_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_coat_texture_index(int material_index) const { return coat_sheen_index[material_index].get_value<DevicePackedTexturedMaterial::CoatSheenIndices::COAT_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_sheen_texture_index(int material_index) const { return coat_sheen_index[material_index].get_value<DevicePackedTexturedMaterial::CoatSheenIndices::SHEEN_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_specular_transmission_texture_index(int material_index) const { return specular_transmission_index[material_index].get_value<DevicePackedTexturedMaterial::SpecularTransmissionIndex::SPECULAR_TRANSMISSION_INDEX>(); }

    /**
     * The 'DevicePackedTexturedMaterial' returned contains all the data read 
     * in all the arrays the SoA material struct for the given 'material_index'
     * 
     * Note that, for example, even if the material has its 'coat' parameter at 0.0f
     * (i.e. it doesn't use coat at all), all the coat parameters (coat thickness, roughness, anisotropy, .....) 
     * will be loaded from global memory and this will be useless global memory accesses
     * 
     * A more performant alternative of this function is 'read_partial_material(int material_index)'
     * 
     * This function is fully commented out because it's actually never used
     */
    //HIPRT_HOST_DEVICE DevicePackedTexturedMaterial read_full_textured_material(int material_index) const
    //{
    //    DevicePackedTexturedMaterial out;

    //    out.set_normal_map_texture_index(this->get_normal_map_texture_index(material_index));
    //    out.set_emission_texture_index(this->get_emission_texture_index(material_index));
    //    out.set_base_color_texture_index(this->get_base_color_texture_index(material_index));

    //    out.set_roughness_metallic_texture_index(this->get_roughness_metallic_texture_index(material_index));
    //    out.set_roughness_texture_index(this->get_roughness_texture_index(material_index));
    //    out.set_metallic_texture_index(this->get_metallic_texture_index(material_index));
    //    out.set_anisotropic_texture_index(this->get_anisotropic_texture_index(material_index));

    //    out.set_specular_texture_index(this->get_specular_texture_index(material_index));
    //    out.set_coat_texture_index(this->get_coat_texture_index(material_index));
    //    out.set_sheen_texture_index(this->get_sheen_texture_index(material_index));
    //    out.set_specular_transmission_texture_index(this->get_specular_transmission_texture_index(material_index));






    //    out.set_emission(this->get_emission(material_index));
    //    out.set_emissive_texture_used(this->get_emissive_texture_used(material_index));

    //    out.set_base_color(this->get_base_color(material_index));

    //    out.set_roughness(this->get_roughness(material_index));
    //    out.set_oren_nayar_sigma(this->get_oren_nayar_sigma(material_index));

    //    // Parameters for Adobe 2023 F82-tint model
    //    out.set_metallic(this->get_metallic(material_index));
    //    out.set_metallic_F90_falloff_exponent(this->get_metallic_F90_falloff_exponent(material_index));
    //    // F0 is not here as it uses the 'base_color' of the material
    //    out.set_metallic_F82(this->get_metallic_F82(material_index));
    //    out.set_metallic_F90(this->get_metallic_F90(material_index));
    //    out.set_anisotropy(this->get_anisotropy(material_index));
    //    out.set_anisotropy_rotation(this->get_anisotropy_rotation(material_index));
    //    out.set_second_roughness_weight(this->get_second_roughness_weight(material_index));
    //    out.set_second_roughness(this->get_second_roughness(material_index));
    //    out.set_metallic_energy_compensation(this->get_do_metallic_energy_compensation(material_index));

    //    // Specular intensity
    //    out.set_specular(this->get_specular(material_index));
    //    // Specular tint intensity. 
    //    // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
    //    out.set_specular_tint(this->get_specular_tint(material_index));
    //    out.set_specular_color(this->get_specular_color(material_index));
    //    // Same as coat darkening but for total internal reflection inside the specular layer
    //    // that sits on top of the diffuse base
    //    //
    //    // Disabled by default for artistic "expectations"
    //    out.set_specular_darkening(this->get_specular_darkening(material_index));
    //    out.set_specular_energy_compensation(this->get_do_specular_energy_compensation(material_index));

    //    out.set_coat(this->get_coat(material_index));
    //    out.set_coat_medium_absorption(this->get_coat_medium_absorption(material_index));
    //    // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
    //    // that will happen inside the coat
    //    out.set_coat_medium_thickness(this->get_coat_medium_thickness(material_index));
    //    out.set_coat_roughness(this->get_coat_roughness(material_index));
    //    // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
    //    // i.e. the specular/metallic/transmission layers.
    //    // 
    //    // The option is however given here to artistically disable
    //    // that behavior by using coat roughening = 0.0f.
    //    out.set_coat_roughening(this->get_coat_roughening(material_index));
    //    // Because of the total internal reflection that can happen inside the coat layer (i.e.
    //    // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
    //    // clearcoat will appear will increased saturation.
    //    out.set_coat_darkening(this->get_coat_darkening(material_index));
    //    out.set_coat_anisotropy(this->get_coat_anisotropy(material_index));
    //    out.set_coat_anisotropy_rotation(this->get_coat_anisotropy_rotation(material_index));
    //    out.set_coat_ior(this->get_coat_ior(material_index));
    //    out.set_coat_energy_compensation(this->get_do_coat_energy_compensation(material_index));

    //    out.set_sheen(this->get_sheen(material_index)); // Sheen strength
    //    out.set_sheen_roughness(this->get_sheen_roughness(material_index));
    //    out.set_sheen_color(this->get_sheen_color(material_index));

    //    out.set_ior(this->get_ior(material_index));
    //    out.set_specular_transmission(this->get_specular_transmission(material_index));

    //    // At what distance is the light absorbed to the given absorption_color
    //    out.set_absorption_at_distance(this->get_absorption_at_distance(material_index));
    //    // Color of the light absorption when traveling through the medium
    //    out.set_absorption_color(this->get_absorption_color(material_index));
    //    out.set_dispersion_scale(this->get_dispersion_scale(material_index));
    //    out.set_dispersion_abbe_number(this->get_dispersion_abbe_number(material_index));
    //    out.set_thin_walled(this->get_thin_walled(material_index));
    //    out.set_glass_energy_compensation(this->get_do_glass_energy_compensation(material_index));

    //    out.set_thin_film(this->get_thin_film(material_index));
    //    out.set_thin_film_ior(this->get_thin_film_ior(material_index));
    //    out.set_thin_film_thickness(this->get_thin_film_thickness(material_index));
    //    out.set_thin_film_kappa_3(this->get_thin_film_kappa_3(material_index));
    //    // Sending the hue film in [0, 1] to the GPU
    //    out.set_thin_film_hue_shift_degrees(get_thin_film_hue_shift_degrees(material_index));
    //    out.set_thin_film_base_ior_override(this->get_thin_film_base_ior_override(material_index));
    //    out.set_thin_film_do_ior_override(this->get_thin_film_do_ior_override(material_index));

    //    // 1.0f makes the material completely opaque
    //    // 0.0f completely transparent (becomes invisible)
    //    out.set_alpha_opacity(this->get_alpha_opacity(material_index));

    //    // Nested dielectric parameter
    //    out.set_dielectric_priority(this->get_dielectric_priority(material_index));

    //    out.set_energy_preservation_monte_carlo_samples(this->get_energy_preservation_monte_carlo_samples(material_index));
    //    // If true, 'energy_preservation_monte_carlo_samples' will be used
    //    // to compute the directional albedo of this material.
    //    // This computed directional albedo is then used to ensure perfect energy conservation
    //    // and preservation. 
    //    // 
    //    // This is however very expensive.
    //    // This is usually only needed on clearcoated materials (but even then, the energy loss due to the absence of multiple scattering between
    //    // the clearcoat layer and the BSDF below may be acceptable).
    //    // 
    //    // Non-clearcoated materials can already ensure perfect (modulo implementation quality) energy 
    //    // conservation/preservation with the precomputed LUTs [Turquin, 2019]. 
    //    // 
    //    // See PrincipledBSDFDoEnergyCompensation in this codebase.
    //    out.set_enforce_strong_energy_conservation(this->get_enforce_strong_energy_conservation(material_index));

    //    return out;
    //}

    /**
     * Only reads the relevant parameters of the material based on what parameters this material is using.
     * For example, if the 'coat' parameter of the material is 0.0f (i.e. the coat isn't used), none of the
     * coat parameters will be read from global memory which saves on memory traffic
     */
    HIPRT_HOST_DEVICE DevicePackedTexturedMaterial read_partial_material(int material_index) const
    {
        DevicePackedTexturedMaterial out;

        out.set_normal_map_texture_index(this->get_normal_map_texture_index(material_index));
        out.set_emission_texture_index(this->get_emission_texture_index(material_index));
        out.set_base_color_texture_index(this->get_base_color_texture_index(material_index));

        out.set_roughness_metallic_texture_index(this->get_roughness_metallic_texture_index(material_index));
        out.set_roughness_texture_index(this->get_roughness_texture_index(material_index));
        out.set_metallic_texture_index(this->get_metallic_texture_index(material_index));
        out.set_anisotropic_texture_index(this->get_anisotropic_texture_index(material_index));

        out.set_specular_texture_index(this->get_specular_texture_index(material_index));
        out.set_coat_texture_index(this->get_coat_texture_index(material_index));
        out.set_sheen_texture_index(this->get_sheen_texture_index(material_index));
        out.set_specular_transmission_texture_index(this->get_specular_transmission_texture_index(material_index));





        out.set_emissive_texture_used(this->get_emissive_texture_used(material_index));
        if (!out.get_emissive_texture_used())
            // Only loading the emission if no emissive texture is used
            out.set_emission(this->get_emission(material_index));

        if (out.get_base_color_texture_index() == MaterialUtils::NO_TEXTURE)
            // Only reading the base color if no base color texture is used
            // (because if we have a base color texture, it's going to override
            // the base color parameter anyway)
            out.set_base_color(this->get_base_color(material_index));

        if (out.get_roughness_texture_index() == MaterialUtils::NO_TEXTURE && out.get_roughness_metallic_texture_index() == MaterialUtils::NO_TEXTURE)
            // Same for the roughness
            out.set_roughness(this->get_roughness(material_index));

        out.set_oren_nayar_sigma(this->get_oren_nayar_sigma(material_index));

        // Parameters for Adobe 2023 F82-tint model
        // Only reading the metallic if no metallic texture is used
        // (because if we have a metallic texture, it's going to override
        // the metallic parameter anyway)
        out.set_metallic(this->get_metallic(material_index));
        if (out.get_metallic() > 0.0f || out.get_metallic_texture_index() != MaterialUtils::NO_TEXTURE || out.get_roughness_metallic_texture_index() != MaterialUtils::NO_TEXTURE)
        {
            // If the metallic parameter isn't 0.0f, i.e. the material does have a metallic lobe,
            // then and only then do we need to load the metallic parameters

            out.set_metallic_F90_falloff_exponent(this->get_metallic_F90_falloff_exponent(material_index));
            // F0 is not here as it uses the 'base_color' of the material
            out.set_metallic_F82(this->get_metallic_F82(material_index));
            out.set_metallic_F90(this->get_metallic_F90(material_index));

            out.set_second_roughness_weight(this->get_second_roughness_weight(material_index));
            out.set_second_roughness(this->get_second_roughness(material_index));

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoMetallicEnergyCompensation == KERNEL_OPTION_TRUE
            out.set_metallic_energy_compensation(this->get_do_metallic_energy_compensation(material_index));
#endif
        }

        if(out.get_anisotropic_texture_index() == MaterialUtils::NO_TEXTURE)
            out.set_anisotropy(this->get_anisotropy(material_index));
        out.set_anisotropy_rotation(this->get_anisotropy_rotation(material_index));

        // Specular intensity
        out.set_specular(this->get_specular(material_index));
        if (out.get_specular() > 0.0f || out.get_specular_texture_index() != MaterialUtils::NO_TEXTURE)
        {
            // We only need to read the various specular parameters if the material actually has a specular lobe

            // Specular tint intensity. 
            out.set_specular_tint(this->get_specular_tint(material_index));
            // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
            out.set_specular_color(this->get_specular_color(material_index));
            // Same as coat darkening but for total internal reflection inside the specular layer
            // that sits on top of the diffuse base
            //
            // Disabled by default for artistic "expectations"
            out.set_specular_darkening(this->get_specular_darkening(material_index));

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoSpecularEnergyCompensation == KERNEL_OPTION_TRUE
            out.set_specular_energy_compensation(this->get_do_specular_energy_compensation(material_index));
#endif
        }

        out.set_coat(this->get_coat(material_index));
        if (out.get_coat() > 0.0f || out.get_coat_texture_index() != MaterialUtils::NO_TEXTURE)
        {
            // We only need to read the coat parameters if the material has a coat lobe
            // (which is when out.get_coat() > 0.0f)

            out.set_coat_medium_absorption(this->get_coat_medium_absorption(material_index));
            // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
            // that will happen inside the coat
            out.set_coat_medium_thickness(this->get_coat_medium_thickness(material_index));
            out.set_coat_roughness(this->get_coat_roughness(material_index));
            // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
            // i.e. the specular/metallic/transmission layers.
            // 
            // The option is however given here to artistically disable
            // that behavior by using coat roughening = 0.0f.
            out.set_coat_roughening(this->get_coat_roughening(material_index));
            // Because of the total internal reflection that can happen inside the coat layer (i.e.
            // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
            // clearcoat will appear will increased saturation.
            out.set_coat_darkening(this->get_coat_darkening(material_index));
            out.set_coat_anisotropy(this->get_coat_anisotropy(material_index));
            out.set_coat_anisotropy_rotation(this->get_coat_anisotropy_rotation(material_index));
            out.set_coat_ior(this->get_coat_ior(material_index));

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoClearcoatEnergyCompensation == KERNEL_OPTION_TRUE
            out.set_coat_energy_compensation(this->get_do_coat_energy_compensation(material_index));
#endif
        }

        out.set_sheen(this->get_sheen(material_index)); // Sheen strength
        if (out.get_sheen() > 0.0f || out.get_sheen_texture_index() != MaterialUtils::NO_TEXTURE)
        {
            out.set_sheen_roughness(this->get_sheen_roughness(material_index));
            out.set_sheen_color(this->get_sheen_color(material_index));
        }

        out.set_ior(this->get_ior(material_index));

        out.set_diffuse_transmission(this->get_diffuse_transmission(material_index));
        out.set_specular_transmission(this->get_specular_transmission(material_index));
        if (out.get_specular_transmission() > 0.0f || out.get_specular_transmission_texture_index() != MaterialUtils::NO_TEXTURE)
        {
            // This is all specific to specular transmission
            out.set_dispersion_scale(this->get_dispersion_scale(material_index));
            out.set_dispersion_abbe_number(this->get_dispersion_abbe_number(material_index));
            out.set_thin_walled(this->get_thin_walled(material_index));

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoGlassEnergyCompensation == KERNEL_OPTION_TRUE
            out.set_glass_energy_compensation(this->get_do_glass_energy_compensation(material_index));
#endif
        }

        if (out.get_specular_transmission() > 0.0f || out.get_diffuse_transmission() > 0.0f || out.get_specular_transmission_texture_index() != MaterialUtils::NO_TEXTURE)
        {
            // This is also applicable to diffuse transmission
            
            // At what distance is the light absorbed to the given absorption_color
            out.set_absorption_at_distance(this->get_absorption_at_distance(material_index));
            // Color of the light absorption when traveling through the medium
            out.set_absorption_color(this->get_absorption_color(material_index));
        }

        out.set_thin_film(this->get_thin_film(material_index));
        if (out.get_thin_film() > 0.0f)
        {
            out.set_thin_film_ior(this->get_thin_film_ior(material_index));
            out.set_thin_film_thickness(this->get_thin_film_thickness(material_index));
            out.set_thin_film_kappa_3(this->get_thin_film_kappa_3(material_index));
            // Sending the hue film in [0, 1] to the GPU
            out.set_thin_film_hue_shift_degrees(get_thin_film_hue_shift_degrees(material_index));
            out.set_thin_film_base_ior_override(this->get_thin_film_base_ior_override(material_index));
            out.set_thin_film_do_ior_override(this->get_thin_film_do_ior_override(material_index));
        }

        // 1.0f makes the material completely opaque
        // 0.0f completely transparent (becomes invisible)
        out.set_alpha_opacity(this->get_alpha_opacity(material_index));

        // Nested dielectric parameter
        out.set_dielectric_priority(this->get_dielectric_priority(material_index));

#if PrincipledBSDFEnforceStrongEnergyConservation == KERNEL_OPTION_TRUE
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
        out.set_enforce_strong_energy_conservation(this->get_enforce_strong_energy_conservation(material_index));
        if (out.get_enforce_strong_energy_conservation())
            out.set_energy_preservation_monte_carlo_samples(this->get_energy_preservation_monte_carlo_samples(material_index));
#endif

        return out;
    }

    Uint2xPacked* normal_map_emission_index = nullptr;
    // If the roughness_metallic texture index is not MaterialUtils::NO_TEXTURE, 
    // then there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness.
    Uint2xPacked* base_color_roughness_metallic_index = nullptr;
    Uint2xPacked* roughness_and_metallic_index = nullptr;
    Uint2xPacked* anisotropic_specular_index = nullptr;
    Uint2xPacked* coat_sheen_index = nullptr;
    // TODO: 1 PACKED UINT IS UNUSED IN HERE
    Uint2xPacked* specular_transmission_index = nullptr;
};

#endif
