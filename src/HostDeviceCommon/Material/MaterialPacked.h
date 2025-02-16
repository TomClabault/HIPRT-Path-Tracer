/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_PACKED_H
#define HOST_DEVICE_COMMON_MATERIAL_PACKED_H

#include "HostDeviceCommon/Packing.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"

 /**
  * Packed material for use in the shaders
  */
struct DevicePackedEffectiveMaterial
{
    enum PackedFlagsIndices : unsigned char
    {
        PACKED_THIN_WALLED = 0,
        PACKED_EMISSIVE_TEXTURE_USED = 1,
        PACKED_THIN_FILM_DO_IOR_OVERRIDE = 2,
        GLASS_ENERGY_COMPENSATION = 3,
        CLEARCOAT_ENERGY_COMPENSATION = 4,
        METALLIC_ENERGY_COMPENSATION = 5,
        SPECULAR_ENERGY_COMPENSATION = 6,
        PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION = 7,
    };

    enum PackedAnisotropyGroupIndices : unsigned char
    {
        PACKED_ANISOTROPY = 0,
        PACKED_ANISOTROPY_ROTATION = 1,
        PACKED_SECOND_ROUGHNESS_WEIGHT = 2,
        PACKED_SECOND_ROUGHNESS = 3,
    };

    enum PackedSpecularGroupIndices : unsigned char
    {
        PACKED_SPECULAR = 0,
        PACKED_SPECULAR_DARKENING = 1,
        PACKED_COAT_ROUGHNESS = 2
    };

    enum PackedCoatGroupIndices : unsigned char
    {
        PACKED_COAT_ROUGHENING = 0,
        PACKED_COAT_DARKENING = 1,
        PACKED_COAT_ANISOTROPY = 2,
        PACKED_COAT_ANISOTROPY_ROTATION = 3,
    };

    enum PackedSheenRoughnessGroupIndices : unsigned char
    {
        PACKED_SHEEN_ROUGHNESS = 0,
        PACKED_SPECULAR_TRANSMISSION = 1,
        PACKED_DISPERSION_SCALE = 2,
        PACKED_THIN_FILM = 3,
    };

    enum PackedAlphaOpacityGroupIndices : unsigned char
    {
        PACKED_ALPHA_OPACITY = 0,
        PACKED_THIN_FILM_HUE_SHIFT = 1,
        PACKED_DIELECTRIC_PRIORITY = 0,
        PACKED_ENERGY_PRESERVATION_SAMPLES = 1,
    };

    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r)
            || !hippt::is_zero(emission.g)
            || !hippt::is_zero(emission.b)
            || get_emissive_texture_used();
    }

    /**
     * This function packs an UnpackedEffectiveMaterial into its packed version.
     * 
     * This is used in the shaders when a material is read after hitting some geometry: 
     * the texture of the material will be evaluated, transforming a 
     * DeviceUnpackedTexturedMaterial into a DeviceUnpackedEffectiveMaterial.
     * 
     * That DeviceUnpackedEffectiveMaterial will then be packed (using the pack() function below)
     * before being written to the G-buffer
     */
    HIPRT_HOST_DEVICE static DevicePackedEffectiveMaterial pack(const DeviceUnpackedEffectiveMaterial& unpacked)
    {
        DevicePackedEffectiveMaterial packed;

        packed.set_emission(unpacked.emission);
        packed.set_emissive_texture_used(unpacked.emissive_texture_used);

        packed.set_base_color(unpacked.base_color);

        packed.set_roughness(unpacked.roughness);
        packed.set_oren_nayar_sigma(unpacked.oren_nayar_sigma);

        packed.set_metallic(unpacked.metallic);
        packed.set_metallic_F90_falloff_exponent(unpacked.metallic_F90_falloff_exponent);
        packed.set_metallic_F82(unpacked.metallic_F82);
        packed.set_metallic_F90(unpacked.metallic_F90);
        packed.set_anisotropy(unpacked.anisotropy);
        packed.set_anisotropy_rotation(unpacked.anisotropy_rotation);
        packed.set_second_roughness_weight(unpacked.second_roughness_weight);
        packed.set_second_roughness(unpacked.second_roughness);
        packed.set_metallic_energy_compensation(unpacked.do_metallic_energy_compensation);

        packed.set_specular(unpacked.specular);
        packed.set_specular_tint(unpacked.specular_tint);
        packed.set_specular_color(unpacked.specular_color);
        packed.set_specular_darkening(unpacked.specular_darkening);
        packed.set_specular_energy_compensation(unpacked.do_specular_energy_compensation);

        packed.set_coat(unpacked.coat);
        packed.set_coat_medium_absorption(unpacked.coat_medium_absorption);
        packed.set_coat_medium_thickness(unpacked.coat_medium_thickness);
        packed.set_coat_roughness(unpacked.coat_roughness);
        packed.set_coat_roughening(unpacked.coat_roughening);
        packed.set_coat_darkening(unpacked.coat_darkening);
        packed.set_coat_anisotropy(unpacked.coat_anisotropy);
        packed.set_coat_anisotropy_rotation(unpacked.coat_anisotropy_rotation);
        packed.set_coat_ior(unpacked.coat_ior);
        packed.set_coat_energy_compensation(unpacked.do_coat_energy_compensation);

        packed.set_sheen(unpacked.sheen);
        packed.set_sheen_roughness(unpacked.sheen_roughness);
        packed.set_sheen_color(unpacked.sheen_color);

        packed.set_ior(unpacked.ior);
        packed.set_specular_transmission(unpacked.specular_transmission);
        packed.set_diffuse_transmission(unpacked.diffuse_transmission);

        packed.set_absorption_at_distance(unpacked.absorption_at_distance);
        packed.set_absorption_color(unpacked.absorption_color);
        packed.set_dispersion_scale(unpacked.dispersion_scale);
        packed.set_dispersion_abbe_number(unpacked.dispersion_abbe_number);
        packed.set_thin_walled(unpacked.thin_walled);
        packed.set_glass_energy_compensation(unpacked.do_glass_energy_compensation);

        packed.set_thin_film(unpacked.thin_film);
        packed.set_thin_film_ior(unpacked.thin_film_ior);
        packed.set_thin_film_thickness(unpacked.thin_film_thickness);
        packed.set_thin_film_kappa_3(unpacked.thin_film_kappa_3);
        packed.set_thin_film_hue_shift_degrees(unpacked.thin_film_hue_shift_degrees);
        packed.set_thin_film_base_ior_override(unpacked.thin_film_base_ior_override);
        packed.set_thin_film_do_ior_override(unpacked.thin_film_do_ior_override);

        packed.set_alpha_opacity(unpacked.alpha_opacity);
        packed.set_dielectric_priority(unpacked.get_dielectric_priority());
        packed.set_energy_preservation_monte_carlo_samples(unpacked.energy_preservation_monte_carlo_samples);
        packed.set_enforce_strong_energy_conservation(unpacked.enforce_strong_energy_conservation);

        return packed;
    }

    HIPRT_HOST_DEVICE DeviceUnpackedEffectiveMaterial unpack() const
    {
        DeviceUnpackedEffectiveMaterial unpacked;

        unpacked.emission = this->get_emission();
        unpacked.emissive_texture_used = this->get_emissive_texture_used();

        unpacked.base_color = this->get_base_color();

        unpacked.roughness = this->get_roughness();
        unpacked.oren_nayar_sigma = this->get_oren_nayar_sigma();

        unpacked.metallic = this->get_metallic();
        unpacked.metallic_F90_falloff_exponent = this->get_metallic_F90_falloff_exponent();
        unpacked.metallic_F82 = this->get_metallic_F82();
        unpacked.metallic_F90 = this->get_metallic_F90();
        unpacked.anisotropy = this->get_anisotropy();
        unpacked.anisotropy_rotation = this->get_anisotropy_rotation();
        unpacked.second_roughness_weight = this->get_second_roughness_weight();
        unpacked.second_roughness = this->get_second_roughness();
        unpacked.do_metallic_energy_compensation = this->get_do_metallic_energy_compensation();

        unpacked.specular = this->get_specular();
        unpacked.specular_tint = this->get_specular_tint();
        unpacked.specular_color = this->get_specular_color();
        unpacked.specular_darkening = this->get_specular_darkening();
        unpacked.do_specular_energy_compensation = this->get_do_specular_energy_compensation();

        unpacked.coat = this->get_coat();
        unpacked.coat_medium_absorption = this->get_coat_medium_absorption();
        unpacked.coat_medium_thickness = this->get_coat_medium_thickness();
        unpacked.coat_roughness = this->get_coat_roughness();
        unpacked.coat_roughening = this->get_coat_roughening();
        unpacked.coat_darkening = this->get_coat_darkening();
        unpacked.coat_anisotropy = this->get_coat_anisotropy();
        unpacked.coat_anisotropy_rotation = this->get_coat_anisotropy_rotation();
        unpacked.coat_ior = this->get_coat_ior();
        unpacked.do_coat_energy_compensation = this->get_do_coat_energy_compensation();

        unpacked.sheen = this->get_sheen();
        unpacked.sheen_roughness = this->get_sheen_roughness();
        unpacked.sheen_color = this->get_sheen_color();

        unpacked.ior = this->get_ior();
        unpacked.specular_transmission = this->get_specular_transmission();
        unpacked.diffuse_transmission = this->get_diffuse_transmission();

        unpacked.absorption_at_distance = this->get_absorption_at_distance();
        unpacked.absorption_color = this->get_absorption_color();
        unpacked.dispersion_scale = this->get_dispersion_scale();
        unpacked.dispersion_abbe_number = this->get_dispersion_abbe_number();
        unpacked.thin_walled = this->get_thin_walled();
        unpacked.do_glass_energy_compensation = this->get_do_glass_energy_compensation();

        unpacked.thin_film = this->get_thin_film();
        unpacked.thin_film_ior = this->get_thin_film_ior();
        unpacked.thin_film_thickness = this->get_thin_film_thickness();
        unpacked.thin_film_kappa_3 = this->get_thin_film_kappa_3();
        unpacked.thin_film_hue_shift_degrees = this->get_thin_film_hue_shift_degrees();
        unpacked.thin_film_base_ior_override = this->get_thin_film_base_ior_override();
        unpacked.thin_film_do_ior_override = this->get_thin_film_do_ior_override();

        unpacked.alpha_opacity = this->get_alpha_opacity();
        unpacked.set_dielectric_priority(this->get_dielectric_priority());
        unpacked.energy_preservation_monte_carlo_samples = this->get_energy_preservation_monte_carlo_samples();
        unpacked.enforce_strong_energy_conservation = this->get_enforce_strong_energy_conservation();

        return unpacked;
    }

    HIPRT_HOST_DEVICE ColorRGB32F get_emission() const { return this->emission; }
    HIPRT_HOST_DEVICE bool get_emissive_texture_used() const { return flags.get_bool<PackedFlagsIndices::PACKED_EMISSIVE_TEXTURE_USED>(); }

    HIPRT_HOST_DEVICE ColorRGB32F get_base_color() const { return base_color_roughness.get_color(); }

    HIPRT_HOST_DEVICE float get_roughness() const { return base_color_roughness.get_float(); }
    HIPRT_HOST_DEVICE float get_oren_nayar_sigma() const { return this->oren_nayar_sigma; }

    HIPRT_HOST_DEVICE float get_metallic() const { return metallic_F90_and_metallic.get_float(); }
    HIPRT_HOST_DEVICE float get_metallic_F90_falloff_exponent() const { return this->metallic_F90_falloff_exponent; }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F82() const { return metallic_F82_packed_and_diffuse_transmission.get_color(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_metallic_F90() const { return metallic_F90_and_metallic.get_color(); }
    HIPRT_HOST_DEVICE float get_anisotropy() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY>(); }
    HIPRT_HOST_DEVICE float get_anisotropy_rotation() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY_ROTATION>(); }
    HIPRT_HOST_DEVICE float get_second_roughness_weight() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS_WEIGHT>(); }
    HIPRT_HOST_DEVICE float get_second_roughness() const { return anisotropy_and_rotation_and_second_roughness.get_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE bool get_do_metallic_energy_compensation() const { return flags.get_bool<PackedFlagsIndices::METALLIC_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_specular() const { return specular_and_darkening_and_coat_roughness.get_float<PackedSpecularGroupIndices::PACKED_SPECULAR>(); }
    HIPRT_HOST_DEVICE float get_specular_tint() const { return specular_color_and_tint_factor.get_float(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_specular_color() const { return specular_color_and_tint_factor.get_color(); }
    HIPRT_HOST_DEVICE float get_specular_darkening() const { return specular_and_darkening_and_coat_roughness.get_float<PackedSpecularGroupIndices::PACKED_SPECULAR_DARKENING>(); }
    HIPRT_HOST_DEVICE bool get_do_specular_energy_compensation() const { return flags.get_bool<PackedFlagsIndices::SPECULAR_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_coat() const { return coat_and_medium_absorption.get_float(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_coat_medium_absorption() const { return coat_and_medium_absorption.get_color(); }
    HIPRT_HOST_DEVICE float get_coat_medium_thickness() const { return this->coat_medium_thickness; }
    HIPRT_HOST_DEVICE float get_coat_roughness() const { return specular_and_darkening_and_coat_roughness.get_float<PackedSpecularGroupIndices::PACKED_COAT_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE float get_coat_roughening() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_ROUGHENING>(); }
    HIPRT_HOST_DEVICE float get_coat_darkening() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_DARKENING>(); }
    HIPRT_HOST_DEVICE float get_coat_anisotropy() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY>(); }
    HIPRT_HOST_DEVICE float get_coat_anisotropy_rotation() const { return coat_roughening_darkening_anisotropy_and_rotation.get_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY_ROTATION>(); }
    HIPRT_HOST_DEVICE float get_coat_ior() const { return this->coat_ior; }
    HIPRT_HOST_DEVICE bool get_do_coat_energy_compensation() const { return flags.get_bool<PackedFlagsIndices::CLEARCOAT_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_sheen() const { return sheen_and_color.get_float(); }
    HIPRT_HOST_DEVICE float get_sheen_roughness() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_SHEEN_ROUGHNESS>(); }
    HIPRT_HOST_DEVICE ColorRGB32F get_sheen_color() const { return sheen_and_color.get_color(); }

    HIPRT_HOST_DEVICE float get_ior() const { return this->ior; }
    HIPRT_HOST_DEVICE float get_specular_transmission() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_SPECULAR_TRANSMISSION>(); }
    HIPRT_HOST_DEVICE float get_diffuse_transmission() const { return metallic_F82_packed_and_diffuse_transmission.get_float(); }
    HIPRT_HOST_DEVICE float get_absorption_at_distance() const { return this->absorption_at_distance; }
    HIPRT_HOST_DEVICE ColorRGB32F get_absorption_color() const { return absorption_color_packed.get_color(); }
    HIPRT_HOST_DEVICE float get_dispersion_scale() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_DISPERSION_SCALE>(); }
    HIPRT_HOST_DEVICE float get_dispersion_abbe_number() const { return this->dispersion_abbe_number; }
    HIPRT_HOST_DEVICE bool get_thin_walled() const { return flags.get_bool<PackedFlagsIndices::PACKED_THIN_WALLED >(); }
    HIPRT_HOST_DEVICE bool get_do_glass_energy_compensation() const { return flags.get_bool<PackedFlagsIndices::GLASS_ENERGY_COMPENSATION>(); }

    HIPRT_HOST_DEVICE float get_thin_film() const { return sheen_roughness_transmission_dispersion_thin_film.get_float<PackedSheenRoughnessGroupIndices::PACKED_THIN_FILM>(); }
    HIPRT_HOST_DEVICE float get_thin_film_ior() const { return this->thin_film_ior; }
    HIPRT_HOST_DEVICE float get_thin_film_thickness() const { return this->thin_film_thickness; }
    HIPRT_HOST_DEVICE float get_thin_film_kappa_3() const { return this->thin_film_kappa_3; }
    HIPRT_HOST_DEVICE float get_thin_film_hue_shift_degrees() const { return alpha_thin_film_hue_dielectric_priority.get_float<PackedAlphaOpacityGroupIndices::PACKED_THIN_FILM_HUE_SHIFT>(); }
    HIPRT_HOST_DEVICE float get_thin_film_base_ior_override() const { return this->thin_film_base_ior_override; }
    HIPRT_HOST_DEVICE bool get_thin_film_do_ior_override() const { return flags.get_bool<PackedFlagsIndices::PACKED_THIN_FILM_DO_IOR_OVERRIDE>(); }

    HIPRT_HOST_DEVICE float get_alpha_opacity() const { return alpha_thin_film_hue_dielectric_priority.get_float<PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(); }
    HIPRT_HOST_DEVICE unsigned char get_dielectric_priority() const { return alpha_thin_film_hue_dielectric_priority.get_uchar<PackedAlphaOpacityGroupIndices::PACKED_DIELECTRIC_PRIORITY>(); }

    HIPRT_HOST_DEVICE unsigned char get_energy_preservation_monte_carlo_samples() const { return alpha_thin_film_hue_dielectric_priority.get_uchar<PackedAlphaOpacityGroupIndices::PACKED_ENERGY_PRESERVATION_SAMPLES>(); }
    HIPRT_HOST_DEVICE bool get_enforce_strong_energy_conservation() const { return flags.get_bool<PackedFlagsIndices::PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION>(); }




    HIPRT_HOST_DEVICE void set_emission(ColorRGB32F emission_) { this->emission = emission_; }
    HIPRT_HOST_DEVICE void set_emissive_texture_used(bool emissive_texture_used) { flags.set_bool<PackedFlagsIndices::PACKED_EMISSIVE_TEXTURE_USED>(emissive_texture_used); }

    HIPRT_HOST_DEVICE void set_base_color(ColorRGB32F base_color) { base_color_roughness.set_color(base_color); }

    HIPRT_HOST_DEVICE void set_roughness(float roughness) { base_color_roughness.set_float(roughness); }
    HIPRT_HOST_DEVICE void set_oren_nayar_sigma(float oren_nayar_sigma_) { this->oren_nayar_sigma = oren_nayar_sigma_; }

    HIPRT_HOST_DEVICE void set_metallic(float metallic) { metallic_F90_and_metallic.set_float(metallic); }
    HIPRT_HOST_DEVICE void set_metallic_F90_falloff_exponent(float metallic_F90_falloff_exponent_) { this->metallic_F90_falloff_exponent = metallic_F90_falloff_exponent_; }
    HIPRT_HOST_DEVICE void set_metallic_F82(ColorRGB32F metallic_F82) { metallic_F82_packed_and_diffuse_transmission.set_color(metallic_F82); }
    HIPRT_HOST_DEVICE void set_metallic_F90(ColorRGB32F metallic_F90) { metallic_F90_and_metallic.set_color(metallic_F90); }
    HIPRT_HOST_DEVICE void set_anisotropy(float anisotropy) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY>(anisotropy); }
    HIPRT_HOST_DEVICE void set_anisotropy_rotation(float anisotropy_rotation) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_ANISOTROPY_ROTATION>(anisotropy_rotation); }
    HIPRT_HOST_DEVICE void set_second_roughness_weight(float second_roughness_weight) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS_WEIGHT>(second_roughness_weight); }
    HIPRT_HOST_DEVICE void set_second_roughness(float second_roughness) { anisotropy_and_rotation_and_second_roughness.set_float<PackedAnisotropyGroupIndices::PACKED_SECOND_ROUGHNESS>(second_roughness); }
    HIPRT_HOST_DEVICE void set_metallic_energy_compensation(bool do_metallic_energy_compensation) { flags.set_bool<PackedFlagsIndices::METALLIC_ENERGY_COMPENSATION>(do_metallic_energy_compensation); }

    HIPRT_HOST_DEVICE void set_specular(float specular) { specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_SPECULAR>(specular); }
    HIPRT_HOST_DEVICE void set_specular_tint(float specular_tint) { specular_color_and_tint_factor.set_float(specular_tint); }
    HIPRT_HOST_DEVICE void set_specular_color(ColorRGB32F specular_color) { specular_color_and_tint_factor.set_color(specular_color); }
    HIPRT_HOST_DEVICE void set_specular_darkening(float specular_darkening) { specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_SPECULAR_DARKENING>(specular_darkening); }
    HIPRT_HOST_DEVICE void set_specular_energy_compensation(bool do_specular_energy_compensation) { flags.set_bool<PackedFlagsIndices::SPECULAR_ENERGY_COMPENSATION>(do_specular_energy_compensation); }

    HIPRT_HOST_DEVICE void set_coat(float coat) { coat_and_medium_absorption.set_float(coat); }
    HIPRT_HOST_DEVICE void set_coat_medium_absorption(ColorRGB32F coat_medium_absorption) { coat_and_medium_absorption.set_color(coat_medium_absorption); }
    HIPRT_HOST_DEVICE void set_coat_medium_thickness(float coat_medium_thickness_) { this->coat_medium_thickness = coat_medium_thickness_; }
    HIPRT_HOST_DEVICE void set_coat_roughness(float coat_roughness) { specular_and_darkening_and_coat_roughness.set_float<PackedSpecularGroupIndices::PACKED_COAT_ROUGHNESS>(coat_roughness); }
    HIPRT_HOST_DEVICE void set_coat_roughening(float coat_roughening) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_ROUGHENING>(coat_roughening); }
    HIPRT_HOST_DEVICE void set_coat_darkening(float coat_darkening) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_DARKENING>(coat_darkening); }
    HIPRT_HOST_DEVICE void set_coat_anisotropy(float coat_anisotropy) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY>(coat_anisotropy); }
    HIPRT_HOST_DEVICE void set_coat_anisotropy_rotation(float coat_anisotropy_rotation) { coat_roughening_darkening_anisotropy_and_rotation.set_float<PackedCoatGroupIndices::PACKED_COAT_ANISOTROPY_ROTATION>(coat_anisotropy_rotation); }
    HIPRT_HOST_DEVICE void set_coat_ior(float coat_ior_) { this->coat_ior = coat_ior_; }
    HIPRT_HOST_DEVICE void set_coat_energy_compensation(bool do_coat_energy_compensation) { flags.set_bool<PackedFlagsIndices::CLEARCOAT_ENERGY_COMPENSATION>(do_coat_energy_compensation); }

    HIPRT_HOST_DEVICE void set_sheen(float sheen) { sheen_and_color.set_float(sheen); }
    HIPRT_HOST_DEVICE void set_sheen_roughness(float sheen_roughness) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_SHEEN_ROUGHNESS>(sheen_roughness); }
    HIPRT_HOST_DEVICE void set_sheen_color(ColorRGB32F sheen_color) { sheen_and_color.set_color(sheen_color); }

    HIPRT_HOST_DEVICE void set_ior(float ior_) { this->ior = ior_; }
    HIPRT_HOST_DEVICE void set_specular_transmission(float specular_transmission) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_SPECULAR_TRANSMISSION>(specular_transmission); }
    HIPRT_HOST_DEVICE void set_diffuse_transmission(float diffuse_transmission) { metallic_F82_packed_and_diffuse_transmission.set_float(diffuse_transmission); }
    HIPRT_HOST_DEVICE void set_absorption_at_distance(float absorption_at_distance_) { this->absorption_at_distance = absorption_at_distance_; }
    HIPRT_HOST_DEVICE void set_absorption_color(ColorRGB32F absorption_color) { absorption_color_packed.set_color(absorption_color); }
    HIPRT_HOST_DEVICE void set_dispersion_scale(float dispersion_scale) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_DISPERSION_SCALE>(dispersion_scale); }
    HIPRT_HOST_DEVICE void set_dispersion_abbe_number(float dispersion_abbe_number_) { this->dispersion_abbe_number = dispersion_abbe_number_; }
    HIPRT_HOST_DEVICE void set_thin_walled(bool thin_walled) { flags.set_bool<PackedFlagsIndices::PACKED_THIN_WALLED >(thin_walled); }
    HIPRT_HOST_DEVICE void set_glass_energy_compensation(bool do_glass_energy_compensation) { flags.set_bool<PackedFlagsIndices::GLASS_ENERGY_COMPENSATION>(do_glass_energy_compensation); }

    HIPRT_HOST_DEVICE void set_thin_film(float thin_film) { sheen_roughness_transmission_dispersion_thin_film.set_float<PackedSheenRoughnessGroupIndices::PACKED_THIN_FILM>(thin_film); }
    HIPRT_HOST_DEVICE void set_thin_film_ior(float thin_film_ior_) { this->thin_film_ior = thin_film_ior_; }
    HIPRT_HOST_DEVICE void set_thin_film_thickness(float thin_film_thickness_) { this->thin_film_thickness = thin_film_thickness_; }
    HIPRT_HOST_DEVICE void set_thin_film_kappa_3(float thin_film_kappa_3_) { this->thin_film_kappa_3 = thin_film_kappa_3_; }
    HIPRT_HOST_DEVICE void set_thin_film_hue_shift_degrees(float thin_film_hue_shift_degrees) { alpha_thin_film_hue_dielectric_priority.set_float<PackedAlphaOpacityGroupIndices::PACKED_THIN_FILM_HUE_SHIFT>(thin_film_hue_shift_degrees); }
    HIPRT_HOST_DEVICE void set_thin_film_base_ior_override(bool thin_film_base_ior_override_) { this->thin_film_base_ior_override = thin_film_base_ior_override_; }
    HIPRT_HOST_DEVICE void set_thin_film_do_ior_override(bool thin_film_do_ior_override) { flags.set_bool<PackedFlagsIndices::PACKED_THIN_FILM_DO_IOR_OVERRIDE>(thin_film_do_ior_override); }

    HIPRT_HOST_DEVICE void set_alpha_opacity(float alpha_opacity) { alpha_thin_film_hue_dielectric_priority.set_float<PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(alpha_opacity); }
    HIPRT_HOST_DEVICE void set_dielectric_priority(unsigned char dielectric_priority) { alpha_thin_film_hue_dielectric_priority.set_uchar<PackedAlphaOpacityGroupIndices::PACKED_ALPHA_OPACITY>(dielectric_priority); }

    HIPRT_HOST_DEVICE void set_energy_preservation_monte_carlo_samples(unsigned char energy_preservation_monte_carlo_samples) { alpha_thin_film_hue_dielectric_priority.set_uchar<PackedAlphaOpacityGroupIndices::PACKED_ENERGY_PRESERVATION_SAMPLES>(energy_preservation_monte_carlo_samples); }
    HIPRT_HOST_DEVICE void set_enforce_strong_energy_conservation(bool enforce_strong_energy_conservation) { flags.set_bool<PackedFlagsIndices::PACKED_ENFORCE_STRONG_ENERGY_CONSERVATION>(enforce_strong_energy_conservation); }

private:
    friend class DevicePackedTexturedMaterialSoAGPUData;
    friend class DevicePackedTexturedMaterialSoACPUData;

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
    UChar8BoolsPacked flags;

    // Full range emission
    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };

    // Base color RGB 3x8 bits + roughness uchar [float in [0,1] packed in 8 bit]
    ColorRGB24bFloat0_1Packed base_color_roughness;

    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian

    // Parameters for Adobe 2023 F82-tint model
    // Packs the SDR F90 color and the metalness parameter
    ColorRGB24bFloat0_1Packed metallic_F90_and_metallic;
    ColorRGB24bFloat0_1Packed metallic_F82_packed_and_diffuse_transmission;
    float metallic_F90_falloff_exponent = 5.0f;

    Float4xPacked anisotropy_and_rotation_and_second_roughness;

    // Packed specular color and the intensity of the tint
    // 
    // Specular tint intensity: Specular will be white if 0.0f and will be 'specular_color' if 1.0f
    ColorRGB24bFloat0_1Packed specular_color_and_tint_factor;

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
    Float4xPacked specular_and_darkening_and_coat_roughness;
    float coat_medium_thickness = 5.0f;

    // Packed:
    //  - Coat
    //      Intensity of the coat. 0.0f disables the coating
    //
    //  - Coat medium absorption color 
    ColorRGB24bFloat0_1Packed coat_and_medium_absorption;

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
    Float4xPacked coat_roughening_darkening_anisotropy_and_rotation;
    float coat_ior = 1.5f;

    // Packed:
    //  - Sheen intensity. 0.0f disables the sheen effect
    //
    //  - Sheen color
    ColorRGB24bFloat0_1Packed sheen_and_color;

    // IOR of the base material
    float ior = 1.40f;

    // Packed:
    //  - Absorption color
    //      Color of the light absorption when traveling through the medium
    // TODO: PACKED FLOAT IS UNUSED IN HERE
    ColorRGB24bFloat0_1Packed absorption_color_packed;
    float absorption_at_distance = 5.0f;

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
    Float4xPacked sheen_roughness_transmission_dispersion_thin_film;

    float dispersion_abbe_number = 20.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_base_ior_override = 1.0f;

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
    Float2xUChar2xPacked alpha_thin_film_hue_dielectric_priority;
};

struct DevicePackedTexturedMaterial : public DevicePackedEffectiveMaterial
{
    enum NormalMapEmissionIndices : unsigned char
    {
        NORMAL_MAP_INDEX = 0,
        EMISSION_INDEX = 1,
    };

    enum BaseColorRoughnessMetallicIndices : unsigned char
    {
        BASE_COLOR_INDEX = 0,
        ROUGHNESS_METALLIC_INDEX = 1,
    };

    enum RoughnessAndMetallicIndices : unsigned char
    {
        ROUGHNESS_INDEX = 0,
        METALLIC_INDEX = 1,
    };

    enum AnisotropicSpecularIndices : unsigned char
    {
        ANISOTROPIC_INDEX = 0,
        SPECULAR_INDEX = 1,
    };

    enum CoatSheenIndices : unsigned char
    {
        COAT_INDEX = 0,
        SHEEN_INDEX = 1,
    };

    enum SpecularTransmissionIndex : unsigned char
    {
        SPECULAR_TRANSMISSION_INDEX = 0,
    };

    HIPRT_HOST_DEVICE DeviceUnpackedTexturedMaterial unpack()
    {
        DeviceUnpackedTexturedMaterial out;

        out.normal_map_texture_index = this->get_normal_map_texture_index();
        out.emission_texture_index = this->get_emission_texture_index();
        out.base_color_texture_index = this->get_base_color_texture_index();

        out.roughness_metallic_texture_index = this->get_roughness_metallic_texture_index();
        out.roughness_texture_index = this->get_roughness_texture_index();
        out.metallic_texture_index = this->get_metallic_texture_index();
        out.anisotropic_texture_index = this->get_anisotropic_texture_index();

        out.specular_texture_index = this->get_specular_texture_index();
        out.coat_texture_index = this->get_coat_texture_index();
        out.sheen_texture_index = this->get_sheen_texture_index();
        out.specular_transmission_texture_index = this->get_specular_transmission_texture_index();






        out.emissive_texture_used = this->get_emissive_texture_used();
        if (!out.emissive_texture_used)
            out.emission = this->get_emission();

        if (out.base_color_texture_index == MaterialUtils::NO_TEXTURE)
            out.base_color = this->get_base_color();

        out.roughness = this->get_roughness();
        out.oren_nayar_sigma = this->get_oren_nayar_sigma();

        // Parameters for Adobe 2023 F82-tint model
        out.metallic = this->get_metallic();
        if (out.metallic > 0.0f || out.metallic_texture_index != MaterialUtils::NO_TEXTURE || out.roughness_metallic_texture_index != MaterialUtils::NO_TEXTURE)
        {
            // We only need to unpack all of this if we actually have a metallic lobe

            out.metallic_F90_falloff_exponent = this->get_metallic_F90_falloff_exponent();
            // F0 is not here as it uses the 'base_color' of the material
            out.metallic_F82 = this->get_metallic_F82();
            out.metallic_F90 = this->get_metallic_F90();

            out.second_roughness_weight = this->get_second_roughness_weight();
            out.second_roughness = this->get_second_roughness();

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoMetallicEnergyCompensation == KERNEL_OPTION_TRUE
            out.do_metallic_energy_compensation = this->get_do_metallic_energy_compensation();
#endif
        }

        out.anisotropy = this->get_anisotropy();
        out.anisotropy_rotation = this->get_anisotropy_rotation();

        // Specular intensity
        out.specular = this->get_specular();
        if (out.specular > 0.0f || out.specular_texture_index != MaterialUtils::NO_TEXTURE)
        {
            // Specular tint intensity. 
            // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
            out.specular_tint = this->get_specular_tint();
            out.specular_color = this->get_specular_color();
            // Same as coat darkening but for total internal reflection inside the specular layer
            // that sits on top of the diffuse base
            //
            // Disabled by default for artistic "expectations"
            out.specular_darkening = this->get_specular_darkening();

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoSpecularEnergyCompensation == KERNEL_OPTION_TRUE
            out.do_specular_energy_compensation = this->get_do_specular_energy_compensation();
#endif
        }

        out.coat = this->get_coat();
        if (out.coat > 0.0f || out.coat_texture_index != MaterialUtils::NO_TEXTURE)
        {
            out.coat_medium_absorption = this->get_coat_medium_absorption();
            // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
            // that will happen inside the coat
            out.coat_medium_thickness = this->get_coat_medium_thickness();
            out.coat_roughness = this->get_coat_roughness();
            // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
            // i.e. the specular/metallic/transmission layers.
            // 
            // The option is however given here to artistically disable
            // that behavior by using coat roughening = 0.0f.
            out.coat_roughening = this->get_coat_roughening();
            // Because of the total internal reflection that can happen inside the coat layer (i.e.
            // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
            // clearcoat will appear will increased saturation.
            out.coat_darkening = this->get_coat_darkening();
            out.coat_anisotropy = this->get_coat_anisotropy();
            out.coat_anisotropy_rotation = this->get_coat_anisotropy_rotation();
            out.coat_ior = this->get_coat_ior();

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoClearcoatEnergyCompensation == KERNEL_OPTION_TRUE
            out.do_coat_energy_compensation = this->get_do_coat_energy_compensation();
#endif
        }

        out.sheen = this->get_sheen(); // Sheen strength
        if (out.sheen > 0.0f || out.sheen_texture_index != MaterialUtils::NO_TEXTURE)
        {
            out.sheen_roughness = this->get_sheen_roughness();
            out.sheen_color = this->get_sheen_color();
        }

        out.ior = this->get_ior();
        out.diffuse_transmission = this->get_diffuse_transmission();
        out.specular_transmission = this->get_specular_transmission();

        if (out.specular_transmission > 0.0f || out.specular_transmission_texture_index != MaterialUtils::NO_TEXTURE)
        {
            // Specular transmission specific 
            out.dispersion_scale = this->get_dispersion_scale();
            out.dispersion_abbe_number = this->get_dispersion_abbe_number();
            out.thin_walled = this->get_thin_walled();

#if PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE && PrincipledBSDFDoGlassEnergyCompensation == KERNEL_OPTION_TRUE
            out.do_glass_energy_compensation = this->get_do_glass_energy_compensation();
#endif
        }
        if (out.specular_transmission > 0.0f || out.diffuse_transmission > 0.0f || out.specular_transmission_texture_index != MaterialUtils::NO_TEXTURE)
        {
            // Also enabled by diffuse transmission as well as specular transmission
            
            // At what distance is the light absorbed to the given absorption_color
            out.absorption_at_distance = this->get_absorption_at_distance();
            // Color of the light absorption when traveling through the medium
            out.absorption_color = this->get_absorption_color();
        }

        out.thin_film = this->get_thin_film();
        if (out.thin_film > 0.0f)
        {
            out.thin_film_ior = this->get_thin_film_ior();
            out.thin_film_thickness = this->get_thin_film_thickness();
            out.thin_film_kappa_3 = this->get_thin_film_kappa_3();
            // Sending the hue film in [0, 1] to the GPU
            out.thin_film_hue_shift_degrees = this->get_thin_film_hue_shift_degrees();
            out.thin_film_base_ior_override = this->get_thin_film_base_ior_override();
            out.thin_film_do_ior_override = this->get_thin_film_do_ior_override();
        }

        // 1.0f makes the material completely opaque
        // 0.0f completely transparent (becomes invisible)
        out.alpha_opacity = this->get_alpha_opacity();

        // Nested dielectric parameter
        out.set_dielectric_priority(this->get_dielectric_priority());

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
        out.enforce_strong_energy_conservation = this->get_enforce_strong_energy_conservation();
        if (out.enforce_strong_energy_conservation)
            out.energy_preservation_monte_carlo_samples = this->get_energy_preservation_monte_carlo_samples();

        return out;
    }

    HIPRT_HOST_DEVICE unsigned short int get_normal_map_texture_index() const { return normal_map_emission_index.get_value<NormalMapEmissionIndices::NORMAL_MAP_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_emission_texture_index() const { return normal_map_emission_index.get_value<NormalMapEmissionIndices::EMISSION_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_base_color_texture_index() const { return base_color_roughness_metallic_index.get_value<BaseColorRoughnessMetallicIndices::BASE_COLOR_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_metallic_texture_index() const { return base_color_roughness_metallic_index.get_value<BaseColorRoughnessMetallicIndices::ROUGHNESS_METALLIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_roughness_texture_index() const { return roughness_and_metallic_index.get_value<RoughnessAndMetallicIndices::ROUGHNESS_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_metallic_texture_index() const { return roughness_and_metallic_index.get_value<RoughnessAndMetallicIndices::METALLIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_anisotropic_texture_index() const { return roughness_and_metallic_index.get_value<AnisotropicSpecularIndices::ANISOTROPIC_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_specular_texture_index() const { return anisotropic_specular_index.get_value<AnisotropicSpecularIndices::SPECULAR_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_coat_texture_index() const { return coat_sheen_index.get_value<CoatSheenIndices::COAT_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_sheen_texture_index() const { return coat_sheen_index.get_value<CoatSheenIndices::SHEEN_INDEX>(); }
    HIPRT_HOST_DEVICE unsigned short int get_specular_transmission_texture_index() const { return specular_transmission_index.get_value<SpecularTransmissionIndex::SPECULAR_TRANSMISSION_INDEX>(); }

    HIPRT_HOST_DEVICE void set_normal_map_texture_index(unsigned short normal_map_index) { normal_map_emission_index.set_value<NormalMapEmissionIndices::NORMAL_MAP_INDEX>(normal_map_index); }
    HIPRT_HOST_DEVICE void set_emission_texture_index(unsigned short emission_index) { normal_map_emission_index.set_value<NormalMapEmissionIndices::EMISSION_INDEX>(emission_index); }
    HIPRT_HOST_DEVICE void set_base_color_texture_index(unsigned short base_color_index) { base_color_roughness_metallic_index.set_value<BaseColorRoughnessMetallicIndices::BASE_COLOR_INDEX>(base_color_index); }
    HIPRT_HOST_DEVICE void set_roughness_metallic_texture_index(unsigned short roughness_metallic_index) { base_color_roughness_metallic_index.set_value<BaseColorRoughnessMetallicIndices::ROUGHNESS_METALLIC_INDEX>(roughness_metallic_index); }
    HIPRT_HOST_DEVICE void set_roughness_texture_index(unsigned short roughness_index) { roughness_and_metallic_index.set_value<RoughnessAndMetallicIndices::ROUGHNESS_INDEX>(roughness_index); }
    HIPRT_HOST_DEVICE void set_metallic_texture_index(unsigned short metallic_index) { roughness_and_metallic_index.set_value<RoughnessAndMetallicIndices::METALLIC_INDEX>(metallic_index); }
    HIPRT_HOST_DEVICE void set_anisotropic_texture_index(unsigned short anisotropic_index) { roughness_and_metallic_index.set_value<AnisotropicSpecularIndices::ANISOTROPIC_INDEX>(anisotropic_index); }
    HIPRT_HOST_DEVICE void set_specular_texture_index(unsigned short specular_index) { anisotropic_specular_index.set_value<AnisotropicSpecularIndices::SPECULAR_INDEX>(specular_index); }
    HIPRT_HOST_DEVICE void set_coat_texture_index(unsigned short coat_index) { coat_sheen_index.set_value<CoatSheenIndices::COAT_INDEX>(coat_index); }
    HIPRT_HOST_DEVICE void set_sheen_texture_index(unsigned short sheen_index) { coat_sheen_index.set_value<CoatSheenIndices::SHEEN_INDEX>(sheen_index); }
    HIPRT_HOST_DEVICE void set_specular_transmission_texture_index(unsigned short _specular_transmission_index) { specular_transmission_index.set_value<SpecularTransmissionIndex::SPECULAR_TRANSMISSION_INDEX>(_specular_transmission_index); }

private:
    friend class DevicePackedTexturedMaterialSoAGPUData;
    friend class DevicePackedTexturedMaterialSoACPUData;

    Uint2xPacked normal_map_emission_index;
    // If the roughness_metallic texture index is not MaterialUtils::NO_TEXTURE, 
    // then there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness.
    Uint2xPacked base_color_roughness_metallic_index;
    Uint2xPacked roughness_and_metallic_index;
    Uint2xPacked anisotropic_specular_index;
    Uint2xPacked coat_sheen_index;
    // TODO: 1 PACKED UINT IS UNUSED IN HERE
    Uint2xPacked specular_transmission_index;
};

#endif
