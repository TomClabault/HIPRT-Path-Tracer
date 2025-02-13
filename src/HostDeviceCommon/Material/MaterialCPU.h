/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_CPU_H
#define HOST_DEVICE_COMMON_MATERIAL_CPU_H

#include "Device/includes/NestedDielectrics.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialPacked.h"
#include "HostDeviceCommon/Material/MaterialUtils.h"

 // Adding this guard to make sure that we never use the CPU materials in the GPU code
#ifndef  __KERNELCC__

// This material structure is only used on the CPU.
// The reason why we have a CPU and a GPU material is because
// we may want to precompute some properties on the CPU before sending them
// to the GPU. This means that the CPU material stores both the precomputed values
// and the values needed for the precomputation itself.
//
// But the GPU only cares about the precomputed values itself, not the ingredients
// to the precomputation so that's why we have separate structures
struct CPUMaterial
{
    /**
     * Function that transforms/packs this material to the version that the GPU is going to use
     */
    DevicePackedTexturedMaterial pack_to_GPU() const
    {
        DevicePackedTexturedMaterial mat;

        mat.set_normal_map_texture_index(this->normal_map_texture_index);
        mat.set_emission_texture_index(this->emission_texture_index);
        mat.set_base_color_texture_index(this->base_color_texture_index);

        mat.set_roughness_metallic_texture_index(this->roughness_metallic_texture_index);
        mat.set_roughness_texture_index(this->roughness_texture_index);
        mat.set_metallic_texture_index(this->metallic_texture_index);
        mat.set_anisotropic_texture_index(this->anisotropic_texture_index);

        mat.set_specular_texture_index(this->specular_texture_index);
        mat.set_coat_texture_index(this->coat_texture_index);
        mat.set_sheen_texture_index(this->sheen_texture_index);
        mat.set_specular_transmission_texture_index(this->specular_transmission_texture_index);






        mat.set_emission(emission * emission_strength * global_emissive_factor);
        mat.set_emissive_texture_used(emissive_texture_used);

        mat.set_base_color(base_color);

        mat.set_roughness(roughness);
        mat.set_oren_nayar_sigma(oren_nayar_sigma);

        // Parameters for Adobe 2023 F82-tint model
        mat.set_metallic(metallic);
        mat.set_metallic_F90_falloff_exponent(metallic_F90_falloff_exponent);
        // F0 is not here as it uses the 'base_color' of the material
        mat.set_metallic_F82(metallic_F82);
        mat.set_metallic_F90(metallic_F90);
        mat.set_anisotropy(anisotropy);
        mat.set_anisotropy_rotation(anisotropy_rotation);
        mat.set_second_roughness_weight(second_roughness_weight);
        mat.set_second_roughness(second_roughness);
        mat.set_metallic_energy_compensation(do_metallic_energy_compensation);

        // Specular intensity
        mat.set_specular(specular);
        // Specular tint intensity. 
        // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
        mat.set_specular_tint(specular_tint);
        mat.set_specular_color(specular_color);
        // Same as coat darkening but for total internal reflection inside the specular layer
        // that sits on top of the diffuse base
        //
        // Disabled by default for artistic "expectations"
        mat.set_specular_darkening(specular_darkening);
        mat.set_specular_energy_compensation(do_specular_energy_compensation);

        mat.set_coat(coat);
        mat.set_coat_medium_absorption(coat_medium_absorption);
        // The coat thickness influences the amount of absorption (given by 'coat_medium_absorption')
        // that will happen inside the coat
        mat.set_coat_medium_thickness(coat_medium_thickness);
        mat.set_coat_roughness(coat_roughness);
        // Physical accuracy requires that a rough clearcoat also roughens what's underneath it
        // i.e. the specular/metallic/transmission layers.
        // 
        // The option is however given here to artistically disable
        // that behavior by using coat roughening = 0.0f.
        mat.set_coat_roughening(coat_roughening);
        // Because of the total internal reflection that can happen inside the coat layer (i.e.
        // light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the
        // clearcoat will appear will increased saturation.
        mat.set_coat_darkening(coat_darkening);
        mat.set_coat_anisotropy(coat_anisotropy);
        mat.set_coat_anisotropy_rotation(coat_anisotropy_rotation);
        mat.set_coat_ior(coat_ior);
        mat.set_coat_energy_compensation(do_coat_energy_compensation);

        mat.set_sheen(sheen); // Sheen strength
        mat.set_sheen_roughness(sheen_roughness);
        mat.set_sheen_color(sheen_color);

        mat.set_ior(ior);
        mat.set_specular_transmission(specular_transmission);
        mat.set_diffuse_transmission(diffuse_transmission);

        // At what distance is the light absorbed to the given absorption_color
        mat.set_absorption_at_distance(absorption_at_distance);
        // Color of the light absorption when traveling through the medium
        mat.set_absorption_color(absorption_color);
        mat.set_dispersion_scale(dispersion_scale);
        mat.set_dispersion_abbe_number(dispersion_abbe_number);
        mat.set_thin_walled(thin_walled);
        mat.set_glass_energy_compensation(do_glass_energy_compensation);

        mat.set_thin_film(thin_film);
        mat.set_thin_film_ior(thin_film_ior);
        mat.set_thin_film_thickness(thin_film_thickness);
        mat.set_thin_film_kappa_3(thin_film_kappa_3);
        // Sending the hue film in [0, 1] to the GPU
        mat.set_thin_film_hue_shift_degrees(thin_film_hue_shift_degrees / 360.0f);
        mat.set_thin_film_base_ior_override(thin_film_base_ior_override);
        mat.set_thin_film_do_ior_override(thin_film_do_ior_override);

        // 1.0f makes the material completely opaque
        // 0.0f completely transparent (becomes invisible)
        mat.set_alpha_opacity(alpha_opacity);

        // Nested dielectric parameter
        mat.set_dielectric_priority(dielectric_priority);

        mat.set_energy_preservation_monte_carlo_samples(energy_preservation_monte_carlo_samples);
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
        mat.set_enforce_strong_energy_conservation(enforce_strong_energy_conservation);

        return mat;
    }

    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r)
            || !hippt::is_zero(emission.g)
            || !hippt::is_zero(emission.b)
            || emissive_texture_used;
    }

    /*
     * Clamps some of the parameters of the material to avoid edge cases like NaNs
     * during rendering (i.e. numerical instabilities)
     */
    HIPRT_HOST_DEVICE void make_safe()
    {
        // The values are going to be packed before being sent to the GPU
        // Packing limits the range of values (most of them in [0, 1] because
        // they are not expected to go higher) so we're clamping the values
        // to avoid out-of-range-packing 
        base_color.clamp(0.0f, 1.0f);

        metallic = hippt::clamp(0.0f, 1.0f, metallic);
        metallic_F82.clamp(0.0f, 1.0f);
        metallic_F90.clamp(0.0f, 1.0f);
        anisotropy = hippt::clamp(0.0f, 1.0f, anisotropy);
        anisotropy_rotation = hippt::clamp(0.0f, 1.0f, anisotropy_rotation);
        second_roughness_weight  = hippt::clamp(0.0f, 1.0f, second_roughness_weight);
        second_roughness = hippt::clamp(0.0f, 1.0f, second_roughness);

        specular = hippt::clamp(0.0f, 1.0f, specular);
        specular_tint = hippt::clamp(0.0f, 1.0f, specular_tint);
        specular_color.clamp(0.0f, 1.0f);
        specular_darkening = hippt::clamp(0.0f, 1.0f, specular_darkening);

        coat = hippt::clamp(0.0f, 1.0f, coat);
        coat_medium_absorption.clamp(0.0f, 1.0f);
        coat_roughness = hippt::clamp(MaterialUtils::ROUGHNESS_CLAMP, 1.0f, coat_roughness);
        coat_roughening = hippt::clamp(0.0f, 1.0f, coat_roughening);
        coat_darkening = hippt::clamp(0.0f, 1.0f, coat_darkening);
        coat_anisotropy = hippt::clamp(0.0f, 1.0f, coat_anisotropy);
        coat_anisotropy_rotation = hippt::clamp(0.0f, 1.0f, coat_anisotropy_rotation);
        
        sheen = hippt::clamp(0.0f, 1.0f, sheen);
        sheen_roughness = hippt::clamp(MaterialUtils::ROUGHNESS_CLAMP, 1.0f, sheen_roughness);
        sheen_color.clamp(0.0f, 1.0f);

        specular_transmission = hippt::clamp(0.0f, 1.0f, specular_transmission);
        // Avoiding zero
        absorption_at_distance = hippt::max(absorption_at_distance, 1.0e-4f);
        absorption_color = ColorRGB32F::max(absorption_color, ColorRGB32F(1.0f / 512.0f));
        absorption_color.clamp(0.0f, 1.0f);

        dispersion_abbe_number = hippt::max(1.0e-5f, dispersion_abbe_number);
        dispersion_scale = hippt::clamp(0.0f, 1.0f, dispersion_scale);

        thin_film = hippt::clamp(0.0f, 1.0f, thin_film);
        thin_film_hue_shift_degrees = hippt::clamp(0.0f, 360.0f, thin_film_hue_shift_degrees);
        thin_film_ior = hippt::max(1.0005f, thin_film_ior);

        alpha_opacity = hippt::clamp(0.0f, 1.0f, alpha_opacity);

        dielectric_priority = hippt::clamp(0, (int)StackPriorityEntry::PRIORITY_BIT_MASK >> StackPriorityEntry::PRIORITY_BIT_SHIFT, dielectric_priority);

        // Clamping to avoid negative emission
        emission = ColorRGB32F::max(ColorRGB32F(0.0f), emission);

        if (specular_transmission == 0.0f && diffuse_transmission == 0.0f)
            // No transmission means that we should never skip this boundary --> max priority
            dielectric_priority = (1 << StackPriorityEntry::PRIORITY_MAXIMUM) - 1;
    }

    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
    float emission_strength = 1.0f; // This factor is baked into 'emission' before being sent to the GPU
    float global_emissive_factor = 1.0f; // This factor is baked into 'emission' before being sent to the GPU
    bool emissive_texture_used = false;

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
    // Whether or not to do energy compensation of the metallic layer
    // for that material
    bool do_metallic_energy_compensation = true;

    // Specular intensity
    float specular = 1.0f;
    // Specular tint intensity. 
    // Specular will be white if 0.0f and will be 'specular_color' if 1.0f
    float specular_tint = 1.0f;
    ColorRGB32F specular_color = ColorRGB32F(1.0f);
    // Same as coat darkening but for total internal reflection inside the specular layer
    // that sits on top of the diffuse base
    // 
    // Disabled by default for "artistic expectations" but this is not physically accurate
    float specular_darkening = 0.0f;
    // Whether or not to do energy compensation of the specular/diffuse layer
    // for that material
    bool do_specular_energy_compensation = false;

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
    // Whether or not to do energy compensation of the clearcoat layer
    // for that material
    bool do_coat_energy_compensation = true;

    float sheen = 0.0f; // Sheen strength
    float sheen_roughness = 0.5f;
    ColorRGB32F sheen_color = ColorRGB32F(1.0f);

    float ior = 1.40f;
    float specular_transmission = 0.0f;
    float diffuse_transmission = 0.0f;
    // At what distance is the light absorbed to the given absorption_color
    float absorption_at_distance = 1.0f;
    // Color of the light absorption when traveling through the medium
    ColorRGB32F absorption_color = ColorRGB32F(1.0f);
    float dispersion_scale = 0.0f;
    float dispersion_abbe_number = 20.0f;
    bool thin_walled = false;
    // Whether or not to do energy compensation of the glass layer
    // for that material
    bool do_glass_energy_compensation = true;

    float thin_film = 0.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_hue_shift_degrees = 0.0f;
    float thin_film_base_ior_override = 1.0f;
    bool thin_film_do_ior_override = false;

    // 1.0f makes the material completely opaque
    // 0.0f completely transparent (becomes invisible)
    float alpha_opacity = 1.0f;

    // Nested dielectric parameter
    int dielectric_priority = 0;

    int energy_preservation_monte_carlo_samples = 12;
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

#endif // #ifndef  __KERNELCC__

#endif
