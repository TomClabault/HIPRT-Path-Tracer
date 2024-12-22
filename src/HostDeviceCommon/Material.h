/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_H
#define HOST_DEVICE_COMMON_MATERIAL_H

#include "Device/includes/NestedDielectrics.h"

#include "HostDeviceCommon/KernelOptions.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Math.h"

enum BRDF
{
    Uninitialized,
    Principled,
    SpecularFresnel
};

// A simplified material is the material effectively evaluated at a point in the scene.
// This means that it doesn't contain texture indices for example since the texture
// has been evaluated at the intersection point and the base color (if we're talking
// about the base color texture) stored directly in the base_color parameter of the
// simplified material.
// 
// This simplified material structure is used in the GBuffer for example
struct SimplifiedRendererMaterial
{
    static constexpr float ROUGHNESS_CLAMP = 1.0e-4f;

    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return !hippt::is_zero(emission.r * emission_strength) 
            || !hippt::is_zero(emission.g * emission_strength) 
            || !hippt::is_zero(emission.b * emission_strength) 
            || emissive_texture_used;
    }

    /*
     * Clamps some of the parameters of the material to avoid edge cases like NaNs
     * during rendering (i.e. numerical instabilities)
     */
    HIPRT_HOST_DEVICE void make_safe()
    {
        roughness = hippt::max(ROUGHNESS_CLAMP, roughness);
        coat_roughness = hippt::max(ROUGHNESS_CLAMP, coat_roughness);
        sheen_roughness = hippt::max(ROUGHNESS_CLAMP, sheen_roughness);

        // Clamping to avoid negative emission
        emission = ColorRGB32F::max(ColorRGB32F(0.0f), emission);
        emission_strength = hippt::max(0.0f, emission_strength);

        // Avoiding zero
        absorption_at_distance = hippt::max(absorption_at_distance, 1.0e-4f);
        absorption_color = ColorRGB32F::max(absorption_color, ColorRGB32F(1.0f / 512.0f));

        thin_film_ior = hippt::max(1.0005f, thin_film_ior);
    }

    /*
     * Some properties of the material can be precomputed
     * This function does it
     */
    HIPRT_HOST_DEVICE void precompute_properties()
    {
        if (specular_transmission == 0.0f)
            // No transmission means that we should never skip this boundary --> max priority
            dielectric_priority = (1 << StackPriorityEntry::PRIORITY_MAXIMUM) - 1;
    }

    HIPRT_HOST_DEVICE static void get_oren_nayar_AB(float sigma, float& out_oren_A, float& out_oren_B)
    {
        float sigma2 = sigma * sigma;
        out_oren_A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
        out_oren_B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    HIPRT_HOST_DEVICE static void get_alphas(float roughness, float anisotropy, float& out_alpha_x, float& out_alpha_y)
    {
        float aspect = sqrtf(1.0f - 0.9f * anisotropy);
        out_alpha_x = hippt::max(ROUGHNESS_CLAMP, roughness * roughness / aspect);
        out_alpha_y = hippt::max(ROUGHNESS_CLAMP, roughness * roughness * aspect);
    }

    HIPRT_HOST_DEVICE static float get_thin_walled_roughness(bool thin_walled, float base_roughness, float relative_eta)
    {
        if (!thin_walled)
            return base_roughness;

        /*
         * Roughness remapping so that a thin walled interface matches better a
         * properly modeled double interface model. Said otherwise: roughness remapping
         * so that the thin walled approximation matches the non thin walled physically correct equivalent
         * 
         * Reference:
         * [Revisiting Physically Based Shading at Imageworks, Christopher Kulla & Alejandro Conty, 2017]
         * 
         * https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
         */
        float remapped = base_roughness * sqrt(3.7f * (relative_eta - 1.0f) * hippt::square(relative_eta - 0.5f) / hippt::pow_3(relative_eta));

        // Remapped roughness starts going above 1.0f starting at relative eta around 1.9f
        // and ends up at 1.39f at relative eta 3.5f
        //
        // Because we don't expect the user to input higher IOR values than that,
        // we remap that remapped roughness from [0.0f, 1.39f] to [0.0f, 1.0f]
        // and if the user inputs higher IOR values than 3.5f, we clamp to 1.0f roughness
        // anyways
        return hippt::clamp(0.0f, 1.0f, remapped / 1.39f);
    }

    HIPRT_HOST_DEVICE void set_emission(ColorRGB32F new_emission)
    {
        emission = new_emission;
    }

    HIPRT_HOST_DEVICE ColorRGB32F get_emission() const
    {
        return emission * emission_strength;
    }

    HIPRT_HOST_DEVICE ColorRGB32F get_original_emission() const
    {
        return emission;
    }

    bool emissive_texture_used = false;
    float emission_strength = 1.0f;
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
    bool thin_walled = false;

    float thin_film = 0.0f;
    float thin_film_ior = 1.3f;
    float thin_film_thickness = 500.0f;
    float thin_film_kappa_3 = 0.0f;
    float thin_film_hue_shift_degrees = 0.0f;
    float thin_film_base_ior_override = 1.0f;
    bool thin_film_do_ior_override = false;
    bool srgb = true;

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
    // See PrincipledBSDFGGXUseMultipleScattering in this codebase.
    bool enforce_strong_energy_conservation = false;

private:
    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
};

struct CPUTexturedRendererMaterial : public SimplifiedRendererMaterial
{
    static constexpr int NO_TEXTURE = -1;
    // When an emissive texture is read and is determine to be
    // constant, no emissive texture will be used. Instead,
    // we'll just set the emission of the material to that constant emission value
    // and the emissive texture index of the material will be replaced by
    // CONSTANT_EMISSIVE_TEXTURE
    static constexpr int CONSTANT_EMISSIVE_TEXTURE = -2;

    int normal_map_texture_index = -1;

    int emission_texture_index = -1;
    int base_color_texture_index = -1;

    // If not -1, there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness
    int roughness_metallic_texture_index = -1;

    int roughness_texture_index = -1;
    int oren_sigma_texture_index = -1;

    int metallic_texture_index = -1;
    int specular_texture_index = -1;
    int specular_tint_texture_index = -1;
    int specular_color_texture_index = -1;
    
    int anisotropic_texture_index = -1;
    int anisotropic_rotation_texture_index = -1;

    int coat_texture_index = -1;
    int coat_roughness_texture_index = -1;
    int coat_ior_texture_index = -1;

    int sheen_texture_index = -1;
    int sheen_roughness_texture_index = -1;
    int sheen_color_texture_index = -1;

    int specular_transmission_texture_index = -1;
};

#endif
