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
    ColorRGB32F base_color = ColorRGB32F{ 0.9868f, 0.9830f, 0.9667f };

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

    ColorRGB32F coat_medium_absorption = ColorRGB32F{ 1.0f, 1.0f, 1.0f };
    float coat = 0.0f;
    float coat_roughness = 0.0f;
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

    // 1.0f makes the material completely opaque
    // 0.0f completely transparent (becomes invisible)
    float alpha_opacity = 1.0f;

    // Nested dielectric parameter
    int dielectric_priority = 0;

private:
    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
};

struct RendererMaterial : public SimplifiedRendererMaterial
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
