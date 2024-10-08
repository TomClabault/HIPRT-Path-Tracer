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
    Disney,
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
    HIPRT_HOST_DEVICE bool is_emissive() const
    {
        return emission.r * emission_strength != 0.0f || emission.g * emission_strength != 0.0f || emission.b * emission_strength != 0.0f || emissive_texture_used;
    }

    /*
     * Clamps some of the parameters of the material to avoid edge cases like NaNs
     * during rendering (i.e. numerical instabilities)
     */
    HIPRT_HOST_DEVICE void make_safe()
    {
        roughness = hippt::max(1.0e-4f, roughness);
        clearcoat_roughness = hippt::max(1.0e-4f, clearcoat_roughness);

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
        precompute_anisotropic();
        precompute_oren_nayar();

        if (specular_transmission == 0.0f)
            // No transmission means that we should never skip this boundary --> max priority
            dielectric_priority = (1 << StackPriorityEntry::PRIORITY_MAXIMUM) - 1;
    }

    HIPRT_HOST_DEVICE void precompute_anisotropic()
    {
        // Precomputing alpha_x and alpha_y related to Disney's anisotropic metallic lobe
        float aspect = sqrt(1.0f - 0.9f * anisotropic);
        alpha_x = hippt::max(1.0e-4f, roughness * roughness / aspect);
        alpha_y = hippt::max(1.0e-4f, roughness * roughness * aspect);
    }

    HIPRT_HOST_DEVICE void precompute_oren_nayar()
    {
        // Oren Nayar base_color BRDF parameters
        float sigma = oren_nayar_sigma;
        float sigma2 = sigma * sigma;
        oren_nayar_A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
        oren_nayar_B = 0.45f * sigma2 / (sigma2 + 0.09f);
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

    BRDF brdf_type = BRDF::Uninitialized;

    bool emissive_texture_used = false;
    float emission_strength = 1.0f;
    ColorRGB32F base_color = ColorRGB32F{ 1.0f, 0.2f, 0.7f };

    float roughness = 0.3f;
    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian
    float oren_nayar_A = 0.86516788142120468442f; // Precomputed A for sigma = 20 degrees
    float oren_nayar_B = 0.74147689828041305929f; // Precomputed B for sigma = 20 degrees
    float subsurface = 0.0f;

    float metallic = 0.0f;
    float specular = 1.0f; // Specular intensity
    float specular_tint = 1.0f; // Specular fresnel strength for the metallic
    ColorRGB32F specular_color = ColorRGB32F(1.0f);
    
    float anisotropic = 0.0f;
    float anisotropic_rotation = 0.0f;
    float alpha_x = 0.0f, alpha_y = 0.0f;

    float clearcoat = 0.0f;
    float clearcoat_roughness = 0.0f;
    float clearcoat_ior = 1.5f;

    float sheen = 0.0f; // Sheen strength
    float sheen_tint = 0.0f; // Sheen tint strength
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
    int subsurface_texture_index = -1;

    int metallic_texture_index = -1;
    int specular_texture_index = -1;
    int specular_tint_texture_index = -1;
    int specular_color_texture_index = -1;
    
    int anisotropic_texture_index = -1;
    int anisotropic_rotation_texture_index = -1;

    int clearcoat_texture_index = -1;
    int clearcoat_roughness_texture_index = -1;
    int clearcoat_ior_texture_index = -1;

    int sheen_texture_index = -1;
    int sheen_tint_color_texture_index = -1;
    int sheen_color_texture_index = -1;

    int specular_transmission_texture_index = -1;
};

#endif
