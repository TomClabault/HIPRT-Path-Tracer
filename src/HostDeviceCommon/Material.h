/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Math.h"

enum BRDF
{
    Uninitialized,
    Disney,
    SpecularFresnel
};

struct RendererMaterial
{
    HIPRT_HOST_DEVICE bool is_emissive()
    {
        return emission.r != 0.0f || emission.g != 0.0f || emission.b != 0.0f;
    }

    /*
     * Clamps some of the parameters of the material to avoid edge cases like NaNs
     * during rendering (i.e. numerical instabilities)
     */
    HIPRT_HOST_DEVICE void make_safe()
    {
        roughness = hippt::max(1.0e-4f, roughness);
        clearcoat_roughness = hippt::max(1.0e-4f, clearcoat_roughness);
        if (abs(ior - 1.0f) < 0.01f)
            ior = 1.01f;

        // Clamping to avoid negative emission
        emission = max(ColorRGB(0.0f), emission);

        // Avoiding zero
        absorption_at_distance = hippt::max(absorption_at_distance, 1.0e-4f);
        absorption_color = max(absorption_color, ColorRGB(1.0f / 255.0f));
    }

    /*
     * Some properties of the material can be precomputed
     * This function does it
     */
    HIPRT_HOST_DEVICE void precompute_properties()
    {
        precompute_anisotropic();
        precompute_oren_nayar();
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

    BRDF brdf_type = BRDF::Uninitialized;

    int normal_map_texture_index = -1;

    int emission_texture_index = -1;
    int base_color_texture_index = -1;
    ColorRGB emission = ColorRGB{ 0.0f, 0.0f, 0.0f };
    ColorRGB base_color = ColorRGB{ 1.0f, 0.2f, 0.7f };

    // If not -1, there is only one texture for the metallic and the roughness parameters in which.
    // case the green channel is the roughness and the blue channel is the metalness
    int roughnes_metallic_texture_index = -1;

    int roughness_texture_index = -1;
    int oren_sigma_texture_index = -1;
    int subsurface_texture_index = -1;
    float roughness = 0.3f;
    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian
    float oren_nayar_A = 0.86516788142120468442f; // Precomputed A for sigma = 20 degrees
    float oren_nayar_B = 0.74147689828041305929f; // Precomputed B for sigma = 20 degrees
    float subsurface = 0.0f;

    int metallic_texture_index = -1;
    int specular_texture_index = -1;
    int specular_tint_texture_index = -1;
    int specular_color_texture_index = -1;
    float metallic = 0.0f;
    float specular = 1.0f; // Specular intensity
    float specular_tint = 1.0f; // Specular fresnel strength for the metallic
    ColorRGB specular_color = ColorRGB(1.0f);
    
    int anisotropic_texture_index = -1;
    int anisotropic_rotation_texture_index = -1;
    float anisotropic = 0.0f;
    float anisotropic_rotation = 0.0f;
    float alpha_x, alpha_y;

    int clearcoat_texture_index = -1;
    int clearcoat_roughness_texture_index = -1;
    int clearcoat_ior_texture_index = -1;
    float clearcoat = 1.0f;
    float clearcoat_roughness = 0.0f;
    float clearcoat_ior = 1.5f;

    int sheen_texture_index = -1;
    int sheen_tint_color_texture_index = -1;
    int sheen_color_texture_index = -1;
    float sheen = 0.0f; // Sheen strength
    float sheen_tint = 0.0f; // Sheen tint strength
    ColorRGB sheen_color = ColorRGB(1.0f);

    int ior_texture_index = -1;
    int specular_transmission_texture_index = -1;
    float ior = 1.40f;
    float specular_transmission = 0.0f;
    // Volume absorption density
    float absorption_at_distance = 1.0f;
    ColorRGB absorption_color = ColorRGB(1.0f);
};

#endif