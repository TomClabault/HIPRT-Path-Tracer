/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "HostDeviceCommon/color.h"

#include "Kernels/includes/HIPRT_maths.h"

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
        roughness = RT_MAX(1.0e-4f, roughness);
        clearcoat_roughness = RT_MAX(1.0e-4f, clearcoat_roughness);
    }

    /*
     * Some properties of the material can be precomputed
     * This function does it
     */
    HIPRT_HOST_DEVICE void precompute_properties()
    {
        // Precomputing alpha_x and alpha_y related to Disney's anisotropic metallic lobe
        float aspect = sqrt(1.0f - 0.9f * anisotropic);
        alpha_x = RT_MAX(1.0e-4f, roughness * roughness / aspect);
        alpha_y = RT_MAX(1.0e-4f, roughness * roughness * aspect);

        // Oren Nayar base_color BRDF parameters
        float sigma = oren_nayar_sigma;
        float sigma2 = sigma * sigma;
        oren_nayar_A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
        oren_nayar_B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    BRDF brdf_type = BRDF::Uninitialized;

    Color emission = Color{ 0.0f, 0.0f, 0.0f };
    Color base_color = Color{ 1.0f, 0.2f, 0.7f };

    float roughness = 1.0f;
    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian
    float oren_nayar_A = 0.86516788142120468442f; // Precomputed A for sigma = 20 degrees
    float oren_nayar_B = 0.74147689828041305929f; // Precomputed B for sigma = 20 degrees
    float subsurface = 0.0f;

    float metallic = 0.0f;
    float specular = 1.0f; // Specular intensity
    float specular_tint = 1.0f; // Specular fresnel strength for the metallic
    Color specular_color = Color(1.0f);
    
    float anisotropic = 0.0f;
    float anisotropic_rotation = 0.0f;
    float alpha_x, alpha_y;

    float clearcoat = 1.0f;
    float clearcoat_roughness = 0.0f;
    float clearcoat_ior = 1.5f;

    float sheen = 0.0f; // Sheen strength
    float sheen_tint = 0.0f; // Sheen tint strength
    Color sheen_color = Color(1.0f);

    float ior = 1.40f;
    float specular_transmission = 0.0f;
};

#endif