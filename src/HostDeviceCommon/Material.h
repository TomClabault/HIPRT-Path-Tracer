/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_H
#define HOST_DEVICE_COMMON_MATERIAL_H

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

        // Clamping to avoid negative emission
        emission = ColorRGB32F::max(ColorRGB32F(0.0f), emission);

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
            dielectric_priority = 65535;
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
    ColorRGB32F emission = ColorRGB32F{ 0.0f, 0.0f, 0.0f };
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
    
    int anisotropic_texture_index = -1;
    int anisotropic_rotation_texture_index = -1;
    float anisotropic = 0.0f;
    float anisotropic_rotation = 0.0f;
    float alpha_x, alpha_y;

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

    // Nested dielectric parameter
    unsigned short int dielectric_priority = 0;
};

struct RendererMaterial : public SimplifiedRendererMaterial
{
    HIPRT_HOST_DEVICE RendererMaterial() {}
    HIPRT_HOST_DEVICE RendererMaterial(SimplifiedRendererMaterial simplified_mat)
    {
        this->brdf_type = simplified_mat.brdf_type;
        this->emission = simplified_mat.emission;
        this->base_color = simplified_mat.base_color;

        this->roughness = simplified_mat.roughness;
        this->oren_nayar_sigma = simplified_mat.oren_nayar_sigma;
        this->oren_nayar_A = simplified_mat.oren_nayar_A;
        this->oren_nayar_B = simplified_mat.oren_nayar_B;
        this->subsurface = simplified_mat.subsurface;

        this->metallic = simplified_mat.metallic;
        this->specular = simplified_mat.specular;
        this->specular_tint = simplified_mat.specular_tint;
        this->specular_color = simplified_mat.specular_color;

        this->anisotropic = simplified_mat.anisotropic;
        this->anisotropic_rotation = simplified_mat.anisotropic_rotation;
        this->alpha_x = simplified_mat.alpha_x;

        this->clearcoat = simplified_mat.clearcoat;
        this->clearcoat_roughness = simplified_mat.clearcoat_roughness;
        this->clearcoat_ior = simplified_mat.clearcoat_ior;

        this->sheen = simplified_mat.sheen;
        this->sheen_tint = simplified_mat.sheen_tint;
        this->sheen_color = simplified_mat.sheen_color;

        this->ior = simplified_mat.ior;
        this->specular_transmission = simplified_mat.specular_transmission;
        this->absorption_at_distance = simplified_mat.absorption_at_distance;
        this->absorption_color = simplified_mat.absorption_color;

        this->precompute_properties();
    }

    HIPRT_HOST_DEVICE SimplifiedRendererMaterial get_simplified_material()
    {
        SimplifiedRendererMaterial simplified;
        simplified.brdf_type = this->brdf_type;
        simplified.emission = this->emission;
        simplified.base_color = this->base_color;

        simplified.roughness = this->roughness;
        simplified.oren_nayar_sigma = this->oren_nayar_sigma;
        simplified.oren_nayar_A = this->oren_nayar_A;
        simplified.oren_nayar_B = this->oren_nayar_B;
        simplified.subsurface = this->subsurface;

        simplified.metallic = this->metallic;
        simplified.specular = this->specular;
        simplified.specular_tint = this->specular_tint;
        simplified.specular_color = this->specular_color;

        simplified.anisotropic = this->anisotropic;
        simplified.anisotropic_rotation = this->anisotropic_rotation;
        simplified.alpha_x = this->alpha_x;

        simplified.clearcoat = this->clearcoat;
        simplified.clearcoat_roughness = this->clearcoat_roughness;
        simplified.clearcoat_ior = this->clearcoat_ior;

        simplified.sheen = this->sheen;
        simplified.sheen_tint = this->sheen_tint;
        simplified.sheen_color = this->sheen_color;

        simplified.ior = this->ior;
        simplified.specular_transmission = this->specular_transmission;
        simplified.absorption_at_distance = this->absorption_at_distance;
        simplified.absorption_color = this->absorption_color;

        simplified.dielectric_priority = this->dielectric_priority;

        simplified.precompute_properties();

        return simplified;
    }

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