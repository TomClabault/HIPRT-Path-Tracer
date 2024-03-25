#ifndef MATERIAL_H
#define MATERIAL_H

#include "HostDeviceCommon/color.h"

enum BRDF
{
    Uninitialized,
    CookTorrance,
    SpecularFresnel
};

struct RendererMaterial
{
    bool is_emissive()
    {
        return emission.r != 0.0f || emission.g != 0.0f || emission.b != 0.0f;
    }

    BRDF brdf_type = BRDF::Uninitialized;

    Color emission = Color{ 0.0f, 0.0f, 0.0f };
    Color diffuse = Color{ 1.0f, 0.2f, 0.7f };

    float roughness = 1.0f;
    float oren_nayar_sigma = 0.34906585039886591538f; // 20 degrees standard deviation in radian
    float oren_nayar_A = 0.86516788142120468442f; // Precomputed A for sigma = 20 degrees
    float oren_nayar_B = 0.74147689828041305929f; // Precomputed A for sigma = 20 degrees
    float subsurface = 0.0f;

    float metalness = 0.0f;
    
    float anisotropic = 0.0f;
    float anisotropic_rotation = 0.0f;
    float alpha_x, alpha_y;

    float ior = 1.40f;
    float transmission_factor = 0.0f;
};

#endif