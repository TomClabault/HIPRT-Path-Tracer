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

    float metalness = 0.0f;
    float roughness = 1.0f;
    float subsurface = 0.0f;
    
    float anisotropy = 0.0f;
    float alpha_x, alpha_y;

    float ior = 1.40f;
    float transmission_factor = 0.0f;
};

#endif