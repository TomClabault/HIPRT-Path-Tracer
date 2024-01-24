#ifndef RENDERER_MATERIAL_H
#define RENDERER_MATERIAL_H

#include "Image/color.h"

enum BRDF
{
    Uninitialized,
    CookTorrance,
    SpecularFresnel
};

struct RendererMaterial
{
    BRDF brdf_type;

    Color emission = Color(0.0f, 0.0f, 0.0f);
    Color diffuse = Color(1.0f, 0.2f, 0.7f);

    float metalness = 0.0f;
    float roughness = 1.0f;
    float ior = 1.40f;
    float transmission_factor = 0.0f;
};

#endif
