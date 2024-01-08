#ifndef RENDERER_MATERIAL_H
#define RENDERER_MATERIAL_H

#include "color.h"

struct RendererMaterial
{
    Color emission = Color(0.0f, 0.0f, 0.0f);
    Color diffuse = Color(1.0f, 0.2f, 0.7f);
    Color subsurface_color = Color(1.0f);

    float metalness = 0.0f;
    float roughness = 1.0f;
    float ior = 1.40f;
};

#endif
