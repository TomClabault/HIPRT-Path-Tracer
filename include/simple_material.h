#ifndef SIMPLE_MATERIAL_H
#define SIMPLE_MATERIAL_H

#include "color.h"

struct SimpleMaterial
{
    Color emission = Color(0.0f, 0.0f, 0.0f);
    Color diffuse = Color(1.0f, 0.2f, 0.7f);

    float metalness = 0.0f;
    float roughness = 1.0f;
};

#endif
