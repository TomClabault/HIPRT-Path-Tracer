#ifndef RAY_STATE_H
#define RAY_STATE_H

enum RayState
{
    BOUNCE,
    MISSED,
    TERMINATED
};

#ifdef __KERNELCC__
typedef hiprtFloat3 VectorType;
typedef hiprtFloat3 PointType;
#define UV_DECLARATION hiprtFloat2 uv;
#else
#include "Maths/vec.h"

typedef Vector VectorType;
typedef Point PointType;
#define UV_DECLARATION float u, v;
#endif

struct LightSourceInformation
{
    int emissive_triangle_index = -1;
    VectorType light_source_normal;
};

struct HitInfo
{
    PointType inter_point;
    VectorType normal_at_intersection;
    UV_DECLARATION;

    float t = -1.0f; // Distance along ray

    int primitive_index = -1;
};

#endif