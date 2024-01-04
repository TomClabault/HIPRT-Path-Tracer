#ifndef HIT_INFO_H
#define HIT_INFO_H

#include "vec.h"

struct HitInfo
{
    Point inter_point;
    Vector normal_at_intersection;

    float t = -1.0f; //Distance along ray
    float u = -1, v = -1; //Barycentric coordinates

    int primitive_index = -1;
};

#endif
