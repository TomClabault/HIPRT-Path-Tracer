/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SPHERE_H
#define SPHERE_H

#include "HostDeviceCommon/HitInfo.h"
#include <hiprt/hiprt_types.h> // for hiprtRay

struct Sphere
{
    Sphere(float3 center, float radius, int primitive_index) : center(center), radius(radius), primitive_index(primitive_index) { };

    inline bool intersect(const hiprtRay &ray, HitInfo& hit_info) const
    {
        float3 L = ray.origin - center;

        //dot(ray._direction, ray._direction) = 1 because direction is normalized
        constexpr float a = 1.0f;
        float b = 2.0f * hippt::dot(ray.direction, L);
        float c = hippt::dot(L, L) - radius * radius;

        float delta = b * b - 4.0f * a * c;
        if (delta < 0.0f)
            return false;
        else
        {
            constexpr float a2 = 2.0f * a;

            if (delta == 0.0f)
                hit_info.t = -b / a2;
            else
            {
                float sqrt_delta = std::sqrt(delta);

                float t1 = (-b - sqrt_delta) / a2;
                float t2 = (-b + sqrt_delta) / a2;

                if (t1 < t2)
                {
                    hit_info.t = t1;
                    if (hit_info.t < 0.0f)
                        hit_info.t = t2;
                }
            }

            if (hit_info.t < 0.0f)
                return false;

            hit_info.inter_point = ray.origin + ray.direction * hit_info.t;
            hit_info.shading_normal = hippt::normalize(hit_info.inter_point - center);
            hit_info.primitive_index = primitive_index;

            return true;
        }
    }

    float3 center;
    float radius;

    int primitive_index;
};

#endif
