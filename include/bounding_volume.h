#ifndef BOUNDING_VOLUME_H
#define BOUNDING_VOLUME_H

#include "bvh_constants.h"
#include "triangle.h"

#include <array>

struct BoundingVolume
{
    static const Vector PLANE_NORMALS[BVHConstants::PLANES_COUNT];

    std::array<float, BVHConstants::PLANES_COUNT> _d_near;
    std::array<float, BVHConstants::PLANES_COUNT> _d_far;

    BoundingVolume()
    {
        for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
        {
            _d_near[i] = INFINITY;
            _d_far[i] = -INFINITY;
        }
    }

    static void triangle_volume(const Triangle& triangle, std::array<float, BVHConstants::PLANES_COUNT>& d_near, std::array<float, BVHConstants::PLANES_COUNT>& d_far)
    {
        for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                float dist = dot(BoundingVolume::PLANE_NORMALS[i], Vector(triangle[j]));

                d_near[i] = std::min(d_near[i], dist);
                d_far[i] = std::max(d_far[i], dist);
            }
        }
    }

    void extend_volume(const std::array<float, BVHConstants::PLANES_COUNT>& d_near, const std::array<float, BVHConstants::PLANES_COUNT>& d_far)
    {
        for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
        {
            _d_near[i] = std::min(_d_near[i], d_near[i]);
            _d_far[i] = std::max(_d_far[i], d_far[i]);
        }
    }

    void extend_volume(const BoundingVolume& volume)
    {
        extend_volume(volume._d_near, volume._d_far);
    }

    void extend_volume(const Triangle& triangle)
    {
        std::array<float, BVHConstants::PLANES_COUNT> d_near;
        std::array<float, BVHConstants::PLANES_COUNT> d_far;

        for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
        {
            d_near[i] = INFINITY;
            d_far[i] = -INFINITY;
        }

        triangle_volume(triangle, d_near, d_far);
        extend_volume(d_near, d_far);
    }

    static bool intersect(const std::array<float, BVHConstants::PLANES_COUNT>& d_near,
                          const std::array<float, BVHConstants::PLANES_COUNT>& d_far,
                          const std::array<float, BVHConstants::PLANES_COUNT>& denoms,
                          const std::array<float, BVHConstants::PLANES_COUNT>& numers)
    {
        float t_near = -INFINITY;
        float t_far = INFINITY;

        for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
        {
            float denom = denoms[i];
            if (denom == 0.0)
                continue;

            //inverse denom to avoid division
            float d_near_i = (d_near[i] - numers[i]) / denom;
            float d_far_i = (d_far[i] - numers[i]) / denom;
            if (denom < 0)
                std::swap(d_near_i, d_far_i);

            t_near = std::max(t_near, d_near_i);
            t_far = std::min(t_far, d_far_i);

            if (t_far < t_near)
                return false;
        }

        return true;
    }

    /**
     * @params denoms Precomputed denominators
     */
    bool intersect(float& t_near, float& t_far, float* denoms, float* numers) const
    {
        t_near = -INFINITY;
        t_far = INFINITY;

        for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
        {
            float denom = denoms[i];
            if (denom == 0.0f)
                continue;

            //inverse denom to avoid division
            float d_near_i = (_d_near[i] - numers[i]) / denom;
            float d_far_i = (_d_far[i] - numers[i]) / denom;
            if (denom < 0.0f)
                std::swap(d_near_i, d_far_i);

            t_near = std::max(t_near, d_near_i);
            t_far = std::min(t_far, d_far_i);

            if (t_far < t_near)
                return false;
        }

        return true;
    }
};

#endif
