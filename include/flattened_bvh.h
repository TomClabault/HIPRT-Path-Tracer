#ifndef FLATTENED_BVH_H
#define FLATTENED_BVH_H

#include "bounding_volume.h"
#include "bvh_constants.h"
#include "hit_info.h"
#include "ray.h"
#include "triangle.h"

#include <array>

class FlattenedBVH
{
public:
    struct Stack
    {
        void push(int value) { stack[index++] = value; }
        int pop() { return stack[--index]; }
        bool empty() { return index == 0; }

        int stack[BVHConstants:: FLATTENED_BVH_MAX_STACK_SIZE];
        int index = 0;
    };

    struct FlattenedNode
    {
        bool intersect_volume(const std::array<float, BVHConstants::PLANES_COUNT>& denoms, const std::array<float, BVHConstants::PLANES_COUNT>& numers) const;

        //Extents of the planes of the bounding volume
        std::array<float, BVHConstants::PLANES_COUNT> d_near;
        std::array<float, BVHConstants::PLANES_COUNT> d_far;

        //Indices of the children in the m_nodes vector
        std::array<int, 8> children = {-1, -1, -1, -1, -1, -1, -1, -1};
        std::array<int, BVHConstants::MAX_TRIANGLES_PER_LEAF> triangles_indices = { -1 };

        int nb_triangles = 0;
        int is_leaf;
    };

    bool intersect(const Ray& ray, HitInfo& hit_info, const std::vector<Triangle>& triangles) const;

    const std::vector<FlattenedNode>& get_nodes() const { return m_nodes; }
    std::vector<FlattenedNode>& get_nodes() { return m_nodes; }

private:
    std::vector<FlattenedNode> m_nodes;
};

#endif
