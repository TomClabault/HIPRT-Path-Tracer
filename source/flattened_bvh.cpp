#include "bounding_volume.h"
#include "bvh.h"
#include "flattened_bvh.h"

bool FlattenedBVH::FlattenedNode::intersect_volume(const std::array<float, BVHConstants::PLANES_COUNT>& denoms, const std::array<float, BVHConstants::PLANES_COUNT>& numers) const
{
    return BoundingVolume::intersect(d_near, d_far, denoms, numers);
}

bool FlattenedBVH::intersect(const Ray& ray, HitInfo& hit_info, const std::vector<Triangle>& triangles) const
{
    hit_info.t = -1;

    FlattenedBVH::Stack stack;
    stack.push(0);//Pushing the root of the BVH

    std::array<float, BVHConstants::PLANES_COUNT> denoms;
    std::array<float, BVHConstants::PLANES_COUNT> numers;

    for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
    {
        denoms[i] = dot(BoundingVolume::PLANE_NORMALS[i], ray.direction);
        numers[i] = dot(BoundingVolume::PLANE_NORMALS[i], Vector(ray.origin));
    }

    float closest_intersection_distance = -1;
    while (!stack.empty())
    {
        int node_index = stack.pop();
        const FlattenedNode& node = m_nodes[node_index];

        if (node.intersect_volume(denoms, numers))
        {
            if (node.is_leaf)
            {
                for (int i = 0; i < node.nb_triangles; i++)
                {
                    int triangle_index = node.triangles_indices[i];

                    HitInfo local_hit_info;
                    if (triangles[triangle_index].intersect(ray, local_hit_info))
                    {
                        if (closest_intersection_distance > local_hit_info.t || closest_intersection_distance == -1)
                        {
                            closest_intersection_distance = local_hit_info.t;
                            hit_info = local_hit_info;
                        }
                    }
                }
            }
            else
                for (int i = 0; i < 8; i++)
                    stack.push(node.children[i]);
        }
    }

    return hit_info.t > -1;
}
