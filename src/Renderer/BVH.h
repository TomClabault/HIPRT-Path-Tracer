/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef BVH_H
#define BVH_H

#include "Device/functions/FilterFunction.h"

#include "Renderer/BoundingVolume.h"
#include "Renderer/BVHConstants.h"
#include "Renderer/Triangle.h"

#include <array>
#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>

#include <hiprt/hiprt_types.h> // for hiprtRay

class BVH
{
public:
    struct OctreeNode
    {
        struct QueueElement
        {
            QueueElement(const BVH::OctreeNode* node, float t_near) : m_node(node), _t_near(t_near) {}

            bool operator > (const QueueElement& a) const
            {
                return _t_near > a._t_near;
            }

            const OctreeNode* m_node;//Reference on the node

            float _t_near;//Intersection distance used to order the elements in the priority queue used
            //by the OctreeNode to compute the intersection with a ray
        };

        OctreeNode(float3 min, float3 max) : m_min(min), m_max(max) {}
        ~OctreeNode()
        {
            if (m_is_leaf)
                return;
            else
            {
                for (int i = 0; i < 8; i++)
                    delete m_children[i];
            }
        }

        /*
          * Once the objects have been inserted in the hierarchy, this function computes
          * the bounding volume of all the node in the hierarchy
          */
        BoundingVolume compute_volume(const std::vector<Triangle>& triangles_geometry)
        {
            if (m_is_leaf)
                for (int triangle_id : m_triangles)
                    m_bounding_volume.extend_volume(triangles_geometry[triangle_id]);
            else
                for (int i = 0; i < 8; i++)
                    m_bounding_volume.extend_volume(m_children[i]->compute_volume(triangles_geometry));

            return m_bounding_volume;
        }

        void create_children(int max_depth, int leaf_max_obj_count)
        {
            float middle_x = (m_min.x + m_max.x) / 2;
            float middle_y = (m_min.y + m_max.y) / 2;
            float middle_z = (m_min.z + m_max.z) / 2;

            m_children[0] = new OctreeNode(m_min, make_float3(middle_x, middle_y, middle_z));
            m_children[1] = new OctreeNode(make_float3(middle_x, m_min.y, m_min.z), make_float3(m_max.x, middle_y, middle_z));
            m_children[2] = new OctreeNode(m_min + make_float3(0, middle_y, 0), make_float3(middle_x, m_max.y, middle_z));
            m_children[3] = new OctreeNode(make_float3(middle_x, middle_y, m_min.z), make_float3(m_max.x, m_max.y, middle_z));
            m_children[4] = new OctreeNode(m_min + make_float3(0, 0, middle_z), make_float3(middle_x, middle_y, m_max.z));
            m_children[5] = new OctreeNode(make_float3(middle_x, m_min.y, middle_z), make_float3(m_max.x, middle_y, m_max.z));
            m_children[6] = new OctreeNode(m_min + make_float3(0, middle_y, middle_z), make_float3(middle_x, m_max.y, m_max.z));
            m_children[7] = new OctreeNode(make_float3(middle_x, middle_y, middle_z), make_float3(m_max.x, m_max.y, m_max.z));
        }

        void insert(const std::vector<Triangle>& triangles_geometry, int triangle_id_to_insert, int current_depth, int max_depth, int leaf_max_obj_count)
        {
            bool depth_exceeded = max_depth != -1 && current_depth == max_depth;

            if (m_is_leaf || depth_exceeded)
            {
                m_triangles.push_back(triangle_id_to_insert);

                if (m_triangles.size() > leaf_max_obj_count && !depth_exceeded)
                {
                    m_is_leaf = false;//This node isn't a leaf anymore

                    create_children(max_depth, leaf_max_obj_count);

                    for (int triangle_id : m_triangles)
                        insert_to_children(triangles_geometry, triangle_id, current_depth, max_depth, leaf_max_obj_count);

                    m_triangles.clear();
                    m_triangles.shrink_to_fit();
                }
            }
            else
                insert_to_children(triangles_geometry, triangle_id_to_insert, current_depth, max_depth, leaf_max_obj_count);

        }

        void insert_to_children(const std::vector<Triangle>& triangles_geometry, int triangle_id_to_insert, int current_depth, int max_depth, int leaf_max_obj_count)
        {
            const Triangle& triangle = triangles_geometry[triangle_id_to_insert];
            float3 bbox_centroid = triangle.bbox_centroid();

            float middle_x = (m_min.x + m_max.x) / 2;
            float middle_y = (m_min.y + m_max.y) / 2;
            float middle_z = (m_min.z + m_max.z) / 2;

            int octant_index = 0;

            if (bbox_centroid.x > middle_x) octant_index += 1;
            if (bbox_centroid.y > middle_y) octant_index += 2;
            if (bbox_centroid.z > middle_z) octant_index += 4;

            m_children[octant_index]->insert(triangles_geometry, triangle_id_to_insert, current_depth + 1, max_depth, leaf_max_obj_count);
        }

        bool intersect(const std::vector<Triangle>& triangles_geometry, const hiprtRay& ray, hiprtHit& hit_info, void* filter_function_payload) const
        {
            float trash;

            float denoms[BVHConstants::PLANES_COUNT];
            float numers[BVHConstants::PLANES_COUNT];

            for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
            {
                denoms[i] = hippt::dot(BoundingVolume::PLANE_NORMALS[i], ray.direction);
                numers[i] = hippt::dot(BoundingVolume::PLANE_NORMALS[i], float3(ray.origin));
            }

            return intersect(triangles_geometry, ray, hit_info, trash, denoms, numers, filter_function_payload);
        }

        bool intersect(const std::vector<Triangle>& triangles_geometry, const hiprtRay& ray, hiprtHit& hit_info, float& t_near, float* denoms, float* numers, void* filter_function_payload) const
        {
            float t_far, trash;

            if (!m_bounding_volume.intersect(trash, t_far, denoms, numers))
                return false;

            if (m_is_leaf)
            {
                for (int triangle_id : m_triangles)
                {
                    const Triangle& triangle = triangles_geometry[triangle_id];

                    hiprtHit localHit;
                    if (triangle.intersect(ray, localHit))
                    {
                        localHit.primID = triangle_id;

                        if (filter_function(ray, nullptr, filter_function_payload, localHit))
                            // Hit is filtered
                            continue;

                        if (localHit.t < hit_info.t || hit_info.t == -1)
                            hit_info = localHit;
                    }
                }

                t_near = hit_info.t;

                return t_near > 0;
            }

            std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> intersection_queue;
            for (int i = 0; i < 8; i++)
            {
                float inter_distance;
                if (m_children[i]->m_bounding_volume.intersect(inter_distance, t_far, denoms, numers))
                    intersection_queue.emplace(QueueElement(m_children[i], inter_distance));
            }

            bool intersection_found = false;
            float closest_inter = 100000000, inter_distance = 100000000;
            while (!intersection_queue.empty())
            {
                QueueElement top_element = intersection_queue.top();
                intersection_queue.pop();

                if (top_element.m_node->intersect(triangles_geometry, ray, hit_info, inter_distance, denoms, numers, filter_function_payload))
                {
                    closest_inter = std::min(closest_inter, inter_distance);
                    intersection_found = true;

                    //If we found an intersection that is closer than
                    //the next element in the queue, we can stop intersecting further
                    if (intersection_queue.empty() || closest_inter < intersection_queue.top()._t_near)
                    {
                        t_near = closest_inter;

                        return true;
                    }
                }
            }

            if (!intersection_found)
                return false;
            else
            {
                t_near = closest_inter;

                return true;
            }
        }

        //If this node has been subdivided (and thus cannot accept any triangles),
        //this boolean will be set to false
        bool m_is_leaf = true;

        std::vector<int> m_triangles;
        std::array<BVH::OctreeNode*, 8> m_children = 
        {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr
        };

        float3 m_min, m_max;
        BoundingVolume m_bounding_volume;
    };

public:
    BVH();
    BVH(std::vector<Triangle>* triangles, int max_depth = 32, int leaf_max_obj_count = 8);
    ~BVH();

    void operator=(BVH&& bvh);
     
    bool intersect(const hiprtRay& ray, hiprtHit& hit_info, void* filter_function_payload) const;

private:
    void build_bvh(int max_depth, int leaf_max_obj_count, float3 min, float3 max, const BoundingVolume& volume);

public:
    OctreeNode* m_root;

    std::vector<Triangle>* m_triangles;
};

#endif
