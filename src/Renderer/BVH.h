/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef BVH_H
#define BVH_H


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

class FlattenedBVH;

class BVH
{
public:
    struct OctreeNode
    {
        struct QueueElement
        {
            QueueElement(const BVH::OctreeNode* node, float t_near) : _node(node), _t_near(t_near) {}

            bool operator > (const QueueElement& a) const
            {
                return _t_near > a._t_near;
            }

            const OctreeNode* _node;//Reference on the node

            float _t_near;//Intersection distance used to order the elements in the priority queue used
            //by the OctreeNode to compute the intersection with a ray
        };

        OctreeNode(float3 min, float3 max) : _min(min), _max(max) {}
        ~OctreeNode()
        {
            if (_is_leaf)
                return;
            else
            {
                for (int i = 0; i < 8; i++)
                    delete _children[i];
            }
        }

        /*
          * Once the objects have been inserted in the hierarchy, this function computes
          * the bounding volume of all the node in the hierarchy
          */
        BoundingVolume compute_volume(const std::vector<Triangle>& triangles_geometry)
        {
            if (_is_leaf)
                for (int triangle_id : _triangles)
                    _bounding_volume.extend_volume(triangles_geometry[triangle_id]);
            else
                for (int i = 0; i < 8; i++)
                    _bounding_volume.extend_volume(_children[i]->compute_volume(triangles_geometry));

            return _bounding_volume;
        }

        void create_children(int max_depth, int leaf_max_obj_count)
        {
            float middle_x = (_min.x + _max.x) / 2;
            float middle_y = (_min.y + _max.y) / 2;
            float middle_z = (_min.z + _max.z) / 2;

            _children[0] = new OctreeNode(_min, float3(middle_x, middle_y, middle_z));
            _children[1] = new OctreeNode(float3(middle_x, _min.y, _min.z), float3(_max.x, middle_y, middle_z));
            _children[2] = new OctreeNode(_min + float3(0, middle_y, 0), float3(middle_x, _max.y, middle_z));
            _children[3] = new OctreeNode(float3(middle_x, middle_y, _min.z), float3(_max.x, _max.y, middle_z));
            _children[4] = new OctreeNode(_min + float3(0, 0, middle_z), float3(middle_x, middle_y, _max.z));
            _children[5] = new OctreeNode(float3(middle_x, _min.y, middle_z), float3(_max.x, middle_y, _max.z));
            _children[6] = new OctreeNode(_min + float3(0, middle_y, middle_z), float3(middle_x, _max.y, _max.z));
            _children[7] = new OctreeNode(float3(middle_x, middle_y, middle_z), float3(_max.x, _max.y, _max.z));
        }

        void insert(const std::vector<Triangle>& triangles_geometry, int triangle_id_to_insert, int current_depth, int max_depth, int leaf_max_obj_count)
        {
            bool depth_exceeded = max_depth != -1 && current_depth == max_depth;

            if (_is_leaf || depth_exceeded)
            {
                _triangles.push_back(triangle_id_to_insert);

                if (_triangles.size() > leaf_max_obj_count && !depth_exceeded)
                {
                    _is_leaf = false;//This node isn't a leaf anymore

                    create_children(max_depth, leaf_max_obj_count);

                    for (int triangle_id : _triangles)
                        insert_to_children(triangles_geometry, triangle_id, current_depth, max_depth, leaf_max_obj_count);

                    _triangles.clear();
                    _triangles.shrink_to_fit();
                }
            }
            else
                insert_to_children(triangles_geometry, triangle_id_to_insert, current_depth, max_depth, leaf_max_obj_count);

        }

        void insert_to_children(const std::vector<Triangle>& triangles_geometry, int triangle_id_to_insert, int current_depth, int max_depth, int leaf_max_obj_count)
        {
            const Triangle& triangle = triangles_geometry[triangle_id_to_insert];
            float3 bbox_centroid = triangle.bbox_centroid();

            float middle_x = (_min.x + _max.x) / 2;
            float middle_y = (_min.y + _max.y) / 2;
            float middle_z = (_min.z + _max.z) / 2;

            int octant_index = 0;

            if (bbox_centroid.x > middle_x) octant_index += 1;
            if (bbox_centroid.y > middle_y) octant_index += 2;
            if (bbox_centroid.z > middle_z) octant_index += 4;

            _children[octant_index]->insert(triangles_geometry, triangle_id_to_insert, current_depth + 1, max_depth, leaf_max_obj_count);
        }

        bool intersect(const std::vector<Triangle>& triangles_geometry, const hiprtRay& ray, HitInfo& hit_info) const
        {
            float trash;

            float denoms[BVHConstants::PLANES_COUNT];
            float numers[BVHConstants::PLANES_COUNT];

            for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
            {
                denoms[i] = hippt::dot(BoundingVolume::PLANE_NORMALS[i], ray.direction);
                numers[i] = hippt::dot(BoundingVolume::PLANE_NORMALS[i], float3(ray.origin));
            }

            return intersect(triangles_geometry, ray, hit_info, trash, denoms, numers);
        }

        bool intersect(const std::vector<Triangle>& triangles_geometry, const hiprtRay& ray, HitInfo& hit_info, float& t_near, float* denoms, float* numers) const
        {
            float t_far, trash;

            if (!_bounding_volume.intersect(trash, t_far, denoms, numers))
                return false;

            if (_is_leaf)
            {
                for (int triangle_id : _triangles)
                {
                    const Triangle& triangle = triangles_geometry[triangle_id];

                    HitInfo local_hit_info;
                    if (triangle.intersect(ray, local_hit_info))
                    {
                        hiprtHit hit;
                        hit.normal = local_hit_info.geometric_normal;
                        hit.primID = triangle_id;
                        hit.t = local_hit_info.t;
                        hit.uv = local_hit_info.uv;

                        if (local_hit_info.t < hit_info.t || hit_info.t == -1)
                        {
                            hit_info = local_hit_info;
                            hit_info.primitive_index = triangle_id;
                        }
                    }
                }

                t_near = hit_info.t;

                return t_near > 0;
            }

            std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> intersection_queue;
            for (int i = 0; i < 8; i++)
            {
                float inter_distance;
                if (_children[i]->_bounding_volume.intersect(inter_distance, t_far, denoms, numers))
                    intersection_queue.emplace(QueueElement(_children[i], inter_distance));
            }

            bool intersection_found = false;
            float closest_inter = 100000000, inter_distance = 100000000;
            while (!intersection_queue.empty())
            {
                QueueElement top_element = intersection_queue.top();
                intersection_queue.pop();

                if (top_element._node->intersect(triangles_geometry, ray, hit_info, inter_distance, denoms, numers))
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
        bool _is_leaf = true;

        std::vector<int> _triangles;
        std::array<BVH::OctreeNode*, 8> _children;

        float3 _min, _max;
        BoundingVolume _bounding_volume;
    };

public:
    BVH();
    BVH(std::vector<Triangle>* triangles, int max_depth = 32, int leaf_max_obj_count = 8);
    ~BVH();

    void operator=(BVH&& bvh);
     
    bool intersect(const hiprtRay& ray, HitInfo& hit_info) const;
    FlattenedBVH flatten() const;

private:
    void build_bvh(int max_depth, int leaf_max_obj_count, float3 min, float3 max, const BoundingVolume& volume);

public:
    OctreeNode* _root;

    std::vector<Triangle>* _triangles;
};

#endif
