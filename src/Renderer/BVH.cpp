/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "Renderer/BVH.h"

const float3 BoundingVolume::PLANE_NORMALS[BVHConstants::PLANES_COUNT] = {
	float3(1, 0, 0),
	float3(0, 1, 0),
    float3(0, 0, 1),
    float3(std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
    float3(-std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
    float3(-std::sqrt(3.0f) / 3, -std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
    float3(std::sqrt(3.0f) / 3, -std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
};

BVH::BVH() : _root(nullptr), _triangles(nullptr) {}
BVH::BVH(std::vector<Triangle>* triangles, int max_depth, int leaf_max_obj_count) : _triangles(triangles)
{
	BoundingVolume volume;
	float3 minimum(INFINITY, INFINITY, INFINITY), maximum(-INFINITY, -INFINITY, -INFINITY);

	for (const Triangle& triangle : *triangles)
	{
		volume.extend_volume(triangle);

		for (int i = 0; i < 3; i++)
		{
			minimum = hippt::min(minimum, triangle[i]);
			maximum = hippt::max(maximum, triangle[i]);
		}
	}

	//We now have a bounding volume to work with
	build_bvh(max_depth, leaf_max_obj_count, minimum, maximum, volume);
}

BVH::~BVH()
{
	delete _root;
}

void BVH::operator=(BVH&& bvh)
{
	_triangles = bvh._triangles;
	_root = bvh._root;

	bvh._root = nullptr;
}

void BVH::build_bvh(int max_depth, int leaf_max_obj_count, float3 min, float3 max, const BoundingVolume& volume)
{
	_root = new OctreeNode(min, max);

    for (int triangle_id = 0; triangle_id < _triangles->size(); triangle_id++)
        _root->insert(*_triangles, triangle_id, 0, max_depth, leaf_max_obj_count);

    _root->compute_volume(*_triangles);
}

bool BVH::intersect(const hiprtRay& ray, HitInfo& hit_info, FilterFunction filter_function, void* payload) const
{
    return _root->intersect(*_triangles, ray, hit_info, filter_function, payload);
}
