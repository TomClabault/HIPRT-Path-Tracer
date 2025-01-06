/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "Renderer/BVH.h"

const float3 BoundingVolume::PLANE_NORMALS[BVHConstants::PLANES_COUNT] = {
	make_float3(1, 0, 0),
	make_float3(0, 1, 0),
    make_float3(0, 0, 1),
    make_float3(std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
    make_float3(-std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
    make_float3(-std::sqrt(3.0f) / 3, -std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
    make_float3(std::sqrt(3.0f) / 3, -std::sqrt(3.0f) / 3, std::sqrt(3.0f) / 3),
};

BVH::BVH() : m_root(nullptr), m_triangles(nullptr) {}
BVH::BVH(std::vector<Triangle>* triangles, int max_depth, int leaf_max_obj_count) : m_triangles(triangles)
{
	BoundingVolume volume;
	float3 minimum = make_float3(INFINITY, INFINITY, INFINITY);
	float3 maximum = make_float3(-INFINITY, -INFINITY, -INFINITY);

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
	delete m_root;
}

void BVH::operator=(BVH&& bvh)
{
	m_triangles = bvh.m_triangles;
	m_root = bvh.m_root;

	bvh.m_root = nullptr;
}

void BVH::build_bvh(int max_depth, int leaf_max_obj_count, float3 min, float3 max, const BoundingVolume& volume)
{
	m_root = new OctreeNode(min, max);

    for (int triangle_id = 0; triangle_id < m_triangles->size(); triangle_id++)
        m_root->insert(*m_triangles, triangle_id, 0, max_depth, leaf_max_obj_count);

    m_root->compute_volume(*m_triangles);
}

bool BVH::intersect(const hiprtRay& ray, hiprtHit& hit_info, void* filter_function_payload) const
{
    return m_root->intersect(*m_triangles, ray, hit_info, filter_function_payload);
}
