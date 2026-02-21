#ifndef SCENE_AABB_H
#define SCENE_AABB_H

#include "HostDeviceCommon/Maths/Math.h"

/**
 * Axis Aligned Bounding Box class
 */
struct AABB
{
	AABB() {}
	AABB(float3_t mini, float3_t maxi) : mini(mini), maxi(maxi) {}

	/**
	 * Extends this bounding box with the given one
	 */
	void extend(const AABB& other)
	{
		mini = hippt::min(mini, other.mini);
		maxi = hippt::max(maxi, other.maxi);
	}

	/**
	 * Extends the bounding box with a vertex
	 */
	void extend(float3_t vertex)
	{
		mini = make_float3(hippt::min(mini.x, vertex.x), hippt::min(mini.y, vertex.y), hippt::min(mini.z, vertex.z));
		maxi = make_float3(hippt::max(maxi.x, vertex.x), hippt::max(maxi.y, vertex.y), hippt::max(maxi.z, vertex.z));
	}

	/**
	 * Returns the length of the longest extent of the bounding box
	 */
	float get_max_extent() const
	{
		return hippt::max(hippt::abs(mini.x - maxi.x), hippt::max(hippt::abs(mini.y - maxi.y), hippt::abs(mini.z - maxi.z)));
	}

	/**
	 * Returns the length of the extent in the coordinate 'coord'
	 *
	 * X = 0, Y = 1, Z = 2
	 */
	float get_extent(int coord) const
	{
		return *(&maxi.x + coord) - *(&mini.x + coord);
	}

	float3_t get_extents() const
	{
		return make_float3(get_extent(0), get_extent(1), get_extent(2));
	}

	float3_t get_center() const
	{
		return (mini + maxi) * 0.5f;
	}

	float area() const
	{
		float3_t extents = get_extents();
		return 2.0f * (extents.x * extents.y + extents.y * extents.z + extents.z * extents.x);
	}

	float3_t mini = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() , std::numeric_limits<float>::max() };
	float3_t maxi = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() , -std::numeric_limits<float>::max() };
};

#endif