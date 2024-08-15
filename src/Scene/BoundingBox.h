#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include "HostDeviceCommon/Math.h"

/**
 * Axis Aligned Bounding Box class
 */
struct BoundingBox
{
	BoundingBox() {}
	BoundingBox(float3 mini, float3 maxi) : mini(mini), maxi(maxi) {}

	/**
	 * Extends this bounding box with the given one
	 */
	void extend(const BoundingBox& other)
	{
		mini = hippt::min(mini, other.mini);
		maxi = hippt::max(maxi, other.maxi);
	}

	/**
	 * Extends the bounding box with a vertex
	 */
	void extend(float3 vertex)
	{
		mini = make_float3(hippt::min(mini.x, vertex.x), hippt::min(mini.y, vertex.y), hippt::min(mini.z, vertex.z));
		maxi = make_float3(hippt::max(maxi.x, vertex.x), hippt::max(maxi.y, vertex.y), hippt::max(maxi.z, vertex.z));
	}

	/**
	 * Returns the length of the longest extent of the bounding box
	 */
	float get_max_extent() const
	{
		return hippt::max(hippt::abs(mini.x - maxi.x), hippt::max(hippt::abs(mini.y -maxi.y), hippt::abs(mini.z - maxi.z)));
	}

	float3 mini = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() , std::numeric_limits<float>::max() };
	float3 maxi = { -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() , -std::numeric_limits<float>::max() };
};

#endif