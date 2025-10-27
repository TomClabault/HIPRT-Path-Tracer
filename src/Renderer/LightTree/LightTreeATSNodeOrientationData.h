/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_ATS_NODE_ORIENTATION_DATA_H
#define RENDERER_LIGHT_TREE_ATS_NODE_ORIENTATION_DATA_H

#include "Device/includes/ONB.h"
#include "Renderer/LightTree/LightTreeATSConstants.h"

struct LightTreeATSNodeOrientationData
{
	HIPRT_HOST void cone_union_with(const LightTreeATSNodeOrientationData& other)
	{
		cone_union_with_internal(other.axis, other.theta_o, other.theta_e);

		if (axis.x != LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS)
			axis = hippt::normalize(axis);
	}

	HIPRT_HOST void cone_union_with(float3 axis_b, float theta_o_b, float theta_e_b)
	{
		cone_union_with_internal(axis_b, theta_o_b, theta_e_b);

		if (axis.x != LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS)
			axis = hippt::normalize(axis);
	}

private:
	HIPRT_HOST void cone_union_with_internal(float3 axis_b, float theta_o_b, float theta_e_b)
	{
		float3 axis_a = this->axis;
		float theta_o_a = this->theta_o;
		float theta_e_a = this->theta_e;

		if (axis_a.x == LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS)
		{
			this->axis = axis_b;
			this->theta_o = theta_o_b;
			this->theta_e = theta_e_b;

			return;
		}
		if (axis_b.x == LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS)
		{
			this->axis = axis_a;
			this->theta_o = theta_o_a;
			this->theta_e = theta_e_a;

			return;
		}

		if (theta_o_b > theta_o_a)
		{
			std::swap(theta_o_a, theta_o_b);
			std::swap(axis_a, axis_b);
			std::swap(theta_e_a, theta_e_b);
		}

		float theta_d = acosf(hippt::clamp(-1.0f, 1.0f, hippt::dot(axis_a, axis_b)));
		float theta_e = hippt::max(theta_e_a, theta_e_b);

		if (hippt::min(theta_d + theta_o_b, (float)M_PI) <= theta_o_a)
		{
			this->axis = axis_a;
			this->theta_o = theta_o_a;
			this->theta_e = theta_e;

			return;
		}
		else
		{
			float theta_o = (theta_o_a + theta_d + theta_o_b) / 2.0f;
			if (theta_o >= M_PI)
			{
				this->axis = axis_a;
				this->theta_o = M_PI;
				this->theta_e = theta_e;

				return;
			}

			float3 cross_prod = hippt::cross(axis_a, axis_b);
			if (hippt::length(cross_prod) < 1.0e-10f)
			{
				this->axis = axis_a;
				this->theta_o = hippt::max(theta_o_a, theta_o_b);
				this->theta_e = theta_e;

				return;
			}

			float theta_r = theta_o - theta_o_a;
			float3 axis = hippt::normalize(rotate_vector(axis_a, hippt::normalize(cross_prod), theta_r));
			//float3 axis = hippt::normalize(rotate_vector(axis_a, hippt::normalize(cross_prod), theta_r));

			this->axis = axis;
			this->theta_o = theta_o;
			this->theta_e = theta_e;

			return;
		}
	}

public:
	// Axis of the cluster
	float3 axis = make_float3(LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS, LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS, LIGHT_TREE_ATS_NODE_UNINITIALIZED_AXIS);
	// Normal bounds
	float theta_o = 0.0f;
	// Emission extents
	float theta_e = 0.0f;
};

#endif
