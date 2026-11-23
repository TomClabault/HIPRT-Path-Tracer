/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H
#define DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H

#include "Device/includes/LightSampling/LightTree/SphericalGaussianUtils.h"
#include "Device/includes/PathGuiding/VMF.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/LightTreeSGSettings.h"
#include "Renderer/LightTree/LightTreeATSConstants.h"

/**
 * Node of the SG light tree: everything that the SG importance of a node and a descent of the light tree read, in 96 bytes that load with
 * six 16-byte loads.
 *
 * The statistics of the emitters of a node that only the splitting heuristics read are in LightTreeSGNodeEnergyStatisticsDevice, in a
 * separate array: in the node, they would make the loads of every node of every descent larger
 */
struct alignas(16) LightTreeSGNodeDevice
{
	HIPRT_DEVICE float get_total_power() const
	{
		// total_power stores the SG amplitude used during importance evaluation. Undo its baked-in normalization when an actual conserved power is needed.
		return total_power * SG_integral(vmf.sharpness);
	}

	// Power-weighted mean of the positions of the emitters of the node
	float3_t spatial_mean = make_float3(0.0f, 0.0f, 0.0f);
	// This contains a baked in division by SG_integral(node.vmf.sharpness)
	float total_power = 0.0f;

	// Note that this VMF is an "average VMF" fitted over the whole node. Splitting the node into several spatial lobes with one VMF
	// distribution per lobe yielded a bit better quality when I tried but the increased size of the nodes made fetching nodes during traversal
	// more expensive and it wasn't worth it overall.
	VMF vmf;

	float3_t orientation_axis	 = make_float3(0.0f, 0.0f, 0.0f);
	float bounding_sphere_radius = 0.0f;

	float3_t bounds_min = make_float3(0.0f, 0.0f, 0.0f);
	float3_t bounds_max = make_float3(0.0f, 0.0f, 0.0f);
	// Orientation cone angle
	float cos_theta_o = 0.0f;
	float sin_theta_o = 0.0f;

	// Power-weighted variance of the positions of the emitters of the node around 'spatial_mean'
	float spatial_variance = 0.0f;
	// Radius of the sphere centered on 'spatial_mean' that contains the bounds of the node: the same as 'bounding_sphere_radius' for an inner
	// node, larger for a leaf, whose bounding sphere only contains the vertices of its triangles
	float support_radius = 0.0f;

	// If triangle count is 0, this contains the left child index
	// If triangle count is > 0, this is the first triangle index in the leaf node
	unsigned int left_child_index_or_first_triangle_index = 0;
	unsigned int triangle_count							  = 0;
};

static_assert(sizeof(LightTreeSGNodeDevice) == 96, "The SG light tree node must stay six 16-byte loads");

/**
 * Statistics of the emitters of a node of the SG light tree, read by the adaptive splitting heuristics
 */
struct LightTreeSGNodeEnergyStatisticsDevice
{
	HIPRT_DEVICE float get_energy_average() const
	{
		// Returns the raw (non-SG-divided) average power, matching the scale of energy_variance
		return energy_average;
	}

	float energy_variance			 = 0.0f;
	float energy_average			 = 0.0f;
	unsigned int total_emitter_count = 0;
};

struct LightTreeSGDevice
{
	// Depth stored in 'node_path_codes' for a node that is too deep for its path to fit in a 32-bit trail
	static constexpr unsigned int INVALID_NODE_PATH_DEPTH = 0xFFFFFFFFu;

	LightTreeSGSettings settings;

	LightTreeSGNodeDevice* nodes = nullptr;
	// One entry per node, same indexing as 'nodes'
	LightTreeSGNodeEnergyStatisticsDevice* node_energy_statistics = nullptr;
	unsigned int* tree_cut_node_indices							  = nullptr;
	unsigned int* second_tree_cut_node_indices					  = nullptr;

	int* indices_array		 = nullptr;
	unsigned int* bit_trails = nullptr;

	// One entry per node: x is the path from the root to the node with the same encoding as 'bit_trails' (bit d set if the path goes to the right
	// child at depth d), y is the depth of the node (0 for the root, INVALID_NODE_PATH_DEPTH if deeper than 32).
	//
	// A triangle is under a node if and only if the first 'depth' bits of its bit trail are the path of the node
	uint2_t* node_path_codes = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H
