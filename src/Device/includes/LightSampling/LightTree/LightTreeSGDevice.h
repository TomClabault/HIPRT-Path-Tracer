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

struct SpatialSGLobeDevice
{
	float3_t mean;
	float variance		 = 0.0f;
	float power			 = 0.0f;
	float support_radius = 0.0f;
};

struct LightTreeSGNodeDevice
{
	HIPRT_DEVICE float get_total_power() const
	{
		// total_power stores the SG amplitude used during importance evaluation. Undo its baked-in normalization when an actual conserved power is needed.
		return total_power * SG_integral(vmf.sharpness);
	}

	HIPRT_DEVICE float get_energy_average() const
	{
		// Returns the raw (non-SG-divided) average power, matching the scale of energy_variance
		return energy_average;
	}

	// Note that this VMF is shared for all lobes of the node. One VMF distribution per each lobe yielded a bit better quality when I tried but the increased
	// size of the nodes made fetching nodes during traversal more expensive and it wasn't worth it overall. So this is instead an "average VMF" fitted over the
	// whole node.
	VMF vmf;

	// Points to this node's contiguous spatial SG lobe range in LightTreeSGDevice storage.
	SpatialSGLobeDevice* spatial_lobes = nullptr;
	unsigned int spatial_lobe_count	   = 1;

	float3_t gaussian_spatial_mean = make_float3(0.0f, 0.0f, 0.0f);

	float3_t orientation_axis = make_float3(0.0f, 0.0f, 0.0f);
	// Orientation cone angle
	float cos_theta_o = 0.0f;
	float sin_theta_o = 0.0f;

	float bounding_sphere_radius = 0.0f;

	// This contains a baked in division by SG_integral(node.vmf.sharpness)
	float total_power				 = 0.0f;
	float energy_variance			 = 0.0f;
	float energy_average			 = 0.0f;
	unsigned int total_emitter_count = 0;

	// If triangle count is 0, this contains the left child index
	// If triangle count is > 0, this is the first triangle index in the leaf node
	unsigned int left_child_index_or_first_triangle_index;
	unsigned int triangle_count;

	float3_t bounds_min;
	float3_t bounds_max;
};

/**
 * Everything that the SG importance of a node and a descent of the light tree read of a node and of its spatial lobe, in 96 bytes: the
 * record of a node loads with six 16-byte loads. The full node and its lobe live in two arrays and take 112 + 24 bytes, of which a descent
 * reads 88 + 24 with 9 loads per node.
 *
 * Only built when every node of the light tree has exactly one spatial lobe (the default). The power and mean of that lobe stand for the total
 * power and spatial mean of the node in the rejection tests (zero power, node behind the surface, backfacing cone): with a single lobe they are
 * the same values up to float rounding, LightTreeSGBuilder::fill_descent_records() logs the number of nodes for which they differ.
 *
 * Read through light_tree_sg_descent_node(); the statistics used by the splitting heuristics (energy variance and average, emitter count) are
 * not in the record and are still read from the nodes
 */
struct alignas(16) LightTreeSGDescentRecordDevice
{
	float3_t lobe_mean;
	// Contains the same baked-in division by SG_integral(vmf_sharpness) as LightTreeSGNodeDevice::total_power
	float lobe_power = 0.0f;

	float3_t vmf_axis;
	float vmf_sharpness = VMF::INVALID_SHARPNESS;

	float3_t orientation_axis;
	float bounding_sphere_radius = 0.0f;

	float3_t bounds_min;
	float bounds_max_x = 0.0f;

	float bounds_max_y = 0.0f;
	float bounds_max_z = 0.0f;
	// Orientation cone angle
	float cos_theta_o = 0.0f;
	float sin_theta_o = 0.0f;

	float lobe_variance		  = 0.0f;
	float lobe_support_radius = 0.0f;

	// If triangle count is 0, this contains the left child index
	// If triangle count is > 0, this is the first triangle index in the leaf node
	unsigned int left_child_index_or_first_triangle_index = 0;
	unsigned int triangle_count							  = 0;
};

static_assert(sizeof(LightTreeSGDescentRecordDevice) == 96, "The descent record must stay six 16-byte loads");

struct LightTreeSGDevice
{
	// Depth stored in 'node_path_codes' for a node that is too deep for its path to fit in a 32-bit trail
	static constexpr unsigned int INVALID_NODE_PATH_DEPTH = 0xFFFFFFFFu;

	LightTreeSGSettings settings;

	LightTreeSGNodeDevice* nodes	   = nullptr;
	SpatialSGLobeDevice* spatial_lobes = nullptr;
	// One record per node, same indexing as 'nodes'. nullptr when the nodes have more than one spatial lobe: the descents of the tree then read
	// the nodes and their lobes
	LightTreeSGDescentRecordDevice* descent_records = nullptr;
	unsigned int* tree_cut_node_indices				= nullptr;
	unsigned int* second_tree_cut_node_indices		= nullptr;

	int* indices_array		 = nullptr;
	unsigned int* bit_trails = nullptr;

	// One entry per node: x is the path from the root to the node with the same encoding as 'bit_trails' (bit d set if the path goes to the right
	// child at depth d), y is the depth of the node (0 for the root, INVALID_NODE_PATH_DEPTH if deeper than 32).
	//
	// A triangle is under a node if and only if the first 'depth' bits of its bit trail are the path of the node
	uint2_t* node_path_codes = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_LIGHT_TREE_SG_DEVICE_H
