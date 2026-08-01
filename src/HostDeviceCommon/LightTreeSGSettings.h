/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LIGHT_TREE_SG_SETTINGS_H
#define HOST_DEVICE_COMMON_LIGHT_TREE_SG_SETTINGS_H

static constexpr int LIGHT_TREE_SG_MAX_SPATIAL_LOBES = 8;

struct LightTreeSGSettings
{
	float light_tree_sg_splitting_variance = 0.92f;
	// These 2 below are both initialized from SGBuilder::to_device()
	unsigned int spatial_lobe_count;
	unsigned int tree_cut_size;
};

#endif
