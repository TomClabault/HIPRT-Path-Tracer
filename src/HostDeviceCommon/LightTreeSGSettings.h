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
	unsigned int spatial_lobe_count		   = 1;
	unsigned int tree_cut_size			   = 64;
};

#endif
