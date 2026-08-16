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
	// These values are initialized from SGBuilder::to_device().
	unsigned int spatial_lobe_count = 0;
	// Effective tree cut size means that this can be 2 if the tree cut size is 4 but only 2 lobes are actually in the SG light tree because the scene is small.
	unsigned int effective_tree_cut_size		= 0;
	unsigned int effective_second_tree_cut_size = 0;

	bool debug_draw_tree_cut_bounding_boxes = false;
	bool debug_draw_random_colors_boxes		= true;
	bool debug_draw_first_tree_cut_boxes	= true;
	bool debug_draw_second_tree_cut_boxes	= false;
};

#endif
