/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_OPTIONS_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_OPTIONS_H

#include "Renderer/LightTree/LightTreeBuilderOptionsCommon.h"

struct LightTreeSGBuilderOptions
{
	int build_split_method = LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED;
	int bin_count		   = 64;

	int cost_function						 = LIGHT_TREE_BUILD_COST_FUNCTION_SAOH;
	bool stop_splitting_if_cost_not_worth_it = false;

	int max_triangles_per_leaf = 1;

	int spatial_lobe_count				 = 8;
	int tree_cut_size					 = 256;
	int tree_cut_size_neural_many_lights = 64;
};

#endif
