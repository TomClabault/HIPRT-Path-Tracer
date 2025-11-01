/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_ATS_BUILDER_OPTIONS_H
#define RENDERER_LIGHT_TREE_ATS_BUILDER_OPTIONS_H

#define LIGHT_TREE_BUILD_OPTION_SPLIT_MIDPOINT 0
#define LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED 1

#define LIGHT_TREE_BUILD_COST_FUNCTION_SAH 0
#define LIGHT_TREE_BUILD_COST_FUNCTION_SAOH 1

struct LightTreeATSBuilderOptions
{
	int build_split_method = LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED;
	int bin_count = 4;

	int cost_function = LIGHT_TREE_BUILD_COST_FUNCTION_SAOH;

	int max_triangles_per_leaf = 1;
};

#endif
