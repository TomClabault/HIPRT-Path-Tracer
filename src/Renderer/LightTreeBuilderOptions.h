/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_BUILDER_OPTIONS_H
#define RENDERER_LIGHT_TREE_BUILDER_OPTIONS_H

#define LIGHT_TREE_BUILD_OPTION_SPLIT_MIDPOINT 0
#define LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED 1

struct LightTreeBuilderOptions
{
	int build_split_method = LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED;
	int bin_count = 4;
};

#endif
