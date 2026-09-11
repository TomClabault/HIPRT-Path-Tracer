/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_OPTIONS_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"
#include "Renderer/LightTree/LightTreeBuilderOptionsCommon.h"

struct LightTreeSGBuilderOptions
{
	static constexpr int LIGHT_TREE_SG_DEFAULT_SPATIAL_LOBE_COUNT = 8;

	int build_split_method = LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED;
	int bin_count		   = 64;

	int cost_function						 = LIGHT_TREE_BUILD_COST_FUNCTION_SAOH;
	bool stop_splitting_if_cost_not_worth_it = false;

	int max_triangles_per_leaf = 1;

	int spatial_lobe_count = (DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator))
								 ? 1
								 : LIGHT_TREE_SG_DEFAULT_SPATIAL_LOBE_COUNT;
	int tree_cut_size	   = 256;
	int tree_cut_size_neural_many_lights = 64;
};

#endif // #ifndef RENDERER_LIGHT_TREE_SG_BUILDER_OPTIONS_H
