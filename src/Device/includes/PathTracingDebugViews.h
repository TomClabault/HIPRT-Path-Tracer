/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PATH_TRACING_DEBUG_VIEWS_H
#define DEVICE_INCLUDES_PATH_TRACING_DEBUG_VIEWS_H

#include "Device/includes/Hash.h"
#include "Device/includes/PathTracing.h"
#include "HostDeviceCommon/KernelOptions/HeatmapOptions.h"

HIPRT_DEVICE bool path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(const HIPRTRenderData& render_data,
																							   unsigned int pixel_index,
																							   unsigned int& out_lightcut_index);

HIPRT_DEVICE bool path_tracing_compute_learning_to_cluster_cut_size_debug_value(const HIPRTRenderData& render_data, int pixel_index, float& out_debug_value)
{
#if LearningToClusterDebugMode != LEARNING_TO_CLUSTER_DEBUG_MODE_LIGHT_CUT_SIZE_HEATMAP
	return false;
#else
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
		return false;

	const IlluminationAwareKDTreeDevice& kd_tree_device = render_data.kd_tree_device;

	unsigned int lightcut_index;
	if (!path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(render_data, static_cast<unsigned int>(pixel_index), lightcut_index))
		return false;

	unsigned int lightcut_size = kd_tree_device.learning_to_cluster.lightcut_data[lightcut_index].lightcut_size;
	out_debug_value			   = static_cast<float>(lightcut_size) / static_cast<float>(LearningToClusterMaximumLightCutSize);

	return true;
#endif // #if LearningToClusterDebugMode != LEARNING_TO_CLUSTER_DEBUG_MODE_LIGHT_CUT_SIZE_HEATMAP
}

HIPRT_DEVICE bool path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(const HIPRTRenderData& render_data,
																							   unsigned int pixel_index,
																							   unsigned int& out_lightcut_index)
{
	out_lightcut_index = IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX;

	if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1 || render_data.buffers.global_triangle_index_to_mesh_index == nullptr)
		return false;

	const IlluminationAwareKDTreeDevice& kd_tree_device = render_data.kd_tree_device;
	if (kd_tree_device.core.nodes == nullptr || kd_tree_device.core.node_capacity == 0 || kd_tree_device.learning_to_cluster.normal_lightcut_sets == nullptr ||
		kd_tree_device.learning_to_cluster.lightcut_data == nullptr)
		return false;

	IlluminationAwareKDTreeSGShadingContext context{};
	context.position	   = render_data.g_buffer.primary_hit_position[pixel_index];
	context.shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
	unsigned int mesh_id   = render_data.buffers.global_triangle_index_to_mesh_index[render_data.g_buffer.first_hit_prim_index[pixel_index]];

	out_lightcut_index = kd_tree_device.resolve_lightcut(context, mesh_id);
	if (out_lightcut_index == IlluminationAwareKDTreeNode::INVALID_LIGHTCUT_INDEX || out_lightcut_index >= kd_tree_device.learning_to_cluster.lightcut_capacity)
		return false;

	return true;
}

HIPRT_DEVICE bool path_tracing_get_learning_to_cluster_cell_normal_face(const HIPRTRenderData& render_data,
																		unsigned int pixel_index,
																		unsigned int& out_guiding_cell_index,
																		unsigned int& out_normal_face)
{
	out_guiding_cell_index = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;
	out_normal_face		   = 0u;

	if (render_data.g_buffer.first_hit_prim_index[pixel_index] == -1)
		return false;

	float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];

	unsigned int guiding_cell_index = render_data.kd_tree_device.core.find_guiding_cell(primary_hit);
	if (guiding_cell_index == IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
		return false;

	float3_t shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
	out_guiding_cell_index	= guiding_cell_index;
	out_normal_face			= illumination_aware_kd_tree_classify_surface_normal_face(shading_normal);
	return true;
}

HIPRT_DEVICE unsigned int path_tracing_get_learning_to_cluster_cell_normal_face_color_seed(unsigned int guiding_cell_index, unsigned int normal_face)
{
	return guiding_cell_index * SurfaceNormalFace_Count + normal_face;
}

HIPRT_DEVICE bool path_tracing_pixel_is_on_learning_to_cluster_cell_normal_face_outline(const HIPRTRenderData& render_data,
																						unsigned int pixel_index,
																						unsigned int guiding_cell_index,
																						unsigned int normal_face)
{
	unsigned int image_width  = render_data.render_settings.render_resolution.x;
	unsigned int image_height = render_data.render_settings.render_resolution.y;
	unsigned int pixel_x	  = pixel_index % image_width;
	unsigned int pixel_y	  = pixel_index / image_width;

	if (pixel_x > 0u)
	{
		unsigned int neighbor_guiding_cell_index;
		unsigned int neighbor_normal_face;
		if (path_tracing_get_learning_to_cluster_cell_normal_face(render_data, pixel_index - 1u, neighbor_guiding_cell_index, neighbor_normal_face) &&
			(neighbor_guiding_cell_index != guiding_cell_index || neighbor_normal_face != normal_face))
			return true;
	}

	if (pixel_x + 1u < image_width)
	{
		unsigned int neighbor_guiding_cell_index;
		unsigned int neighbor_normal_face;
		if (path_tracing_get_learning_to_cluster_cell_normal_face(render_data, pixel_index + 1u, neighbor_guiding_cell_index, neighbor_normal_face) &&
			(neighbor_guiding_cell_index != guiding_cell_index || neighbor_normal_face != normal_face))
			return true;
	}

	if (pixel_y > 0u)
	{
		unsigned int neighbor_guiding_cell_index;
		unsigned int neighbor_normal_face;
		if (path_tracing_get_learning_to_cluster_cell_normal_face(render_data, pixel_index - image_width, neighbor_guiding_cell_index, neighbor_normal_face) &&
			(neighbor_guiding_cell_index != guiding_cell_index || neighbor_normal_face != normal_face))
			return true;
	}

	if (pixel_y + 1u < image_height)
	{
		unsigned int neighbor_guiding_cell_index;
		unsigned int neighbor_normal_face;
		if (path_tracing_get_learning_to_cluster_cell_normal_face(render_data, pixel_index + image_width, neighbor_guiding_cell_index, neighbor_normal_face) &&
			(neighbor_guiding_cell_index != guiding_cell_index || neighbor_normal_face != normal_face))
			return true;
	}

	return false;
}

HIPRT_DEVICE bool path_tracing_pixel_is_on_learning_to_cluster_cell_normal_face_mesh_id_outline(const HIPRTRenderData& render_data,
																								unsigned int pixel_index,
																								unsigned int lightcut_index)
{
	unsigned int image_width  = render_data.render_settings.render_resolution.x;
	unsigned int image_height = render_data.render_settings.render_resolution.y;
	unsigned int pixel_x	  = pixel_index % image_width;
	unsigned int pixel_y	  = pixel_index / image_width;

	if (pixel_x > 0u)
	{
		unsigned int neighbor_lightcut_index;
		if (path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(render_data, pixel_index - 1u, neighbor_lightcut_index) &&
			neighbor_lightcut_index != lightcut_index)
			return true;
	}

	if (pixel_x + 1u < image_width)
	{
		unsigned int neighbor_lightcut_index;
		if (path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(render_data, pixel_index + 1u, neighbor_lightcut_index) &&
			neighbor_lightcut_index != lightcut_index)
			return true;
	}

	if (pixel_y > 0u)
	{
		unsigned int neighbor_lightcut_index;
		if (path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(render_data, pixel_index - image_width, neighbor_lightcut_index) &&
			neighbor_lightcut_index != lightcut_index)
			return true;
	}

	if (pixel_y + 1u < image_height)
	{
		unsigned int neighbor_lightcut_index;
		if (path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(render_data, pixel_index + image_width, neighbor_lightcut_index) &&
			neighbor_lightcut_index != lightcut_index)
			return true;
	}

	return false;
}

HIPRT_DEVICE bool path_tracing_pixel_is_on_tree_cut_bounding_box_edge(const HIPRTRenderData& render_data, int pixel_index, unsigned int& out_box_index)
{
	const LightTreeSGSettings& settings = render_data.light_tree_sg.settings;
	unsigned int* tree_cut_node_indices = render_data.light_tree_sg.tree_cut_node_indices;
	unsigned int tree_cut_size			= settings.effective_tree_cut_size;

	if (tree_cut_node_indices == nullptr || render_data.light_tree_sg.nodes == nullptr || tree_cut_size == 0)
		return false;

	for (unsigned int tree_cut_node_index = 0; tree_cut_node_index < tree_cut_size; tree_cut_node_index++)
	{
		const unsigned int node_index = tree_cut_node_indices[tree_cut_node_index];
		if (node_index == 0xFFFFFFFF)
			continue;

		const LightTreeSGNodeDevice& node = render_data.light_tree_sg.nodes[node_index];
		if (aabb_rasterize_pixel_is_on_aabb_edge(render_data.current_camera, render_data.render_settings.render_resolution, pixel_index, node.bounds_min,
												 node.bounds_max))
		{
			out_box_index = tree_cut_node_index;

			return true;
		}
	}

	return false;
}

HIPRT_DEVICE void path_tracing_compute_debug_view_debug_color(const HIPRTRenderData& render_data, int pixel_index, ColorRGB32F& out_debug_color)
{
	out_debug_color = DEFAULT_DEBUG_COLOR;

	// Modifying the ray color such that we display some debug color to the screen

#if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// We have a first hit
		float3_t primary_hit	= render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
		float3_t view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);

		unsigned int trash_checksum;
		NEEPlusPlusContext context;
		context.envmap		   = false;
		context.point_on_light = make_float3(0, 0, 0);
		context.shaded_point   = primary_hit;

		out_debug_color = ColorRGB32F::random_color(render_data.nee_plus_plus.hash_context(context, render_data.current_camera, trash_checksum));
		out_debug_color *= (render_data.render_settings.sample_number + 1);
		out_debug_color *= hippt::dot(shading_normal, view_direction);
	}
// #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
#elif LearningToClusterDebugMode != LEARNING_TO_CLUSTER_DEBUG_MODE_NO_DEBUG &&                                                                                 \
	ILLUMINATION_AWARE_KD_TREE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator, DirectLightSamplingStrategy)
#if LearningToClusterDebugMode == LEARNING_TO_CLUSTER_DEBUG_MODE_LIGHT_CUT_SIZE_HEATMAP
	float learning_to_cluster_debug_value;
	if (path_tracing_compute_learning_to_cluster_cut_size_debug_value(render_data, pixel_index, learning_to_cluster_debug_value))
		out_debug_color = map_0_1_to_heatmap_color_by_index<LearningToClusterDebugModeHeatmapIndex>(learning_to_cluster_debug_value) *
						  (render_data.render_settings.sample_number + 1);
#elif LearningToClusterDebugMode ==                                                                                                                            \
	LEARNING_TO_CLUSTER_DEBUG_MODE_CELL_NORMAL_FACE_SOLID // #if LearningToClusterDebugMode == LEARNING_TO_CLUSTER_DEBUG_MODE_LIGHT_CUT_SIZE_HEATMAP
	unsigned int guiding_cell_index;

	unsigned int normal_face;
	if (path_tracing_get_learning_to_cluster_cell_normal_face(render_data, pixel_index, guiding_cell_index, normal_face))
		out_debug_color = ColorRGB32F::random_color(path_tracing_get_learning_to_cluster_cell_normal_face_color_seed(guiding_cell_index, normal_face)) *
						  (render_data.render_settings.sample_number + 1);
#elif LearningToClusterDebugMode ==                                                                                                                            \
	LEARNING_TO_CLUSTER_DEBUG_MODE_CELL_NORMAL_FACE_MESH_ID_SOLID // #if LearningToClusterDebugMode == LEARNING_TO_CLUSTER_DEBUG_MODE_CELL_NORMAL_FACE_SOLID
	unsigned int lightcut_index;
	if (path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(render_data, pixel_index, lightcut_index))
		out_debug_color = ColorRGB32F::random_color(lightcut_index) * (render_data.render_settings.sample_number + 1);
#elif LearningToClusterDebugMode ==                                                                                                                            \
	LEARNING_TO_CLUSTER_DEBUG_MODE_CELL_NORMAL_FACE_MESH_ID_OUTLINE							// #if LearningToClusterDebugMode ==
																							// LEARNING_TO_CLUSTER_DEBUG_MODE_CELL_NORMAL_FACE_MESH_ID_SOLID
	unsigned int lightcut_index;
	if (path_tracing_get_learning_to_cluster_cell_normal_face_mesh_id_lightcut_index(render_data, pixel_index, lightcut_index) &&
		path_tracing_pixel_is_on_learning_to_cluster_cell_normal_face_mesh_id_outline(render_data, pixel_index, lightcut_index))
		out_debug_color = ColorRGB32F::random_color(lightcut_index) * (render_data.render_settings.sample_number + 1);
#elif LearningToClusterDebugMode == LEARNING_TO_CLUSTER_DEBUG_MODE_CELL_NORMAL_FACE_OUTLINE // #if LearningToClusterDebugMode ==
																							// LEARNING_TO_CLUSTER_DEBUG_MODE_CELL_NORMAL_FACE_MESH_ID_OUTLINE
	unsigned int guiding_cell_index;
	unsigned int normal_face;
	if (path_tracing_get_learning_to_cluster_cell_normal_face(render_data, pixel_index, guiding_cell_index, normal_face) &&
		path_tracing_pixel_is_on_learning_to_cluster_cell_normal_face_outline(render_data, pixel_index, guiding_cell_index, normal_face))
		out_debug_color = ColorRGB32F::random_color(path_tracing_get_learning_to_cluster_cell_normal_face_color_seed(guiding_cell_index, normal_face)) *
						  (render_data.render_settings.sample_number + 1);
#endif // LearningToClusterDebugMode // #if LearningToClusterDebugMode == LEARNING_TO_CLUSTER_DEBUG_MODE_LIGHT_CUT_SIZE_HEATMAP

// #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
#elif IlluminationAwareKDTreeDebugMode != ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_NO_DEBUG &&                                                                    \
	ILLUMINATION_AWARE_KD_TREE_IS_ENABLED(DirectLightNEEEstimator, DirectLightSamplingStrategy)
#if IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_SOLID
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// We have a first hit
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];

		unsigned int guiding_cell_index = render_data.kd_tree_device.core.find_guiding_cell(primary_hit);
		if (guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode ==                                                                                                                      \
	ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE // #if IlluminationAwareKDTreeDebugMode ==
															   // ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_SOLID
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// A cell outline is detected where a neighboring primary-hit pixel belongs to a different guiding cell.
		const unsigned int guiding_cell_index = render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		const unsigned int image_width		  = render_data.render_settings.render_resolution.x;
		const unsigned int image_height		  = render_data.render_settings.render_resolution.y;
		const unsigned int pixel_x			  = pixel_index % image_width;
		const unsigned int pixel_y			  = pixel_index / image_width;
		bool is_cell_outline				  = false;

		if (pixel_x > 0)
		{
			const unsigned int neighbor_pixel_index = pixel_index - 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_x + 1 < image_width)
		{
			const unsigned int neighbor_pixel_index = pixel_index + 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_y > 0)
		{
			const unsigned int neighbor_pixel_index = pixel_index - image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (pixel_y + 1 < image_height)
		{
			const unsigned int neighbor_pixel_index = pixel_index + image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
		}

		if (is_cell_outline && guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
	}
#elif IlluminationAwareKDTreeDebugMode ==                                                                                                                      \
	ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_OUTLINE_AND_LOOKAHEAD // #if IlluminationAwareKDTreeDebugMode ==
																			 // ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_SOLID
	if (render_data.g_buffer.first_hit_prim_index[pixel_index] != -1)
	{
		// A cell outline is detected where a neighboring primary-hit pixel belongs to a different guiding cell.
		unsigned int guiding_cell_index	  = render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		unsigned int lookahead_cell_index = render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[pixel_index]);
		unsigned int image_width		  = render_data.render_settings.render_resolution.x;
		unsigned int image_height		  = render_data.render_settings.render_resolution.y;
		unsigned int pixel_x			  = pixel_index % image_width;
		unsigned int pixel_y			  = pixel_index / image_width;

		bool is_cell_outline		   = false;
		bool is_lookahead_cell_outline = false;

		if (pixel_x > 0)
		{
			unsigned int neighbor_pixel_index = pixel_index - 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_x + 1 < image_width)
		{
			unsigned int neighbor_pixel_index = pixel_index + 1;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_y > 0)
		{
			unsigned int neighbor_pixel_index = pixel_index - image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (pixel_y + 1 < image_height)
		{
			unsigned int neighbor_pixel_index = pixel_index + image_width;
			if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
				render_data.kd_tree_device.core.find_guiding_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) != guiding_cell_index)
				is_cell_outline = true;
			else if (render_data.g_buffer.first_hit_prim_index[neighbor_pixel_index] != -1 &&
					 render_data.kd_tree_device.core.find_lookahead_cell(render_data.g_buffer.primary_hit_position[neighbor_pixel_index]) !=
						 lookahead_cell_index)
				is_lookahead_cell_outline = true;
		}

		if (is_cell_outline && guiding_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1);
		else if (is_lookahead_cell_outline && lookahead_cell_index != IlluminationAwareKDTreeNode::INVALID_NODE_INDEX)
			// Using the same color as the encompassing guiding cell but darker
			out_debug_color = ColorRGB32F::random_color(guiding_cell_index) * (render_data.render_settings.sample_number + 1) * 0.5f;
	}
#endif // LightTreeSG debug mode // #if IlluminationAwareKDTreeDebugMode == ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE_KD_TREE_LEAF_SOLID

#elif SSBNPermutationDebugHashGrid == KERNEL_OPTION_TRUE									  // #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	ColorRGB32F color = ColorRGB32F::random_color(render_data.ssbn_settings.screen_space_hash_grid[pixel_index].x);
	color *= render_data.render_settings.sample_number + 1;

	out_debug_color = ColorRGB32F(color);
#elif SSBNPermutationDebugSeeds == KERNEL_OPTION_TRUE										  // #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	ColorRGB32F color = ColorRGB32F(render_data.get_input_random_seed(pixel_index) / (float)((unsigned int)(-1)));
	color *= render_data.render_settings.sample_number + 1;

	out_debug_color = ColorRGB32F(color);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_GRID_CELLS && ReSTIRPGEnable == KERNEL_OPTION_TRUE // #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
	float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();

	ColorRGB32F color;

	unsigned int checksum;
	unsigned int cell_index =
		render_data.render_settings.restir_pg_settings.get_hash_grid_cell_index_from_position_data(primary_hit, normal, render_data.current_camera, checksum) %
		render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells;
	if (!HashGrid::resolve_collision<ReSTIRPGHashGridCollisionResolveSteps, false>(
			render_data.render_settings.restir_pg_settings.hash_grid_checksums, render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells,
			cell_index, checksum))
		color = ColorRGB32F();
	else
		color = ColorRGB32F::random_color(cell_index);

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_DIRECTION &&                                                                                 \
	ReSTIRPGEnable == KERNEL_OPTION_TRUE // #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	ColorRGB32F color;

	if (render_data.render_settings.sample_number == 0 || render_data.render_settings.nb_bounces == 0)
	{
		// At sample 0 we don't have the distributions yet so we can't fetch the directions from the distributions themselves but we can just display the
		// directions that the directions are initialized with which are directions on the fibonacci sphere
		color = ColorRGB32F(fibonacci_sphere_direction(render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number,
													   ReSTIRPGDistributionComponentCount))
					.abs();
	}
	else
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();

		ReSTIRPGDistribution distribution =
			render_data.render_settings.restir_pg_settings.get_distribution_from_position_data(primary_hit, normal, render_data.current_camera);
		if (distribution.distribution_components[0].weight == 0.0f)
			color = ColorRGB32F(0.0f);
		else if (!hippt::is_finite(distribution.distribution_components[0].weight))
			color = ColorRGB32F(1.0e10f, 0.0e10f, 1.0e10f);
		else
			color = ColorRGB32F(
						hippt::normalize(
							distribution.distribution_components[render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number]
								.vmf.axis))
						.abs();
	}

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_SHARPNESS &&                                                                                 \
	ReSTIRPGEnable == KERNEL_OPTION_TRUE // #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	ColorRGB32F color;
	if (render_data.render_settings.sample_number == 0)
		// At sample 0 all distributions are at sharpness 50
		color = ColorRGB32F(50.0f);
	else
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		ReSTIRPGDistribution distribution =
			render_data.render_settings.restir_pg_settings.get_distribution_from_position_data(primary_hit, normal, render_data.current_camera);

		color = ColorRGB32F(distribution.distribution_components[render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number]
								.vmf.sharpness)
					.abs();
	}

	color /= render_data.render_settings.restir_pg_settings.debug_normalization_factor;

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#elif ReSTIRPGDebugMode == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_WEIGHT &&                                                                                    \
	ReSTIRPGEnable == KERNEL_OPTION_TRUE // #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG
	ColorRGB32F color;
	if (render_data.render_settings.sample_number == 0)
		// At sample 0 all distributions are at sharpness 50
		color = ColorRGB32F(0.25f);
	else
	{
		float3_t primary_hit = render_data.g_buffer.primary_hit_position[pixel_index];
		float3_t normal		 = render_data.g_buffer.geometric_normals[pixel_index].unpack();
		ReSTIRPGDistribution distribution =
			render_data.render_settings.restir_pg_settings.get_distribution_from_position_data(primary_hit, normal, render_data.current_camera);

		color = ColorRGB32F(
					distribution.distribution_components[render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number].weight)
					.abs();
	}

	color *= render_data.render_settings.restir_pg_settings.debug_normalization_factor;

	out_debug_color = color * (render_data.render_settings.sample_number + 1);
#endif									 // Switch on the debugging option // #if NEEPlusPlusDebugMode != NEE_PLUS_PLUS_DEBUG_MODE_NO_DEBUG

	// Draw the SG tree cut bounding boxes last so they remain visible on top of any other debug view.
	unsigned int box_index = -1;
	if (render_data.light_tree_sg.settings.debug_draw_tree_cut_bounding_boxes &&
		path_tracing_pixel_is_on_tree_cut_bounding_box_edge(render_data, pixel_index, box_index))
	{
		if (render_data.light_tree_sg.settings.debug_draw_random_colors_boxes)
			out_debug_color = ColorRGB32F::random_color(box_index) * (render_data.render_settings.sample_number + 1);
		else
			out_debug_color = ColorRGB32F(2.0f, 0.0f, 0.0f) * (render_data.render_settings.sample_number + 1);
	}
}

HIPRT_DEVICE void path_tracing_compute_debug_view_debug_color(
	const HIPRTRenderData& render_data, RayPayloadCommon& ray_payload, int pixel_index, Xorshift32Generator& rng, ColorRGB32F& out_debug_color)
{
	path_tracing_compute_debug_view_debug_color(render_data, pixel_index, out_debug_color);
}

#endif // #ifndef DEVICE_INCLUDES_PATH_TRACING_DEBUG_VIEWS_H
