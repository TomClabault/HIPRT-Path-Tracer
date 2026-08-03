/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/LightTree/LightTreeSGBuilder.h"

#include <algorithm>
#include <limits>

void LightTreeSGBuilder::build_light_tree(const std::vector<int>& emissive_triangles_primitive_indices,
										  const std::vector<float>& triangles_average_emissive_power_luminance,
										  const std::vector<int>& triangle_indices,
										  const std::vector<float3_t>& vertices_positions)
{
	m_light_tree_ats_builder.build_light_tree(emissive_triangles_primitive_indices, triangles_average_emissive_power_luminance, triangle_indices,
											  vertices_positions);

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Building SG light tree...");

	auto start = std::chrono::high_resolution_clock::now();
	m_nodes.resize(m_light_tree_ats_builder.get_nodes().size());
	if (m_nodes.empty())
	{
		m_first_tree_cut_node_indices.clear();
		m_second_tree_cut_node_indices.clear();
		m_effective_first_tree_cut_size	 = 0;
		m_effective_second_tree_cut_size = 0;

		return;
	}

	compute_node_spherical_gaussian(0, LightTreeBuilderTrianglesData(emissive_triangles_primitive_indices, triangle_indices, vertices_positions));
	compute_tree_cut();

	auto stop = std::chrono::high_resolution_clock::now();

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "SG Light tree construction time: %ldms",
							std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void LightTreeSGBuilder::compute_tree_cut()
{
	compute_tree_cut_for_size(m_first_tree_cut_size, m_first_tree_cut_node_indices, m_effective_first_tree_cut_size);
	compute_tree_cut_for_size(m_second_tree_cut_size, m_second_tree_cut_node_indices, m_effective_second_tree_cut_size);
}

void LightTreeSGBuilder::compute_tree_cut_for_size(int tree_cut_size, std::vector<unsigned int>& tree_cut_node_indices, unsigned int& effective_tree_cut_size)
{
	const std::vector<LightTreeATSNode>& ats_nodes = m_light_tree_ats_builder.get_nodes();
	tree_cut_node_indices						   = { 0 };

	size_t frontier_node_position = 0;
	while (tree_cut_node_indices.size() < static_cast<size_t>(tree_cut_size) && frontier_node_position < tree_cut_node_indices.size())
	{
		const LightTreeATSNode& ats_node = ats_nodes[tree_cut_node_indices[frontier_node_position]];
		if (ats_node.triangle_count != 0)
		{
			frontier_node_position++;

			continue;
		}

		unsigned int left_child_index  = ats_node.left_child_index;
		unsigned int right_child_index = left_child_index + 1;
		tree_cut_node_indices.erase(tree_cut_node_indices.begin() + frontier_node_position);
		tree_cut_node_indices.push_back(left_child_index);
		tree_cut_node_indices.push_back(right_child_index);
	}

	effective_tree_cut_size = static_cast<unsigned int>(tree_cut_node_indices.size());
	tree_cut_node_indices.resize(static_cast<size_t>(tree_cut_size), 0xFFFFFFFF);
}

void LightTreeSGBuilder::compute_node_spherical_gaussian(unsigned int node_index, const LightTreeBuilderTrianglesData& triangle_data)
{
	const std::vector<LightTreeATSBuilder::PrefetchedTriangle>& prefetched_triangles = m_light_tree_ats_builder.get_prefetched_triangles();
	const std::vector<LightTreeATSNode>& ats_nodes									 = m_light_tree_ats_builder.get_nodes();
	const LightTreeATSNode& ats_node												 = ats_nodes[node_index];

	LightTreeSGNode& sg_node = m_nodes[node_index];

	if (ats_node.triangle_count == 0)
	{
		// Recurse children if not a leaf

		compute_node_spherical_gaussian(ats_node.left_child_index, triangle_data);
		compute_node_spherical_gaussian(ats_node.left_child_index + 1, triangle_data);

		LightTreeSGNode& left_node	= m_nodes[ats_node.left_child_index];
		LightTreeSGNode& right_node = m_nodes[ats_node.left_child_index + 1];

		float left_weight = 1.0f, right_weight = 1.0f;
		if (left_node.total_power == 0.0f)
			left_weight = 0.0f;
		if (right_node.total_power == 0.0f)
			right_weight = 0.0f;

		if (left_weight == 0.0f && right_weight == 0.0f)
		{
			// Both children are empty
			sg_node.total_power = 0.0f;

			return;
		}

		left_weight	 = left_node.total_power / (left_node.total_power + right_node.total_power);
		right_weight = right_node.total_power / (left_node.total_power + right_node.total_power);

		sg_node.orientation_axis = ats_node.orientation_data.axis;
		sg_node.theta_o			 = ats_node.orientation_data.theta_o;
		sg_node.mean_axis		 = left_weight * left_node.mean_axis + right_weight * right_node.mean_axis;
		sg_node.total_power		 = left_node.total_power + right_node.total_power;
		sg_node.bounds.extend(left_node.bounds);
		sg_node.bounds.extend(right_node.bounds);

		LightTreeSGSpatialLobeBuild lobes[LIGHT_TREE_SG_MAX_SPATIAL_LOBES * 2];
		int lobe_count = 0;
		for (int lobe_index = 0; lobe_index < LIGHT_TREE_SG_MAX_SPATIAL_LOBES; lobe_index++)
		{
			if (left_node.spatial_lobes[lobe_index].power > 0.0)
				lobes[lobe_count++] = left_node.spatial_lobes[lobe_index];
			if (right_node.spatial_lobes[lobe_index].power > 0.0)
				lobes[lobe_count++] = right_node.spatial_lobes[lobe_index];
		}

		LightTreeSGLobeReduction reduction = light_tree_sg_reduce_lobes(lobes, lobe_count, m_spatial_lobe_count);
		for (int lobe_index = 0; lobe_index < LIGHT_TREE_SG_MAX_SPATIAL_LOBES; lobe_index++)
			sg_node.spatial_lobes[lobe_index] = reduction.lobes[lobe_index];
		sg_node.spatial_mean = light_tree_sg_lobes_mean(sg_node.spatial_lobes, m_spatial_lobe_count);

		sg_node.total_emitter_count = left_node.total_emitter_count + right_node.total_emitter_count;
		if (sg_node.total_emitter_count > 0)
		{
			sg_node.energy_average = (left_node.total_emitter_count * left_node.energy_average + right_node.total_emitter_count * right_node.energy_average) /
									 sg_node.total_emitter_count;
			sg_node.energy_variance =
				hippt::max(0.0f, (left_node.total_emitter_count * (left_node.energy_variance + left_node.energy_average * left_node.energy_average) +
								  right_node.total_emitter_count * (right_node.energy_variance + right_node.energy_average * right_node.energy_average)) /
										 sg_node.total_emitter_count -
									 sg_node.energy_average * sg_node.energy_average);
		}

		sg_node.compute_vmf();

		float radius_squared = 0.0f;

		// Computing the bounding sphere of the node from the AABB of the node
		for (int corner_index = 0; corner_index < 8; ++corner_index)
		{
			const float3_t corner = make_float3((corner_index & 1) ? sg_node.bounds.mini.x : sg_node.bounds.maxi.x,
												(corner_index & 2) ? sg_node.bounds.mini.y : sg_node.bounds.maxi.y,
												(corner_index & 4) ? sg_node.bounds.mini.z : sg_node.bounds.maxi.z);

			radius_squared = hippt::max(radius_squared, hippt::length2(corner - sg_node.spatial_mean));
		}

		sg_node.bounding_sphere_radius = hippt::sqrt(radius_squared);

		sg_node.left_child_index = ats_node.left_child_index;
		sg_node.triangle_count	 = 0;
	}
	else
	{
		// Leaf node, compute data necessary for building the spherical gaussian
		sg_node.mean_axis	= make_float3(0.0f, 0.0f, 0.0f);
		sg_node.total_power = 0.0f;
		for (int lobe_index = 0; lobe_index < LIGHT_TREE_SG_MAX_SPATIAL_LOBES; lobe_index++)
			sg_node.spatial_lobes[lobe_index] = LightTreeSGSpatialLobeBuild();

		double3_t sum_position	 = make_double3(0.0, 0.0, 0.0);
		double sum_second_moment = 0.0;
		double sum_power		 = 0.0;

		double sum_energy			= 0.0;
		double sum_energy_squared	= 0.0;
		unsigned int triangle_count = ats_node.triangle_count;
		for (int i = 0; i < ats_node.triangle_count; i++)
		{
			unsigned int linear_emissive_triangle_index = ats_node.first_triangle_index + i;
			int emissive_triangle_index = m_light_tree_ats_builder.bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
			const LightTreeATSBuilder::PrefetchedTriangle& triangle = prefetched_triangles[emissive_triangle_index];

			float3_t p0 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 0]];
			float3_t p1 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 1]];
			float3_t p2 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 2]];

			float3_t e1 = p1 - p0;
			float3_t e2 = p2 - p0;
			const float3_t triangle_variance_diag =
				make_float3((e1.x * e1.x + e2.x * e2.x - e1.x * e2.x) / 18.0f, (e1.y * e1.y + e2.y * e2.y - e1.y * e2.y) / 18.0f,
							(e1.z * e1.z + e2.z * e2.z - e1.z * e2.z) / 18.0f);
			const double centroid_squared = static_cast<double>(triangle.centroid.x) * triangle.centroid.x +
											static_cast<double>(triangle.centroid.y) * triangle.centroid.y +
											static_cast<double>(triangle.centroid.z) * triangle.centroid.z;
			const double triangle_intrinsic_variance = triangle_variance_diag.x + triangle_variance_diag.y + triangle_variance_diag.z;
			sum_second_moment += (centroid_squared + triangle_intrinsic_variance) * triangle.power;

			// 0.5f * triangle normal from the paper
			sg_node.mean_axis += 0.5f * triangle.normal * triangle.power;
			sg_node.total_power += triangle.power;
			sum_power += triangle.power;

			sum_position += make_double3(static_cast<double>(triangle.centroid.x) * triangle.power, static_cast<double>(triangle.centroid.y) * triangle.power,
										 static_cast<double>(triangle.centroid.z) * triangle.power);

			sum_energy += triangle.power;
			sum_energy_squared += hippt::square(triangle.power);
		}

		sg_node.total_emitter_count = triangle_count;

		if (triangle_count == 0)
			return;
		else if (sum_power <= 0.0)
			return;

		double3_t mean_position = sum_position / sum_power;
		double mean_squared		= hippt::dot(mean_position, mean_position);
		double variance			= std::max(0.0, sum_second_moment / sum_power - mean_squared);

		sg_node.energy_average	= sum_energy / triangle_count;
		sg_node.energy_variance = hippt::max(0.0f, static_cast<float>(sum_energy_squared / triangle_count - hippt::square(sum_energy / triangle_count)));
		sg_node.mean_axis /= sg_node.total_power;
		sg_node.spatial_mean = make_float3(static_cast<float>(mean_position.x), static_cast<float>(mean_position.y), static_cast<float>(mean_position.z));
		sg_node.spatial_lobes[0].power	  = sum_power;
		sg_node.spatial_lobes[0].mean	  = mean_position;
		sg_node.spatial_lobes[0].variance = variance;

		float bounding_sphere_radius = 0.0f;
		for (int i = 0; i < ats_node.triangle_count; i++)
		{
			unsigned int linear_emissive_triangle_index = ats_node.first_triangle_index + i;
			int emissive_triangle_index = m_light_tree_ats_builder.bvh_triangle_index_to_emissive_triangle_index(linear_emissive_triangle_index);
			const LightTreeATSBuilder::PrefetchedTriangle& triangle = prefetched_triangles[emissive_triangle_index];

			float3_t p0 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 0]];
			float3_t p1 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 1]];
			float3_t p2 = triangle_data.vertices_positions
							  [triangle_data.triangle_vertex_indices[triangle_data.emissive_triangles_primitive_indices[emissive_triangle_index] * 3 + 2]];

			sg_node.bounds.extend(p0);
			sg_node.bounds.extend(p1);
			sg_node.bounds.extend(p2);

			bounding_sphere_radius =
				hippt::max(bounding_sphere_radius, hippt::max(hippt::max(hippt::length(sg_node.spatial_mean - p0), hippt::length(sg_node.spatial_mean - p1)),
															  hippt::length(sg_node.spatial_mean - p2)));
		}

		// Computes vMF parameters with the mean axis (normalized by the function)
		sg_node.spatial_lobes[0].bounds = sg_node.bounds;
		sg_node.compute_vmf();
		sg_node.triangle_count		   = triangle_count;
		sg_node.first_triangle_index   = ats_node.first_triangle_index;
		sg_node.bounding_sphere_radius = bounding_sphere_radius;
		sg_node.orientation_axis	   = ats_node.orientation_data.axis;
		sg_node.theta_o				   = ats_node.orientation_data.theta_o;
	}
}

/**
 * Merges multiple SG spatial lobes into a single lobe, weighted by their power. The membership bitmask indicates which lobe of 'lobes' to merge.
 */
LightTreeSGSpatialLobeBuild LightTreeSGBuilder::light_tree_sg_lobes_merge(const LightTreeSGSpatialLobeBuild* lobes,
																		  int lobe_count,
																		  unsigned int membership_mask)
{
	LightTreeSGSpatialLobeBuild result;

	for (int lobe_index = 0; lobe_index < lobe_count; lobe_index++)
	{
		if ((membership_mask & (1u << lobe_index)) == 0)
			continue;

		const LightTreeSGSpatialLobeBuild& lobe = lobes[lobe_index];
		result.power += lobe.power;
		result.mean += lobe.power * lobe.mean;
		result.bounds.extend(lobe.bounds);
	}

	if (result.power <= 0.0)
		return result;

	result.mean /= result.power;

	for (int lobe_index = 0; lobe_index < lobe_count; lobe_index++)
	{
		if ((membership_mask & (1u << lobe_index)) == 0)
			continue;

		const LightTreeSGSpatialLobeBuild& lobe = lobes[lobe_index];
		result.variance += lobe.power * (lobe.variance + hippt::length2(lobe.mean - result.mean));
	}

	result.variance /= result.power;

	return result;
}

LightTreeSGSpatialLobeBuild LightTreeSGBuilder::light_tree_sg_merge_lobes(const LightTreeSGSpatialLobeBuild& first, const LightTreeSGSpatialLobeBuild& second)
{
	LightTreeSGSpatialLobeBuild lobes[2] = { first, second };

	return light_tree_sg_lobes_merge(lobes, 2, 0b11);
}

/**
 * Takes a bunch of SG spatial nodes (typically 4: 2 of the left child and 2 of the right child) and reduces them to a target number of lobes (typically 2)
 * for the merged parent node.
 */
LightTreeSGBuilder::LightTreeSGLobeReduction LightTreeSGBuilder::light_tree_sg_reduce_lobes(const LightTreeSGSpatialLobeBuild* lobes,
																							int lobe_count,
																							int target_lobe_count)
{
	LightTreeSGBuilder::LightTreeSGLobeReduction result;

	// Active working set of (merged) lobes that we will iteratively reduce to the target number of lobes
	LightTreeSGSpatialLobeBuild working_lobes[LIGHT_TREE_SG_MAX_SPATIAL_LOBES * 2];
	for (int lobe_index = 0; lobe_index < lobe_count; lobe_index++)
		working_lobes[lobe_index] = lobes[lobe_index];

	// We're going to iteratively merge the two best (lower cost) lobes until we reach the target number of lobes
	while (lobe_count > target_lobe_count)
	{
		double lowest_merge_cost = std::numeric_limits<double>::infinity();

		int first_merge_index  = 0;
		int second_merge_index = 1;

		// Iterate all combinations of pairs of lobes to find the lowest merge cost
		for (int first_index = 0; first_index < lobe_count; first_index++)
		{
			for (int second_index = first_index + 1; second_index < lobe_count; second_index++)
			{
				const double combined_power = working_lobes[first_index].power + working_lobes[second_index].power;

				// Ward's cost: https://en.wikipedia.org/wiki/Ward%27s_method
				const double merge_cost = combined_power > 0.0 ? working_lobes[first_index].power * working_lobes[second_index].power / combined_power *
																	 hippt::length2(working_lobes[first_index].mean - working_lobes[second_index].mean)
															   : 0.0;

				if (merge_cost < lowest_merge_cost)
				{
					lowest_merge_cost  = merge_cost;
					first_merge_index  = first_index;
					second_merge_index = second_index;
				}
			}
		}

		// Merge
		working_lobes[first_merge_index] = light_tree_sg_merge_lobes(working_lobes[first_merge_index], working_lobes[second_merge_index]);

		// Compact the array by removing the second merged lobe
		for (int lobe_index = second_merge_index; lobe_index + 1 < lobe_count; lobe_index++)
			working_lobes[lobe_index] = working_lobes[lobe_index + 1];

		lobe_count--;
	}

	for (int lobe_index = 0; lobe_index < lobe_count; lobe_index++)
		result.lobes[lobe_index] = working_lobes[lobe_index];

	std::sort(result.lobes, result.lobes + lobe_count,
			  [](const LightTreeSGSpatialLobeBuild& first, const LightTreeSGSpatialLobeBuild& second) { return first.power > second.power; });

	return result;
}

float3_t LightTreeSGBuilder::light_tree_sg_lobes_mean(const LightTreeSGSpatialLobeBuild* lobes, int lobe_count)
{
	double total_power = 0.0;
	double3_t mean	   = make_double3(0.0, 0.0, 0.0);

	for (int lobe_index = 0; lobe_index < lobe_count; lobe_index++)
	{
		total_power += lobes[lobe_index].power;
		mean += lobes[lobe_index].power * lobes[lobe_index].mean;
	}

	if (total_power <= 0.0)
		return make_float3(0.0f, 0.0f, 0.0f);

	return make_float3(static_cast<float>(mean.x / total_power), static_cast<float>(mean.y / total_power), static_cast<float>(mean.z / total_power));
}

void LightTreeSGBuilder::cleanup()
{
	m_light_tree_ats_builder.cleanup();
	m_nodes.clear();
}

LightTreeATSBuilderOptions& LightTreeSGBuilder::get_build_options()
{
	return m_light_tree_ats_builder.get_build_options();
}

int LightTreeSGBuilder::get_spatial_lobe_count() const
{
	return m_spatial_lobe_count;
}

void LightTreeSGBuilder::set_spatial_lobe_count(int spatial_lobe_count)
{
	m_spatial_lobe_count = hippt::clamp(1, LIGHT_TREE_SG_MAX_SPATIAL_LOBES, spatial_lobe_count);
}

int LightTreeSGBuilder::get_tree_cut_size() const
{
	return m_first_tree_cut_size;
}

void LightTreeSGBuilder::set_tree_cut_size(int tree_cut_size)
{
	m_first_tree_cut_size = hippt::clamp(1, 2000000000, tree_cut_size);
}

int LightTreeSGBuilder::get_second_tree_cut_size() const
{
	return m_second_tree_cut_size;
}

void LightTreeSGBuilder::set_second_tree_cut_size(int second_tree_cut_size)
{
	m_second_tree_cut_size = hippt::clamp(1, IlluminationAwareKDTreeMaximumLightCutSize, second_tree_cut_size);
}

unsigned int LightTreeSGBuilder::get_effective_second_tree_cut_size() const
{
	return m_effective_second_tree_cut_size;
}

const std::vector<unsigned int>& LightTreeSGBuilder::get_tree_cut_node_indices() const
{
	return m_first_tree_cut_node_indices;
}

const std::vector<unsigned int>& LightTreeSGBuilder::get_second_tree_cut_node_indices() const
{
	return m_second_tree_cut_node_indices;
}
