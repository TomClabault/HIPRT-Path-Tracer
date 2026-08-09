/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/LightTree/LightTreeSGBuilderNISML.h"

void LightTreeSGBuilderNISML::build_lookup(const std::vector<LightTreeATSNode>& ats_nodes,
										   const std::vector<unsigned int>& bit_trails,
										   const std::vector<int>& triangle_indices,
										   const std::vector<int>& emissive_triangles_primitive_indices,
										   unsigned int total_scene_triangle_count)
{
	unsigned int invalid_cluster_slot = 0xFF;
	unsigned int invalid_node_index	  = 0xFFFFFFFF;

	triangle_to_neural_cluster.assign(total_scene_triangle_count, static_cast<unsigned char>(invalid_cluster_slot));
	neural_cluster_node_depths.assign(effective_tree_cut_size_neural_many_lights, 0);

	std::vector<unsigned int> node_to_cluster_slot(ats_nodes.size(), invalid_cluster_slot);
	for (unsigned int cluster_slot = 0; cluster_slot < effective_tree_cut_size_neural_many_lights; cluster_slot++)
	{
		unsigned int node_index = tree_cut_node_indices_neural_many_lights[cluster_slot];
		if (node_index != invalid_node_index && node_index < ats_nodes.size())
			node_to_cluster_slot[node_index] = cluster_slot;
	}

	for (unsigned int linear_triangle_index = 0; linear_triangle_index < bit_trails.size(); linear_triangle_index++)
	{
		if (linear_triangle_index >= triangle_indices.size())
			continue;

		int emissive_triangle_index = triangle_indices[linear_triangle_index];
		if (emissive_triangle_index < 0 || static_cast<unsigned int>(emissive_triangle_index) >= emissive_triangles_primitive_indices.size())
			continue;

		int global_triangle_index = emissive_triangles_primitive_indices[emissive_triangle_index];
		if (global_triangle_index < 0 || static_cast<unsigned int>(global_triangle_index) >= total_scene_triangle_count)
			continue;

		unsigned int node_index = 0;
		unsigned int depth		= 0;
		while (node_index < ats_nodes.size())
		{
			unsigned int cluster_slot = node_to_cluster_slot[node_index];
			if (cluster_slot != invalid_cluster_slot)
			{
				triangle_to_neural_cluster[global_triangle_index] = static_cast<unsigned char>(cluster_slot);
				neural_cluster_node_depths[cluster_slot]		  = static_cast<unsigned char>(depth);

				break;
			}

			if (ats_nodes[node_index].triangle_count != 0 || depth >= sizeof(unsigned int) * 8)
				break;

			unsigned int left_child_index = ats_nodes[node_index].left_child_index;
			node_index					  = (bit_trails[linear_triangle_index] & (1u << depth)) == 0 ? left_child_index : left_child_index + 1;
			depth++;
		}
	}
}

void LightTreeSGBuilderNISML::free()
{
	m_device_cluster_node_indices_buffer.free();
	m_device_triangle_to_cluster_buffer.free();
	m_device_cluster_node_depths_buffer.free();
	tree_cut_node_indices_neural_many_lights.clear();
	effective_tree_cut_size_neural_many_lights = 0;
	triangle_to_neural_cluster.clear();
	neural_cluster_node_depths.clear();
}

size_t LightTreeSGBuilderNISML::get_VRAM_usage_bytes() const
{
	return m_device_cluster_node_indices_buffer.get_byte_size() + m_device_triangle_to_cluster_buffer.get_byte_size() +
		   m_device_cluster_node_depths_buffer.get_byte_size();
}
