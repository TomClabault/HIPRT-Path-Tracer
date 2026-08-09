/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_LIGHT_TREE_SG_BUILDER_NISML_H
#define RENDERER_LIGHT_TREE_SG_BUILDER_NISML_H

#include "Device/includes/Neural/NISML/NISMLDevice.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/LightTree/LightTreeATSNode.h"

#include <vector>

struct LightTreeSGBuilderNISML
{
	void build_lookup(const std::vector<LightTreeATSNode>& ats_nodes,
					  const std::vector<unsigned int>& bit_trails,
					  const std::vector<int>& triangle_indices,
					  const std::vector<int>& emissive_triangles_primitive_indices,
					  unsigned int total_scene_triangle_count);

	template <template <typename> typename DataContainer>
	void to_device(NISMLDevice& nisml_device)
	{
		if constexpr (std::is_same_v<DataContainer<int>, std::vector<int>>)
		{
			nisml_device.cluster_node_indices = tree_cut_node_indices_neural_many_lights.data();
			nisml_device.triangle_to_cluster  = triangle_to_neural_cluster.data();
			nisml_device.cluster_node_depths  = neural_cluster_node_depths.data();
		}
		else
		{
			if (tree_cut_node_indices_neural_many_lights.empty())
				m_device_cluster_node_indices_buffer.free();
			else if (m_device_cluster_node_indices_buffer.get_byte_size() != tree_cut_node_indices_neural_many_lights.size() * sizeof(unsigned int))
				m_device_cluster_node_indices_buffer = OrochiBuffer<unsigned int>(tree_cut_node_indices_neural_many_lights);
			if (triangle_to_neural_cluster.empty())
				m_device_triangle_to_cluster_buffer.free();
			else if (m_device_triangle_to_cluster_buffer.get_byte_size() != triangle_to_neural_cluster.size() * sizeof(unsigned char))
				m_device_triangle_to_cluster_buffer = OrochiBuffer<unsigned char>(triangle_to_neural_cluster);
			if (neural_cluster_node_depths.empty())
				m_device_cluster_node_depths_buffer.free();
			else if (m_device_cluster_node_depths_buffer.get_byte_size() != neural_cluster_node_depths.size() * sizeof(unsigned char))
				m_device_cluster_node_depths_buffer = OrochiBuffer<unsigned char>(neural_cluster_node_depths);

			nisml_device.cluster_node_indices = m_device_cluster_node_indices_buffer.data();
			nisml_device.triangle_to_cluster  = m_device_triangle_to_cluster_buffer.data();
			nisml_device.cluster_node_depths  = m_device_cluster_node_depths_buffer.data();
		}

		nisml_device.cluster_count = effective_tree_cut_size_neural_many_lights;
	}

	void free();

	size_t get_VRAM_usage_bytes() const;

	std::vector<unsigned int> tree_cut_node_indices_neural_many_lights;
	unsigned int effective_tree_cut_size_neural_many_lights = 0;
	std::vector<unsigned char> triangle_to_neural_cluster;
	std::vector<unsigned char> neural_cluster_node_depths;
	int tree_cut_size_neural_many_lights = 64;

private:
	OrochiBuffer<unsigned int> m_device_cluster_node_indices_buffer;
	OrochiBuffer<unsigned char> m_device_triangle_to_cluster_buffer;
	OrochiBuffer<unsigned char> m_device_cluster_node_depths_buffer;
};

#endif
