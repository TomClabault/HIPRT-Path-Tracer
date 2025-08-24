/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_EMISSIVE_MESH_ALIAS_TABLE_GPU_DATA_H
#define RENDERER_EMISSIVE_MESH_ALIAS_TABLE_GPU_DATA_H

#include "Device/includes/EmissiveMeshesDataDevice.h"

#include "Renderer/CPUGPUCommonDataStructures/AliasTableHost.h"

#include "Scene/SceneParser.h"

/**
 * Contains an alias table for sampling emissive *meshes* in the scene according 
 * to their emissive power. Textured emitter are not considered
 * 
 * This also contains the alias tables of all meshes in a single linear buffer which can
 * be used to sample an emissive triangle within a mesh according to its power.
 */
template <template <typename> typename DataContainer>
struct EmissiveMeshesAliasTablesHost
{
	template <typename T>
	static void upload_to_device_buffer(DataContainer<T>& device_buffer, const std::vector<T>& input_data)
	{
		if constexpr (std::is_same<DataContainer<T>, std::vector<T>>::value)
			device_buffer = input_data;
		else if constexpr (std::is_same<DataContainer<T>, OrochiBuffer<T>>::value)
		{
			if (device_buffer.size() == 0)
				device_buffer.resize(input_data.size());
			device_buffer.upload_data(input_data);
		}
	}

	template <typename T>
	static void upload_to_device_buffer_partial(DataContainer<T>& device_buffer, const std::vector<T>& input_data, size_t start_index, size_t element_count)
	{
		if constexpr (std::is_same<DataContainer<T>, std::vector<T>>::value)
			std::copy(input_data.begin(), input_data.begin() + element_count, device_buffer.begin() + start_index);
		else if constexpr (std::is_same<DataContainer<T>, OrochiBuffer<T>>::value)
			device_buffer.upload_data_partial(start_index, input_data.data(), element_count);
	}

	void load_from_emissive_meshes(const Scene& parsed_scene)
	{
		const std::vector<ParsedEmissiveMesh>& emissive_meshes = parsed_scene.parsed_emissive_meshes.emissive_meshes;

		std::vector<unsigned int> offsets(emissive_meshes.size());
		std::vector<unsigned int> alias_tables_sizes(emissive_meshes.size());
		offsets[0] = 0u;

		unsigned int total_alias_tables_entries_count = 0;
		for (int i = 0; i < emissive_meshes.size(); i++)
		{
			if (i > 0)
				offsets.at(i) = offsets.at(i - 1) + emissive_meshes.at(i - 1).emissive_triangle_count;
			alias_tables_sizes.at(i) = emissive_meshes.at(i).emissive_triangle_count;

			total_alias_tables_entries_count += emissive_meshes[i].emissive_triangle_count;
		}

		upload_to_device_buffer(m_offsets_into_alias_table, offsets);
		upload_to_device_buffer(m_meshes_alias_tables_sizes, alias_tables_sizes);

		m_alias_tables_probas.resize(total_alias_tables_entries_count);
		m_alias_tables_aliases.resize(total_alias_tables_entries_count);

		m_binned_faces_indices.resize(total_alias_tables_entries_count);
		m_binned_faces_start_index.resize(emissive_meshes.size() * ParsedEmissiveMesh::BinningNormals.size());
		m_binned_faces_counts.resize(emissive_meshes.size() * ParsedEmissiveMesh::BinningNormals.size());
		m_binned_faces_total_power.resize(emissive_meshes.size() * ParsedEmissiveMesh::BinningNormals.size());
		m_binned_faces_mesh_face_index_to_bin_index.resize(total_alias_tables_entries_count);

		unsigned int cumulative_start_index = 0;
		for (int mesh_index = 0; mesh_index < emissive_meshes.size(); mesh_index++)
		{
			upload_to_device_buffer_partial<float>(m_alias_tables_probas, emissive_meshes[mesh_index].alias_probas, cumulative_start_index, emissive_meshes[mesh_index].emissive_triangle_count);
			upload_to_device_buffer_partial<int>(m_alias_tables_aliases, emissive_meshes[mesh_index].alias_aliases, cumulative_start_index, emissive_meshes[mesh_index].emissive_triangle_count);
			
			// Uploading binned faces data
			upload_to_device_buffer_partial<unsigned int>(m_binned_faces_indices, emissive_meshes[mesh_index].binned_faces_indices, cumulative_start_index, emissive_meshes[mesh_index].emissive_triangle_count);
			upload_to_device_buffer_partial<unsigned int>(m_binned_faces_start_index, emissive_meshes[mesh_index].binned_faces_start_index, mesh_index * ParsedEmissiveMesh::BinningNormals.size(), ParsedEmissiveMesh::BinningNormals.size());
			upload_to_device_buffer_partial<unsigned int>(m_binned_faces_counts, emissive_meshes[mesh_index].binned_faces_counts, mesh_index * ParsedEmissiveMesh::BinningNormals.size(), ParsedEmissiveMesh::BinningNormals.size());
			upload_to_device_buffer_partial<float>(m_binned_faces_total_power, emissive_meshes[mesh_index].binned_faces_total_power, mesh_index * ParsedEmissiveMesh::BinningNormals.size(), ParsedEmissiveMesh::BinningNormals.size());
			upload_to_device_buffer_partial<unsigned int>(m_binned_faces_mesh_face_index_to_bin_index, emissive_meshes[mesh_index].binned_faces_mesh_face_index_to_bin_index, cumulative_start_index, emissive_meshes[mesh_index].emissive_triangle_count);
			
			cumulative_start_index += emissive_meshes[mesh_index].emissive_triangle_count;
		}

		// Now computing an alias table on all the meshes of the scene to be able to sample a
		// mesh according to its total power
		float total_meshes_power_sum = 0.0f;
		std::vector<float> emissive_meshes_power(emissive_meshes.size());
		for (int i = 0; i < emissive_meshes.size(); i++)
		{
			emissive_meshes_power[i] = emissive_meshes[i].total_mesh_emissive_power;
			total_meshes_power_sum += emissive_meshes[i].total_mesh_emissive_power;
		}

		std::vector<float> meshes_alias_table_probas;
		std::vector<int> meshes_alias_table_aliases;
		Utils::compute_alias_table(emissive_meshes_power, total_meshes_power_sum, meshes_alias_table_probas, meshes_alias_table_aliases);

		// Uploading the alias table of the meshes
		upload_to_device_buffer(m_meshes_alias_table.aliases, meshes_alias_table_aliases);
		upload_to_device_buffer(m_meshes_alias_table.probas, meshes_alias_table_probas);
		m_meshes_alias_table.size = meshes_alias_table_aliases.size();
		m_meshes_alias_table.sum_elements = total_meshes_power_sum;

		std::vector<float3> meshes_average_points(parsed_scene.parsed_emissive_meshes.emissive_meshes.size());
		for (int i = 0; i < meshes_average_points.size(); i++)
			meshes_average_points[i] = parsed_scene.parsed_emissive_meshes.emissive_meshes[i].average_mesh_point;

		upload_to_device_buffer(m_meshes_average_points, meshes_average_points);
		upload_to_device_buffer(m_meshes_total_power, emissive_meshes_power);

		// Uploading some more data needed for sampling at runtime
		std::vector<float> meshes_PDFs(emissive_meshes.size());
		for (int i = 0; i < emissive_meshes.size(); i++)
			meshes_PDFs[i] = emissive_meshes[i].total_mesh_emissive_power / total_meshes_power_sum;

		upload_to_device_buffer(m_meshes_emissive_triangles_PDFs, parsed_scene.parsed_emissive_meshes.emissive_meshes_triangles_PDFs);
		upload_to_device_buffer(m_meshes_emissive_triangles_indices, parsed_scene.emissive_triangles_primitive_indices);
		upload_to_device_buffer(m_meshes_PDFs, meshes_PDFs);
	}

	EmissiveMeshesDataDevice to_device()
	{
		EmissiveMeshesDataDevice out;

		out.alias_table_count = m_offsets_into_alias_table.size();
		out.offsets = m_offsets_into_alias_table.data();
		out.individual_alias_tables_sizes = m_meshes_alias_tables_sizes.data();

		out.meshes_alias_table = m_meshes_alias_table.to_device();
		out.meshes_PDFs = m_meshes_PDFs.data();
		out.meshes_average_points = m_meshes_average_points.data();
		out.meshes_total_power = m_meshes_total_power.data();

		out.binned_faces_indices = m_binned_faces_indices.data();
		out.binned_faces_start_index = m_binned_faces_start_index.data();
		out.binned_faces_counts = m_binned_faces_counts.data();
		out.binned_faces_total_power = m_binned_faces_total_power.data();
		out.binned_faces_mesh_face_index_to_bin_index = m_binned_faces_mesh_face_index_to_bin_index.data();

		out.alias_tables_aliases = m_alias_tables_aliases.data();
		out.alias_tables_probas = m_alias_tables_probas.data();
		out.meshes_emissive_triangles_PDFs = m_meshes_emissive_triangles_PDFs.data();
		out.meshes_triangle_indices = m_meshes_emissive_triangles_indices.data();

		return out;
	}

	// Alias table for sampling a single mesh in the scene proportional to its power
	AliasTableHost<DataContainer> m_meshes_alias_table;
	// Offsets of the individual meshes' alias tables in the big linear
	// buffer that contains all the alias tables of the meshes
	DataContainer<unsigned int> m_offsets_into_alias_table;
	// How big are the alias tables of each mesh for sampling an emissive triangle within each mesh
	DataContainer<unsigned int> m_meshes_alias_tables_sizes;
	// PDF that the 'meshes_alias_table' samples a given mesh index
	DataContainer<float> m_meshes_PDFs;
	DataContainer<float3> m_meshes_average_points;
	DataContainer<float> m_meshes_total_power;

	DataContainer<unsigned int> m_binned_faces_indices;
	DataContainer<unsigned int> m_binned_faces_start_index;
	DataContainer<unsigned int> m_binned_faces_counts;
	DataContainer<float> m_binned_faces_total_power;
	DataContainer<unsigned int> m_binned_faces_mesh_face_index_to_bin_index;

	// Concatenation of the alias_probas of the alias tables of all emissive meshes of the scene
	DataContainer<float> m_alias_tables_probas;
	// Same for the aliases
	DataContainer<int> m_alias_tables_aliases;
	DataContainer<float> m_meshes_emissive_triangles_PDFs;
	DataContainer<int> m_meshes_emissive_triangles_indices;
};

#endif
