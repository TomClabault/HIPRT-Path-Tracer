/**
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_CELLS_LIGHT_DISTRIBUTIONS_SOA_DEVICE_H
#define REGIR_CELLS_LIGHT_DISTRIBUTIONS_SOA_DEVICE_H

#ifndef __KERNELCC__
template <template <typename> typename DataContainer>
struct ReGIRCellsLightDistributionsSoAHost;
#endif // #ifndef __KERNELCC__

struct ReGIRCellsLightDistributionsSoADevice
{
	using ReGIRCellsLightDistributionsMeshIndicesPackingType = unsigned long long int;

	static constexpr unsigned int NO_AVAILABLE_LIGHT_DISTRIBUTION = 0xFFFFFFFF;

	HIPRT_DEVICE float get_PDF(unsigned int hash_grid_cell_index, unsigned int CDF_table_index) const
	{
		unsigned int offset = light_distribution_offsets[hash_grid_cell_index];

		if (CDF_table_index == 0)
			return all_cdfs[offset + CDF_table_index] / 65535.0f;
		else
		{
			return (all_cdfs[offset + CDF_table_index] - all_cdfs[offset + CDF_table_index - 1]) / 65535.0f;
		}
	}

	HIPRT_DEVICE unsigned int get_emissive_mesh_index(unsigned hash_grid_cell_index, unsigned int CDF_table_index) const
	{
		// Mesh indices are tightly packed in 64 bit integer elements
		// (or however many bits ReGIRCellsLightDistributionsMeshIndicesPackingType is)
		// so we need to extract the right bits from the right element and this function
		// does the unpacking

		using PackingType = ReGIRCellsLightDistributionsMeshIndicesPackingType;

		unsigned int bit_offset_start_in_element = (CDF_table_index * bits_per_mesh_index) % (sizeof(PackingType) * 8);
		unsigned int element_index				 = CDF_table_index * bits_per_mesh_index / (sizeof(PackingType) * 8);

		if (bit_offset_start_in_element + bits_per_mesh_index > sizeof(PackingType) * 8)
		{
			// If the mesh index is straddling two differents elements

			unsigned int bits_in_first_element	= sizeof(PackingType) * 8 - bit_offset_start_in_element;
			unsigned int bits_in_second_element = bits_per_mesh_index - bits_in_first_element;

			unsigned int bits_in_first_element_mask	 = (1 << bits_in_first_element) - 1;
			unsigned int bits_in_second_element_mask = (1 << bits_in_second_element) - 1;

			unsigned int first_part = (emissive_meshes_indices_packed[mesh_indices_offsets[hash_grid_cell_index] + element_index] >>
									   bit_offset_start_in_element) &
									  bits_in_first_element_mask;
			unsigned int second_part = (emissive_meshes_indices_packed[mesh_indices_offsets[hash_grid_cell_index] + element_index + 1]) &
									   bits_in_second_element_mask;

			return first_part | (second_part << bits_in_first_element);
		}
		else
			// Packed mesh index not straddling, just need to fetch the bits
			return (emissive_meshes_indices_packed[mesh_indices_offsets[hash_grid_cell_index] + element_index] >> bit_offset_start_in_element) &
				   ((1 << bits_per_mesh_index) - 1);
	}

	unsigned short int* all_cdfs = nullptr;

	// How many entries in the light distribution of each cell
	unsigned short int* light_distribution_sizes = nullptr;
	// At which index does each light distribution start in the 'all_cdfs' buffer
	unsigned int* light_distribution_offsets = nullptr;

private:
	// How many bits are needed to store one mesh index
	unsigned int bits_per_mesh_index = 0;
	// For each cell, how many elements of type ReGIRCellsLightDistributionsMeshIndicesPackingType
	// are used to store the mesh indices associated with that cell
	unsigned int mesh_indices_element_count_per_cell = 0;

	// Contains the indices of the meshes associated with the entries of the alias table
	//
	// For example, if the alias tables are 4 entries long but there are 20 emissive
	// meshes in the scene, only 4 of those meshes are going to be retained in the alias
	// table at alias table indices 0, 1, 2, and 3.
	//
	// We're going to need this buffer to map the alias table indices 0, 1, 2 and 3 to
	// the true mesh indices within the scene (could be 8, 5, 12, 19 for example, completely
	// arbitrary)
	//
	// Each mesh index only consumes the right number of bits to be represented depending on the
	// number of emissive meshes in the scene.
	ReGIRCellsLightDistributionsMeshIndicesPackingType* emissive_meshes_indices_packed = nullptr;
	// For each grid cell, offset in the 'emissive_meshes_indices_packed' buffer where the packed emissive
	// meshes indices start
	unsigned int* mesh_indices_offsets = nullptr;

#ifndef __KERNELCC__
	template <template <typename> typename OtherContainer>
	friend struct ReGIRCellsLightDistributionsSoAHost;
#endif // #ifndef __KERNELCC__
};

#endif // #ifndef REGIR_CELLS_LIGHT_DISTRIBUTIONS_SOA_DEVICE_H
