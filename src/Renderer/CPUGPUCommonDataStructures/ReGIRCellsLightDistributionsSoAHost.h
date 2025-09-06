/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_CELLS_LIGHT_DISTRIBUTIONS_SOA_HOST_H
#define REGIR_CELLS_LIGHT_DISTRIBUTIONS_SOA_HOST_H

#include "Device/includes/ReSTIR/ReGIR/CellsLightDistributionsSoADevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

using ReGIRCellsLightDistributionsMeshIndicesPackingType = ReGIRCellsLightDistributionsSoADevice::ReGIRCellsLightDistributionsMeshIndicesPackingType;

class ReGIRCellsLightDistributionsHostUtils
{
public:
	static unsigned int get_bits_per_packed_mesh_index(unsigned int emissive_mesh_count)
	{
		return std::ceil(std::log2(emissive_mesh_count)) + 1;
	}

	static unsigned int get_packed_mesh_indices_count_per_cell(unsigned int emissive_mesh_count, unsigned int light_distribution_size)
	{
		unsigned int bits_per_mesh_index = get_bits_per_packed_mesh_index(emissive_mesh_count);
		// How many ReGIRCellsLightDistributionsMeshIndicesPackingType elements do we need per cell to store all the mesh indices of that cell
		unsigned int mesh_indices_count_per_cell = std::ceil(bits_per_mesh_index * light_distribution_size / (float)(sizeof(ReGIRCellsLightDistributionsMeshIndicesPackingType) * 8));

		return mesh_indices_count_per_cell;
	}

	static std::vector<ReGIRCellsLightDistributionsMeshIndicesPackingType> pack_mesh_indices(const std::vector<unsigned int>::const_iterator& sorted_mesh_indices_start, unsigned int emissive_mesh_count, unsigned int light_distribution_size)
	{
		std::vector<ReGIRCellsLightDistributionsMeshIndicesPackingType> packed(ReGIRCellsLightDistributionsHostUtils::get_packed_mesh_indices_count_per_cell(emissive_mesh_count, light_distribution_size), 0);

		unsigned int bits_per_mesh_index = ReGIRCellsLightDistributionsHostUtils::get_bits_per_packed_mesh_index(emissive_mesh_count);
		constexpr unsigned int BITS_PER_ELEMENT = sizeof(ReGIRCellsLightDistributionsMeshIndicesPackingType) * 8;
		for (int mesh_index = 0; mesh_index < light_distribution_size; mesh_index++)
		{
			unsigned int sorted_mesh_index = *(sorted_mesh_indices_start + mesh_index);

			// Which element we're going to pack that mesh index into
			unsigned int element_index = mesh_index * bits_per_mesh_index / BITS_PER_ELEMENT;
			unsigned int bit_offset_start_in_element = (mesh_index * bits_per_mesh_index) % BITS_PER_ELEMENT;

			if (bit_offset_start_in_element + bits_per_mesh_index > BITS_PER_ELEMENT)
			{
				// If the mesh index is straddling two differents elements

				unsigned int bits_in_first_element = BITS_PER_ELEMENT - bit_offset_start_in_element;
				unsigned int bits_in_second_element = bits_per_mesh_index - bits_in_first_element;
				unsigned int bits_in_first_element_mask = (1 << bits_in_first_element) - 1;
				unsigned int bits_in_second_element_mask = (1 << bits_in_second_element) - 1;

				ReGIRCellsLightDistributionsMeshIndicesPackingType first_part = static_cast<ReGIRCellsLightDistributionsMeshIndicesPackingType>(sorted_mesh_index & bits_in_first_element_mask) << bit_offset_start_in_element;
				ReGIRCellsLightDistributionsMeshIndicesPackingType second_part = (sorted_mesh_index >> bits_in_first_element) & bits_in_second_element_mask;

				packed[element_index] |= first_part;
				packed[element_index + 1] |= second_part;
			}
			else
			{
				// If the mesh index is fully contained in a single element

				ReGIRCellsLightDistributionsMeshIndicesPackingType bitmask = (1 << bits_per_mesh_index) - 1;
				ReGIRCellsLightDistributionsMeshIndicesPackingType bits = (sorted_mesh_index & bitmask) << bit_offset_start_in_element;

				packed[element_index] |= bits;
			}
		}

		return packed;
	}
};

// TODO maybe a CDF would be fast enough and would use less memory (probably? because with all the packing we can do on the alias table this may not be true / worth it)
template <template <typename> typename DataContainer>
using ReGIRCellsLightDistributionsSoAHostInternal = GenericSoA<DataContainer,
	unsigned short int,		// CDF as normalized unsigned short int (0-65535)
	ReGIRCellsLightDistributionsMeshIndicesPackingType	// Indices of the emissive meshes associated with each entries of the CDF
														// Only the right number of bits are used (so if we have 1000 emissive meshes,
														// only 10 bits are used). These bits are tightly packed in 64 bit integer
														// (so we may have some mesh index striding two differents 64 bit integers
														// sometimes)
>;

enum ReGIRCellsLightDistributionsSoAHostBuffers
{
	REGIR_CELLS_LIGHT_DISTRIBUTIONS_CDF,
	REGIR_CELLS_LIGHT_DISTRIBUTIONS_MESHES_INDICES
};

template <template <typename> typename DataContainer>
struct ReGIRCellsLightDistributionsSoAHost
{
	void resize(size_t new_number_of_cells, unsigned int light_distribution_size, unsigned int emissive_meshes_count)
	{
		soa.resize(new_number_of_cells * light_distribution_size, { REGIR_CELLS_LIGHT_DISTRIBUTIONS_MESHES_INDICES });

		soa.template get_buffer<REGIR_CELLS_LIGHT_DISTRIBUTIONS_MESHES_INDICES>().resize(new_number_of_cells * ReGIRCellsLightDistributionsHostUtils::get_packed_mesh_indices_count_per_cell(emissive_meshes_count, light_distribution_size));

		m_light_distribution_size = light_distribution_size;
		m_emissive_mesh_count = emissive_meshes_count;
	}

	void free()
	{
		soa.free();
	}

	std::size_t get_byte_size() const
	{
		return soa.get_byte_size();
	}

	unsigned int size() const
	{
		return soa.size();
	}

	ReGIRCellsLightDistributionsSoADevice to_device(const HIPRTRenderData& render_data)
	{
		ReGIRCellsLightDistributionsSoADevice cells_light_distributions;

		cells_light_distributions.all_cdfs = soa.template get_buffer_data_ptr<ReGIRCellsLightDistributionsSoAHostBuffers::REGIR_CELLS_LIGHT_DISTRIBUTIONS_CDF>();
		cells_light_distributions.emissive_meshes_indices_packed = soa.template get_buffer_data_ptr<ReGIRCellsLightDistributionsSoAHostBuffers::REGIR_CELLS_LIGHT_DISTRIBUTIONS_MESHES_INDICES>();

		cells_light_distributions.light_distribution_size = hippt::min(m_light_distribution_size, render_data.buffers.emissive_meshes_data.alias_table_count);
		cells_light_distributions.mesh_indices_element_count_per_cell = ReGIRCellsLightDistributionsHostUtils::get_packed_mesh_indices_count_per_cell(m_emissive_mesh_count, cells_light_distributions.light_distribution_size);
		cells_light_distributions.bits_per_mesh_index = ReGIRCellsLightDistributionsHostUtils::get_bits_per_packed_mesh_index(m_emissive_mesh_count);

		return cells_light_distributions;
	}

	ReGIRCellsLightDistributionsSoAHostInternal<DataContainer> soa;

	unsigned int m_light_distribution_size = 0;
	unsigned int m_emissive_mesh_count = 0;
};

#endif
