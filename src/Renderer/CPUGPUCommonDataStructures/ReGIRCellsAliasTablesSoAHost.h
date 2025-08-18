/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_CELLS_ALIAS_TABLES_SOA_HOST_H
#define REGIR_CELLS_ALIAS_TABLES_SOA_HOST_H

#include "Device/includes/ReSTIR/ReGIR/CellsAliasTablesSoADevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

// TODO maybe a CDF would be fast enough and would use less memory (probably? because with all the packing we can do on the alias table this may not be true / worth it)
template <template <typename> typename DataContainer>
using ReGIRCellsAliasTablesSoAHostInternal = GenericSoA<DataContainer,
	// TODO this can probably be uint 16 unorm?
	float,  // alias table probas
	// TODO, this can be an unsigned char if the alias table is 256 max elements
	int, // alias table aliases
	// TODO we can probably do something a bit clever to be able to recompute the alias table PDF on the fly without having
	//		to store a full other buffer just for that
	float, // PDFs that each cell samples a given mesh index within its own cell-alias-table
	unsigned int // Indices of the emissive meshes associated with the entries of the alias table at each cell
>;

enum ReGIRCellsAliasTablesSoAHostBuffers
{
	REGIR_CELLS_ALIAS_TABLES_PROBAS,
	REGIR_CELLS_ALIAS_TABLES_ALIASES,
	REGIR_CELLS_ALIAS_PDFS,
	REGIR_CELLS_EMISSIVE_MESHES_INDICES
};

template <template <typename> typename DataContainer>
struct ReGIRCellsAliasTablesSoAHost
{
	void resize(unsigned int new_number_of_cells, unsigned int alias_tables_size)
	{
		soa.resize(new_number_of_cells * alias_tables_size);

		m_alias_table_size = alias_tables_size;
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

	ReGIRCellsAliasTablesSoADevice to_device(const HIPRTRenderData& render_data)
	{
		ReGIRCellsAliasTablesSoADevice cells_alias_tables;

		cells_alias_tables.all_alias_tables_probas = soa.template get_buffer_data_ptr<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_ALIAS_TABLES_PROBAS>();
		cells_alias_tables.all_alias_tables_aliases = soa.template get_buffer_data_ptr<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_ALIAS_TABLES_ALIASES>();
		cells_alias_tables.all_alias_tables_PDFs = soa.template get_buffer_data_ptr<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_ALIAS_PDFS>();
		cells_alias_tables.emissive_meshes_indices = soa.template get_buffer_data_ptr<ReGIRCellsAliasTablesSoAHostBuffers::REGIR_CELLS_EMISSIVE_MESHES_INDICES>();
		// The size of the light distributions at each cell is the minimum between the target alias table
		// size and the number of emissive meshes in the scene
		cells_alias_tables.alias_table_size = hippt::min(m_alias_table_size, render_data.buffers.emissive_meshes_alias_tables.alias_table_count);

		return cells_alias_tables;
	}

	ReGIRCellsAliasTablesSoAHostInternal<DataContainer> soa;

	unsigned int m_alias_table_size = 0;
};

#endif
