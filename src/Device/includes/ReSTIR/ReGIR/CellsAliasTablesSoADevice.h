/**
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef REGIR_CELLS_ALIAS_TABLES_SOA_DEVICE_H
#define REGIR_CELLS_ALIAS_TABLES_SOA_DEVICE_H

struct ReGIRCellsAliasTablesSoADevice
{
	// These buffers are all NUMBER OF REGIR CELLS * ALIAS TABLE SIZE big
	// Contains the probas of all the alias tables of all the cells concatenated in one buffer
	float* all_alias_tables_probas = nullptr;
	// Same for the aliases
	int* all_alias_tables_aliases = nullptr;
	// Same for the PDFs
	float* all_alias_tables_PDFs = nullptr;
	// Contains the indices of the meshes associated with the entries of the alias table
	//
	// For example, if the alias tables are 4 entries long but there are 20 emissive
	// meshes in the scene, only 4 of those meshes are going to be retained in the alias
	// table at alias table indices 0, 1, 2, and 3.
	//
	// We're going to need this buffer to map the alias table indices 0, 1, 2 and 3 to
	// the true mesh indices within the scene (could be 8, 5, 12, 19 for example, completely
	// arbitrary)
	unsigned int* emissive_meshes_indices = nullptr;

	// How many entries in the alias tables of each cell
	unsigned int alias_table_size = 1;
};

#endif
