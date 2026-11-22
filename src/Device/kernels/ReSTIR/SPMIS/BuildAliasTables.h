/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_RESTIR_SPMIS_BUILD_ALIAS_TABLES_H
#define KERNELS_RESTIR_SPMIS_BUILD_ALIAS_TABLES_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/RenderData.h"

// Same maximum number of non-zero pixels per cell as ReSTIR_PT_SPMIS_BuildCDFs, which scans a cell with one block of 1024 threads
#define RESTIR_PT_SPMIS_ALIAS_TABLE_MAX_CELL_SIZE 1024u

/**
 * Writes the alias table entry 'index' of a cell: the probability of keeping the bin and the index (in the cell) of its alias
 */
HIPRT_DEVICE void spmis_write_alias_table_entry(float* cell_alias_table, unsigned int index, float probability, unsigned int alias)
{
	cell_alias_table[index * RESTIR_PT_SPMIS_UNION_DOMAIN_ALIAS_TABLE_FLOAT_COUNT + 0] = probability;
	// Exact as a float: the cells have at most RESTIR_PT_SPMIS_ALIAS_TABLE_MAX_CELL_SIZE pixels
	cell_alias_table[index * RESTIR_PT_SPMIS_UNION_DOMAIN_ALIAS_TABLE_FLOAT_COUNT + 1] = static_cast<float>(alias);
}

/**
 * Builds the alias table of one cell from its importances scaled so that they sum to 'count' (an importance of 1 is the average), with the sweeping
 * construction: one pointer runs over the light entries (scaled importance <= 1), one over the heavy entries (> 1). The current heavy entry fills
 * the bins of the light entries until its remaining mass drops to 1 or below, it then becomes a bin itself, filled by the next heavy entry.
 *
 * Every entry is written exactly once. The entries that the rounding of the remaining mass leaves unassigned at the end get a probability of 1
 */
HIPRT_DEVICE void spmis_build_cell_alias_table(const float* scaled_importances, unsigned int count, float* cell_alias_table)
{
	unsigned int light = 0;
	while (light < count && scaled_importances[light] > 1.0f)
		light++;

	unsigned int heavy = 0;
	while (heavy < count && scaled_importances[heavy] <= 1.0f)
		heavy++;

	float heavy_remaining_mass = heavy < count ? scaled_importances[heavy] : 0.0f;
	// Runs until the heavy entries are exhausted, not the light ones: a heavy entry whose remaining mass dropped to 1 or below after the last light
	// entry still needs the next heavy entry to fill its bin
	while (heavy < count)
	{
		if (heavy_remaining_mass > 1.0f)
		{
			if (light >= count)
				// Remaining mass above 1 with no light entry left: rounding only
				break;

			// The heavy entry fills the rest of the bin of the light entry
			float light_probability = scaled_importances[light];
			spmis_write_alias_table_entry(cell_alias_table, light, light_probability, heavy);
			heavy_remaining_mass -= 1.0f - light_probability;

			light++;
			while (light < count && scaled_importances[light] > 1.0f)
				light++;
		}
		else
		{
			// The heavy entry has become light: its bin is filled by the next heavy entry
			unsigned int next_heavy = heavy + 1;
			while (next_heavy < count && scaled_importances[next_heavy] <= 1.0f)
				next_heavy++;

			if (next_heavy >= count)
				break;

			spmis_write_alias_table_entry(cell_alias_table, heavy, heavy_remaining_mass, next_heavy);
			heavy_remaining_mass = scaled_importances[next_heavy] - (1.0f - heavy_remaining_mass);
			heavy				 = next_heavy;
		}
	}

	// Entries left by the rounding of the remaining mass: the current heavy entry, the heavy entries after it and the light entries not reached
	for (unsigned int index = heavy; index < count; index++)
		if (scaled_importances[index] > 1.0f)
			spmis_write_alias_table_entry(cell_alias_table, index, 1.0f, index);

	for (unsigned int index = light; index < count; index++)
		if (scaled_importances[index] <= 1.0f)
			spmis_write_alias_table_entry(cell_alias_table, index, 1.0f, index);
}

#ifdef __KERNELCC__
/**
 * Alias table of the alive cell 'cell_alive_index', built by the whole block: the threads load the importances, thread 0 builds the table
 */
HIPRT_DEVICE void spmis_build_cell_alias_table_block(unsigned short int* cell_non_zero_reservoir_counters,
													 unsigned int* cell_offsets,
													 unsigned int* cell_alive_list,
													 unsigned int* pixel_indices_sorted,
													 ReSTIRPTReservoir* input_reservoirs,
													 float* cell_cdfs,
													 float* cell_alias_tables,
													 unsigned int size,
													 unsigned int cell_alive_index)
{
	unsigned int cell_index = cell_alive_list[cell_alive_index];
	if (cell_index >= size)
		return;

	unsigned int count = hippt::min(static_cast<unsigned int>(cell_non_zero_reservoir_counters[cell_index]), RESTIR_PT_SPMIS_ALIAS_TABLE_MAX_CELL_SIZE);
	if (count == 0)
		// No table to build
		return;

	unsigned int cell_offset = cell_offsets[cell_index];
	// SPMIS CDF layout: cdf[0] holds the sum of the importances of the cell
	float importance_scale = count / cell_cdfs[cell_offset];

	__shared__ float scaled_importances[RESTIR_PT_SPMIS_ALIAS_TABLE_MAX_CELL_SIZE];
	for (unsigned int index_in_cell = threadIdx.x; index_in_cell < count; index_in_cell += blockDim.x)
	{
		const ReSTIRPTReservoir& input_reservoir = input_reservoirs[pixel_indices_sorted[cell_offset + index_in_cell]];

		// Same importance as ReSTIR_PT_SPMIS_BuildCDFs
		float importance				  = input_reservoir.UCW * input_reservoir.sample.target_function * input_reservoir.confidence;
		scaled_importances[index_in_cell] = importance * importance_scale;
	}

	__syncthreads();

	// The sweep is sequential: one thread builds the table of the cell from the shared importances
	if (threadIdx.x == 0)
		spmis_build_cell_alias_table(scaled_importances, count, cell_alias_tables + cell_offset * RESTIR_PT_SPMIS_UNION_DOMAIN_ALIAS_TABLE_FLOAT_COUNT);
}
#endif // #ifdef __KERNELCC__

/**
 * Alias tables of the cells (ReSTIRPTSPMISUnionDomainSettings::cell_alias_tables), read by stage 1 of the neighbor selection of the union domain reuse.
 *
 * Same distribution as the CDF of the cell built by ReSTIR_PT_SPMIS_BuildCDFs (same importance c W p_hat of every non-zero pixel, same sum cdf[0]), which
 * must have run before. Sampling the table costs one load where inverting the CDF costs the LUT loads and the dependent loads of the binary search
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
__launch_bounds__(RESTIR_PT_SPMIS_ALIAS_TABLE_BLOCK_SIZE) ReSTIR_PT_SPMIS_BuildAliasTables(unsigned short int* cell_non_zero_reservoir_counters,
																						   unsigned int* cell_offsets,
																						   unsigned int* cell_alive_list,
																						   unsigned int* pixel_indices_sorted,
																						   ReSTIRPTReservoir* input_reservoirs,
																						   float* cell_cdfs,
																						   float* cell_alias_tables,
																						   unsigned int* cell_alive_count,
																						   unsigned int size)
#else  // #ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void)
inline ReSTIR_PT_SPMIS_BuildAliasTables(AtomicType<unsigned short int>* cell_non_zero_reservoir_counters,
										unsigned int* cell_offsets,
										unsigned int* cell_alive_list,
										unsigned int cell_alive_count,
										unsigned int* pixel_indices_sorted,
										ReSTIRPTReservoir* input_reservoirs,
										float* cell_cdfs,
										float* cell_alias_tables)
#endif // #ifdef __KERNELCC__
{
#ifndef __KERNELCC__
	float scaled_importances[RESTIR_PT_SPMIS_ALIAS_TABLE_MAX_CELL_SIZE];

	for (unsigned int cell_alive_index = 0; cell_alive_index < cell_alive_count; cell_alive_index++)
	{
		unsigned int cell_index = cell_alive_list[cell_alive_index];
		unsigned int count = hippt::min(static_cast<unsigned int>(cell_non_zero_reservoir_counters[cell_index]), RESTIR_PT_SPMIS_ALIAS_TABLE_MAX_CELL_SIZE);
		if (count == 0)
			continue;

		unsigned int cell_offset = cell_offsets[cell_index];
		// SPMIS CDF layout: cdf[0] holds the sum of the importances of the cell
		float importance_scale = count / cell_cdfs[cell_offset];

		for (unsigned int index_in_cell = 0; index_in_cell < count; index_in_cell++)
		{
			const ReSTIRPTReservoir& input_reservoir = input_reservoirs[pixel_indices_sorted[cell_offset + index_in_cell]];

			// Same importance as ReSTIR_PT_SPMIS_BuildCDFs
			float importance				  = input_reservoir.UCW * input_reservoir.sample.target_function * input_reservoir.confidence;
			scaled_importances[index_in_cell] = importance * importance_scale;
		}

		spmis_build_cell_alias_table(scaled_importances, count, cell_alias_tables + cell_offset * RESTIR_PT_SPMIS_UNION_DOMAIN_ALIAS_TABLE_FLOAT_COUNT);
	}
#else  // #ifndef __KERNELCC__
	// The grid has a fixed number of blocks: the host launches this kernel without reading the number of alive cells back, see ReSTIR_PT_SPMIS_BuildCDFs.
	// Each block loops over the cells
	unsigned int alive_count = *cell_alive_count;
	for (unsigned int cell_alive_index = blockIdx.x; cell_alive_index < alive_count; cell_alive_index += gridDim.x)
	{
		spmis_build_cell_alias_table_block(cell_non_zero_reservoir_counters, cell_offsets, cell_alive_list, pixel_indices_sorted, input_reservoirs, cell_cdfs,
										   cell_alias_tables, size, cell_alive_index);

		// The next cell of the block rewrites the shared importances that thread 0 may still be reading to build the table of this cell
		__syncthreads();
	}
#endif // #ifndef __KERNELCC__
}

#endif // #ifndef KERNELS_RESTIR_SPMIS_BUILD_ALIAS_TABLES_H
