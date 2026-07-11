/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_SPMIS_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_SPMIS_DATA_HOST_H

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRSPMISDataHostInternal = GenericSoA<DataContainer,
											   unsigned int,										 // All pixel hashes
											   GenericAtomicType<unsigned int, DataContainer>,		 // All pixel hashes checksums
											   unsigned int,										 // All pixel index in cell
											   unsigned int,										 // All pixel reuse cell pixel index
											   unsigned int,										 // Important pixel indices sorting values
											   GenericAtomicType<unsigned short int, DataContainer>, // Cell pixels counters
											   GenericAtomicType<unsigned short int, DataContainer>, // Cells non-zero reservoir counters
											   GenericAtomicType<unsigned int, DataContainer>,		 // Cell global offset counter
											   GenericAtomicType<unsigned int, DataContainer>,		 // Cell total count counter
											   GenericAtomicType<unsigned char, DataContainer>,		 // Cell occupied
											   unsigned int,										 // Cell alive list
											   unsigned int,										 // Cell offsets
											   GenericAtomicType<unsigned int, DataContainer>,		 // Cells confidence sums
											   float,												 // Cells CDFs
											   unsigned short int,									 // Cells CDF LUTs for speeding up CDF sampling
											   unsigned int,										 // Cells CDF LUT offsets
											   float>;												 // Cells variance

enum ReSTIRSPMISDataHostBuffers
{
	RESTIR_SPMIS_ALL_PIXEL_HASHES,
	RESTIR_SPMIS_ALL_PIXEL_HASHES_CHECKSUMS,
	RESTIR_SPMIS_ALL_PIXEL_INDEX_IN_CELL,
	RESTIR_SPMIS_ALL_PIXEL_REUSE_CELL_PIXEL_INDEX,
	RESTIR_SPMIS_PIXEL_INDICES_SORTED,
	RESTIR_SPMIS_CELL_COUNTERS,
	RESTIR_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS,
	RESTIR_SPMIS_CELL_GLOBAL_OFFSET_COUNTER,
	RESTIR_SPMIS_CELL_TOTAL_COUNT_COUNTER,
	RESTIR_SPMIS_CELL_OCCUPIED,
	RESTIR_SPMIS_CELL_ALIVE_LIST,
	RESTIR_SPMIS_CELL_OFFSETS,
	RESTIR_SPMIS_CELL_CONFIDENCE_SUMS,
	RESTIR_SPMIS_CELL_CDFS,
	RESTIR_SPMIS_CELL_CDF_LUTS,
	RESTIR_SPMIS_CELL_CDF_LUT_OFFSETS,
	RESTIR_SPMIS_CELL_VARIANCE,
};

template <template <typename> typename DataContainer>
struct ReSTIRSPMISDataHost
{
	void resize(unsigned int width, unsigned int height)
	{
		// RESTIR_SPMIS_CELL_CDF_LUTS is resized when creating spmis cells, not here
		m_spmis_data.resize(width * height, { RESTIR_SPMIS_CELL_GLOBAL_OFFSET_COUNTER, RESTIR_SPMIS_CELL_TOTAL_COUNT_COUNTER, RESTIR_SPMIS_CELL_CDF_LUTS });

		m_spmis_data.template resize_one_buffer<RESTIR_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>(1);
		m_spmis_data.template resize_one_buffer<RESTIR_SPMIS_CELL_TOTAL_COUNT_COUNTER>(1);

		reset();
	}

	void reset()
	{
		if (maximum_size() == 0)
			return;

		m_spmis_data.template memset_buffer<RESTIR_SPMIS_CELL_OCCUPIED>(0);
		m_spmis_data.template memset_buffer<RESTIR_SPMIS_CELL_TOTAL_COUNT_COUNTER>(0);
		m_spmis_data.template memset_buffer<RESTIR_SPMIS_CELL_ALIVE_LIST>(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
	}

	bool free()
	{
		if (maximum_size() > 0)
		{
			m_spmis_data.free();

			return true;
		}

		return false;
	}

	std::size_t get_byte_size() const
	{
		return m_spmis_data.get_byte_size();
	}

	std::size_t maximum_size() const
	{
		return m_spmis_data.maximum_size();
	}

	void to_device(HIPRTRenderData& render_data)
	{
		if (maximum_size() == 0)
		{
			render_data.render_settings.restir_pt_settings.spmis_settings.pixel_hashes_count = 0;

			render_data.render_settings.restir_pt_settings.spmis_settings.all_pixel_hashes					= nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.all_pixel_hashes_checksums		= nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.all_pixels_index_in_cell			= nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.all_pixels_reuse_cell_pixel_index = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.pixel_indices_sorted				= nullptr;

			render_data.render_settings.restir_pt_settings.spmis_settings.cell_pixels_counters			   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_non_zero_reservoir_counters = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_global_offset_counter	   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_total_count_counter		   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_occupied					   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_alive_list				   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_offsets					   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_confidence_sums			   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdfs						   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts					   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_lut_offsets			   = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.cell_variance					   = nullptr;

			return;
		}

		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixel_hashes =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_ALL_PIXEL_HASHES>();
		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixel_hashes_checksums =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_SPMIS_ALL_PIXEL_HASHES_CHECKSUMS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixels_index_in_cell =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_ALL_PIXEL_INDEX_IN_CELL>();
		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixels_reuse_cell_pixel_index =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_ALL_PIXEL_REUSE_CELL_PIXEL_INDEX>();
		render_data.render_settings.restir_pt_settings.spmis_settings.pixel_hashes_count = (unsigned int)maximum_size();
		render_data.render_settings.restir_pt_settings.spmis_settings.pixel_indices_sorted =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_PIXEL_INDICES_SORTED>();

		render_data.render_settings.restir_pt_settings.spmis_settings.cell_pixels_counters =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_SPMIS_CELL_COUNTERS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_non_zero_reservoir_counters =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_global_offset_counter =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_total_count_counter =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_SPMIS_CELL_TOTAL_COUNT_COUNTER>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_occupied =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_SPMIS_CELL_OCCUPIED>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_alive_list =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_CELL_ALIVE_LIST>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_offsets = m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_CELL_OFFSETS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_confidence_sums =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_SPMIS_CELL_CONFIDENCE_SUMS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdfs = m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_CELL_CDFS>();
		// The pointer is not set here because the buffer is resized when creating the SPMIS cells, not here so we may not have a valid pointer to set at all
		// since the buffer hasn't been allocated (resized)
		// render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts = m_spmis_data.template
		// get_buffer_data_ptr<RESTIR_SPMIS_CELL_CDF_LUTS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_lut_offsets =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_CELL_CDF_LUT_OFFSETS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_variance = m_spmis_data.template get_buffer_data_ptr<RESTIR_SPMIS_CELL_VARIANCE>();
	}

	ReSTIRSPMISDataHostInternal<DataContainer> m_spmis_data;
};

#endif
