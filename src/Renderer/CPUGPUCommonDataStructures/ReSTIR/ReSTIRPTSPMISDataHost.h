/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_SPMIS_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_SPMIS_DATA_HOST_H

#include "HostDeviceCommon/ReSTIR/ReSTIRPTSPMISSettings.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRPTSPMISDataHostInternal = GenericSoA<DataContainer,
												 unsigned int,										   // All pixel hashes
												 GenericAtomicType<unsigned int, DataContainer>,	   // All pixel hashes checksums
												 unsigned int,										   // All pixel index in cell
												 unsigned int,										   // All pixel reuse cell pixel indices
												 unsigned int,										   // Important pixel indices sorting values
												 GenericAtomicType<unsigned short int, DataContainer>, // Cell pixels counters
												 GenericAtomicType<unsigned short int, DataContainer>, // Cells non-zero reservoir counters
												 GenericAtomicType<unsigned int, DataContainer>,	   // Cell global offset counter
												 GenericAtomicType<unsigned int, DataContainer>,	   // Cell total count counter
												 GenericAtomicType<unsigned char, DataContainer>,	   // Cell occupied
												 unsigned int,										   // Cell alive list
												 unsigned int,										   // Cell offsets
												 GenericAtomicType<unsigned int, DataContainer>,	   // Cells confidence sums
												 float,												   // Cells CDFs
												 unsigned short int,								   // Cells CDF LUTs for speeding up CDF sampling
												 unsigned int,										   // Cells CDF LUT offsets
												 float,												   // Cells variance
												 float,												   // Union domain reuse neighbor records
												 float,												   // Union domain reuse cell alias tables
												 unsigned int>;										   // Union domain reuse domain cache

enum ReSTIRPTSPMISDataHostBuffers
{
	RESTIR_PT_SPMIS_ALL_PIXEL_HASHES,
	RESTIR_PT_SPMIS_ALL_PIXEL_HASHES_CHECKSUMS,
	RESTIR_PT_SPMIS_ALL_PIXEL_INDEX_IN_CELL,
	RESTIR_PT_SPMIS_ALL_PIXEL_REUSE_CELL_PIXEL_INDEX,
	RESTIR_PT_SPMIS_PIXEL_INDICES_SORTED,
	RESTIR_PT_SPMIS_CELL_COUNTERS,
	RESTIR_PT_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS,
	RESTIR_PT_SPMIS_CELL_GLOBAL_OFFSET_COUNTER,
	RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER,
	RESTIR_PT_SPMIS_CELL_OCCUPIED,
	RESTIR_PT_SPMIS_CELL_ALIVE_LIST,
	RESTIR_PT_SPMIS_CELL_OFFSETS,
	RESTIR_PT_SPMIS_CELL_CONFIDENCE_SUMS,
	RESTIR_PT_SPMIS_CELL_CDFS,
	RESTIR_PT_SPMIS_CELL_CDF_LUTS,
	RESTIR_PT_SPMIS_CELL_CDF_LUT_OFFSETS,
	RESTIR_PT_SPMIS_CELL_VARIANCE,
	RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORDS,
	RESTIR_PT_SPMIS_UNION_DOMAIN_CELL_ALIAS_TABLES,
	RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE,
};

template <template <typename> typename DataContainer>
struct ReSTIRPTSPMISDataHost
{
	void resize(unsigned int width, unsigned int height)
	{
		bool union_domain_buffers_were_allocated = union_domain_buffers_allocated();

		// RESTIR_PT_SPMIS_CELL_CDF_LUTS is resized when creating spmis cells, not here
		// The union domain reuse buffers are only allocated while the union domain reuse is used, see resize_union_domain_buffers()
		m_spmis_data.resize(width * height, { RESTIR_PT_SPMIS_CELL_GLOBAL_OFFSET_COUNTER, RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER,
											  RESTIR_PT_SPMIS_CELL_CDF_LUTS, RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORDS,
											  RESTIR_PT_SPMIS_UNION_DOMAIN_CELL_ALIAS_TABLES, RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE });
		m_spmis_data.template get_buffer<RESTIR_PT_SPMIS_ALL_PIXEL_REUSE_CELL_PIXEL_INDEX>().resize(width * height * RESTIR_PT_SPMIS_REUSE_CELL_CACHE_SIZE);

		m_spmis_data.template resize_one_buffer<RESTIR_PT_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>(1);
		m_spmis_data.template resize_one_buffer<RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER>(1);

		if (union_domain_buffers_were_allocated)
			// Keeping the union domain reuse buffers allocated, at the new resolution
			resize_union_domain_buffers(width, height);

		reset();
	}

	/**
	 * Allocates the buffers of the SPMIS union domain reuse (ReSTIRPTSPMISUnionDomainSettings) for the given render resolution. Not done by resize(): these
	 * buffers are only needed while the union domain reuse is used
	 */
	void resize_union_domain_buffers(unsigned int width, unsigned int height)
	{
		// Not using resize_one_buffer() so that the maximum size of the SoA, which is the number of hash grid cells, isn't changed
		m_spmis_data.template get_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORDS>().resize(width * height *
																								 RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORD_FLOAT_COUNT);
		m_spmis_data.template get_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CELL_ALIAS_TABLES>().resize(width * height *
																								  RESTIR_PT_SPMIS_UNION_DOMAIN_ALIAS_TABLE_FLOAT_COUNT);
		m_spmis_data.template get_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE>().resize(width * height * RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE_STRIDE);

		// The newly allocated cache must not hold any domain
		reset_union_domain_cache();
	}

	/**
	 * Returns true if the buffers were allocated
	 */
	bool free_union_domain_buffers()
	{
		if (!union_domain_buffers_allocated())
			return false;

		m_spmis_data.template free_one_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORDS>();
		m_spmis_data.template free_one_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CELL_ALIAS_TABLES>();
		m_spmis_data.template free_one_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE>();

		return true;
	}

	bool union_domain_buffers_allocated() const
	{
		return m_spmis_data.template get_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE>().size() > 0;
	}

	void reset()
	{
		if (maximum_size() == 0)
			return;

		m_spmis_data.template memset_buffer<RESTIR_PT_SPMIS_CELL_OCCUPIED>(0);
		m_spmis_data.template memset_buffer<RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER>(0);
		m_spmis_data.template memset_buffer<RESTIR_PT_SPMIS_CELL_ALIVE_LIST>(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

		m_spmis_data.template memset_buffer<RESTIR_PT_SPMIS_ALL_PIXEL_REUSE_CELL_PIXEL_INDEX>(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
		// The union domain cache is emptied whenever the reuse cell cache is
		reset_union_domain_cache();
	}

	void reset_union_domain_cache()
	{
		if (!union_domain_buffers_allocated())
			return;

		// A cached cell count of HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX marks an empty cache
		m_spmis_data.template memset_buffer<RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE>(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
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

			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.neighbor_records	 = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.cell_alias_tables = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.domain_cache		 = nullptr;

			return;
		}

		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixel_hashes =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_ALL_PIXEL_HASHES>();
		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixel_hashes_checksums =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_PT_SPMIS_ALL_PIXEL_HASHES_CHECKSUMS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixels_index_in_cell =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_ALL_PIXEL_INDEX_IN_CELL>();
		render_data.render_settings.restir_pt_settings.spmis_settings.all_pixels_reuse_cell_pixel_index =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_ALL_PIXEL_REUSE_CELL_PIXEL_INDEX>();
		render_data.render_settings.restir_pt_settings.spmis_settings.pixel_hashes_count = (unsigned int)maximum_size();
		render_data.render_settings.restir_pt_settings.spmis_settings.pixel_indices_sorted =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_PIXEL_INDICES_SORTED>();

		render_data.render_settings.restir_pt_settings.spmis_settings.cell_pixels_counters =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_PT_SPMIS_CELL_COUNTERS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_non_zero_reservoir_counters =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_PT_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_global_offset_counter =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_PT_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_total_count_counter =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_PT_SPMIS_CELL_TOTAL_COUNT_COUNTER>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_occupied =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_PT_SPMIS_CELL_OCCUPIED>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_alive_list =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_CELL_ALIVE_LIST>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_offsets = m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_CELL_OFFSETS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_confidence_sums =
			m_spmis_data.template get_buffer_data_atomic_ptr<RESTIR_PT_SPMIS_CELL_CONFIDENCE_SUMS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdfs = m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_CELL_CDFS>();
		// The pointer is not set here because the buffer is resized when creating the SPMIS cells, not here so we may not have a valid pointer to set at all
		// since the buffer hasn't been allocated (resized)
		// render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts = m_spmis_data.template
		// get_buffer_data_ptr<RESTIR_PT_SPMIS_CELL_CDF_LUTS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_lut_offsets =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_CELL_CDF_LUT_OFFSETS>();
		render_data.render_settings.restir_pt_settings.spmis_settings.cell_variance =
			m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_CELL_VARIANCE>();

		if (union_domain_buffers_allocated())
		{
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.neighbor_records =
				m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_UNION_DOMAIN_NEIGHBOR_RECORDS>();
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.cell_alias_tables =
				m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_UNION_DOMAIN_CELL_ALIAS_TABLES>();
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.domain_cache =
				m_spmis_data.template get_buffer_data_ptr<RESTIR_PT_SPMIS_UNION_DOMAIN_CACHE>();
		}
		else
		{
			// Union domain reuse not used, its buffers are not allocated
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.neighbor_records	 = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.cell_alias_tables = nullptr;
			render_data.render_settings.restir_pt_settings.spmis_settings.union_domain.domain_cache		 = nullptr;
		}
	}

	ReSTIRPTSPMISDataHostInternal<DataContainer> m_spmis_data;
};

#endif // #ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_SPMIS_DATA_HOST_H
