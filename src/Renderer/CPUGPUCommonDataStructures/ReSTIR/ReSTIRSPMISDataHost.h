/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_SPMIS_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_SPMIS_DATA_HOST_H

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRSPMISDataHostInternal = GenericSoA<DataContainer,
											   unsigned int>; // Pixel hashes

enum ReSTIRSPMISDataHostBuffers
{
	RESTIR_SPMIS_PIXEL_HASHES,
};

template <template <typename> typename DataContainer>
struct ReSTIRSPMISDataHost
{
	void resize(unsigned int width, unsigned int height)
	{
		m_spmis_data.resize(width * height);
	}

	void reset()
	{
		if (size() == 0)
			return;

		m_spmis_data.memset_buffer<RESTIR_SPMIS_PIXEL_HASHES>(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
	}

	bool free()
	{
		if (size() > 0)
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

	std::size_t size() const
	{
		return m_spmis_data.size();
	}

	void to_device(HIPRTRenderData& render_data)
	{
		if (size() == 0)
		{
			render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings.pixel_hashes = nullptr;

			return;
		}

		render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings.pixel_hashes =
			m_spmis_data.get_buffer_data_ptr<RESTIR_SPMIS_PIXEL_HASHES>();
	}

	ReSTIRSPMISDataHostInternal<DataContainer> m_spmis_data;
};

#endif
