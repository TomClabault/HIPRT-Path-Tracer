/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_DIRECTIONAL_SPATIAL_REUSE_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_DIRECTIONAL_SPATIAL_REUSE_DATA_HOST_H

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRDirectionalSpatialReuseDataHostInternal =
	GenericSoA<DataContainer,
			   unsigned char,											  // Spatial reuse radius
			   unsigned long long int,									  // Spatial reuse direction mask (encoded as a bitfield of valid reuse directions)
			   GenericAtomicType<unsigned long long int, DataContainer>,  // Spatial reuse statistics: total number of reuse attempts
			   GenericAtomicType<unsigned long long int, DataContainer>>; // Spatial reuse statistics: total number of reuse hits

enum ReSTIRDirectionalSpatialReuseDataHostBuffers
{
	RESTIR_DIRECTIONAL_SPATIAL_REUSE_RADIUS,
	RESTIR_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL,
	RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_TOTAL,
	RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_HITS,
};

template <template <typename> typename DataContainer>
struct ReSTIRDirectionalSpatialReuseDataHost
{
	void resize(unsigned int width, unsigned int height)
	{
		m_spatial_reuse_data.resize(width * height,
									{ RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_TOTAL, RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_HITS });

		m_spatial_reuse_data.resize_one_buffer<RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_TOTAL>(1);
		m_spatial_reuse_data.resize_one_buffer<RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_HITS>(1);
	}

	void reset()
	{
		if (size() == 0)
			return;

		m_spatial_reuse_data.memset_buffer<RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_TOTAL>(0);
		m_spatial_reuse_data.memset_buffer<RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_HITS>(0);
	}

	bool free()
	{
		if (size() > 0)
		{
			m_spatial_reuse_data.free();

			return true;
		}

		return false;
	}

	std::size_t get_byte_size() const
	{
		return m_spatial_reuse_data.get_byte_size();
	}

	std::size_t size() const
	{
		return m_spatial_reuse_data.size();
	}

	void to_device(HIPRTRenderData& render_data)
	{
		if (size() == 0)
		{
			render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull = nullptr;
			render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_radius			   = nullptr;
			render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_reuse_hit_rate_total				   = nullptr;
			render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_reuse_hit_rate_hits				   = nullptr;

			return;
		}

		render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull =
			m_spatial_reuse_data.get_buffer_data_ptr<RESTIR_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL>();
		render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_radius =
			m_spatial_reuse_data.get_buffer_data_ptr<RESTIR_DIRECTIONAL_SPATIAL_REUSE_RADIUS>();
		render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_reuse_hit_rate_total =
			m_spatial_reuse_data.get_buffer_data_atomic_ptr<RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_TOTAL>();
		render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_reuse_hit_rate_hits =
			m_spatial_reuse_data.get_buffer_data_atomic_ptr<RESTIR_DIRECTIONAL_SPATIAL_REUSE_STATISTICS_HIT_HITS>();
	}

	ReSTIRDirectionalSpatialReuseDataHostInternal<DataContainer> m_spatial_reuse_data;
};

#endif
