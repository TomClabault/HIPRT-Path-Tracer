/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_DIRECTIONAL_SPATIAL_REUSE_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_DIRECTIONAL_SPATIAL_REUSE_DATA_HOST_H

#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRDirectionalSpatialReuseDataHostInternal =
	GenericSoA<DataContainer,
			   unsigned char,			// Spatial reuse radius
			   unsigned long long int>; // Spatial reuse direction mask (encoded as a bitfield of valid reuse directions)

enum ReSTIRDirectionalSpatialReuseDataHostBuffers
{
	RESTIR_DIRECTIONAL_SPATIAL_REUSE_RADIUS,
	RESTIR_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL,
};

template <template <typename> typename DataContainer>
struct ReSTIRDirectionalSpatialReuseDataHost
{
	void resize(unsigned int width, unsigned int height)
	{
		m_spatial_reuse_data.resize(width * height);
	}

	void reset()
	{
		if (size() == 0)
			return;
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

	template <int ReSTIRVariant>
	void to_device(HIPRTRenderData& render_data)
	{
		ReSTIRCommonSpatialPassSettings& common_spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIRVariant>(render_data);

		if (size() == 0)
		{
			common_spatial_pass_settings.per_pixel_spatial_reuse_directions_mask_ull = nullptr;
			common_spatial_pass_settings.per_pixel_spatial_reuse_radius				 = nullptr;

			return;
		}

		common_spatial_pass_settings.per_pixel_spatial_reuse_directions_mask_ull =
			m_spatial_reuse_data.template get_buffer_data_ptr<RESTIR_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL>();
		common_spatial_pass_settings.per_pixel_spatial_reuse_radius =
			m_spatial_reuse_data.template get_buffer_data_ptr<RESTIR_DIRECTIONAL_SPATIAL_REUSE_RADIUS>();
	}

	ReSTIRDirectionalSpatialReuseDataHostInternal<DataContainer> m_spatial_reuse_data;
};

#endif
