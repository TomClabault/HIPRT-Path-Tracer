/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_DIRECTIONAL_SPATIAL_REUSE_DATA_HOST_H
#define RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_DIRECTIONAL_SPATIAL_REUSE_DATA_HOST_H

#include "HostDeviceCommon/RenderData.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using ReSTIRPTDirectionalSpatialReuseDataHostInternal =
	GenericSoA<DataContainer,
			   unsigned char,			// Spatial reuse radius
			   unsigned long long int>; // Spatial reuse direction mask (encoded as a bitfield of valid reuse directions)

enum ReSTIRPTDirectionalSpatialReuseDataHostBuffers
{
	RESTIR_PT_DIRECTIONAL_SPATIAL_REUSE_RADIUS,
	RESTIR_PT_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL,
};

template <template <typename> typename DataContainer>
struct ReSTIRPTDirectionalSpatialReuseDataHost
{
	void resize(unsigned int width, unsigned int height)
	{
		m_spatial_reuse_data.resize(width * height);
	}

	void reset()
	{
		if (maximum_size() == 0)
			return;
	}

	bool free()
	{
		if (maximum_size() > 0)
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

	std::size_t maximum_size() const
	{
		return m_spatial_reuse_data.maximum_size();
	}

	void to_device(HIPRTRenderData& render_data)
	{
		ReSTIRPTSpatialPassSettings& spatial_pass_settings = render_data.render_settings.restir_pt_settings.spatial_pass;

		if (maximum_size() == 0)
		{
			spatial_pass_settings.per_pixel_spatial_reuse_directions_mask_ull = nullptr;
			spatial_pass_settings.per_pixel_spatial_reuse_radius			  = nullptr;

			return;
		}

		spatial_pass_settings.per_pixel_spatial_reuse_directions_mask_ull =
			m_spatial_reuse_data.template get_buffer_data_ptr<RESTIR_PT_DIRECTIONAL_SPATIAL_REUSE_DIRECTION_MASK_ULL>();
		spatial_pass_settings.per_pixel_spatial_reuse_radius = m_spatial_reuse_data.template get_buffer_data_ptr<RESTIR_PT_DIRECTIONAL_SPATIAL_REUSE_RADIUS>();
	}

	ReSTIRPTDirectionalSpatialReuseDataHostInternal<DataContainer> m_spatial_reuse_data;
};

#endif // #ifndef RENDERER_CPU_GPU_COMMON_DATA_STRUCTURES_RESTIR_PT_DIRECTIONAL_SPATIAL_REUSE_DATA_HOST_H
