/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_RESTIR_PT_RENDER_PASS_COMMON_H
#define GPU_RENDERER_RESTIR_PT_RENDER_PASS_COMMON_H

#include "HostDeviceCommon/ReSTIR/ReSTIRPTSettingsHelper.h"
#include "Renderer/GPURenderer.h"

class ReSTIRPTRenderPassCommon
{
public:
	static constexpr const char* const DIRECTIONAL_REUSE_KERNEL_FUNCTION_NAME = "ReSTIR_PT_Directional_Reuse_Compute";
	static constexpr const char* const DIRECTIONAL_REUSE_KERNEL_FILE		  = DEVICE_KERNELS_DIRECTORY "/ReSTIR/DirectionalReuseCompute.h";

	static constexpr float AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE = 0.025f;

	template <template <typename> typename DataContainer>
	static void resize_common_buffers(GPURenderer* renderer,
									  int new_width,
									  int new_height,
									  ReSTIRPTDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		directional_spatial_reuse_data.resize(new_width, new_height);
	}

	template <template <typename> typename DataContainer>
	static bool pre_render_update_common_buffers(const HIPRTRenderData& render_data,
												 ReSTIRPTDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		return pre_render_update_directional_reuse_buffers(render_data, directional_spatial_reuse_data);
	}

	template <template <typename> typename DataContainer>
	static bool pre_render_update_directional_reuse_buffers(const HIPRTRenderData& render_data,
															ReSTIRPTDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		ReSTIRPTCommonSpatialPassSettings spatial_pass_settings = ReSTIRPTSettingsHelper::get_restir_pt_spatial_pass_settings(render_data);

		bool render_data_invalidated = false;
		if (spatial_pass_settings.do_adaptive_directional_spatial_reuse(render_data.render_settings.accumulate))
		{
			if (directional_spatial_reuse_data.maximum_size() == 0)
			{
				directional_spatial_reuse_data.resize(render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y);

				render_data_invalidated = true;
			}
		}
		else
		{
			if (directional_spatial_reuse_data.maximum_size() > 0)
			{
				directional_spatial_reuse_data.free();

				render_data_invalidated = true;
			}
		}

		return render_data_invalidated;
	}

	template <template <typename> typename DataContainer>
	static void reset_common_buffers(ReSTIRPTDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		directional_spatial_reuse_data.reset();
	}

	template <template <typename> typename DataContainer>
	static bool free_common_buffers(ReSTIRPTDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		return directional_spatial_reuse_data.free();
	}

	template <template <typename> typename DataContainer>
	static void update_render_data_common_buffers(HIPRTRenderData& render_data,
												  ReSTIRPTDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		directional_spatial_reuse_data.to_device(render_data);
	}
};

#endif // #ifndef GPU_RENDERER_RESTIR_PT_RENDER_PASS_COMMON_H
