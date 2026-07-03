/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_RESTIR_RENDER_PASS_COMMON_H
#define GPU_RENDERER_RESTIR_RENDER_PASS_COMMON_H

#include "HostDeviceCommon/ReSTIR/ReSTIRSettingsHelper.h"
#include "Renderer/GPURenderer.h"

class ReSTIRRenderPassCommon
{
public:
	static constexpr const char* const DIRECTIONAL_REUSE_KERNEL_FUNCTION_NAME				= "ReSTIR_Directional_Reuse_Compute";
	static constexpr const char* const DIRECTIONAL_REUSE_KERNEL_FILE						= DEVICE_KERNELS_DIRECTORY "/ReSTIR/DirectionalReuseCompute.h";
	static constexpr const char* const DIRECTIONAL_REUSE_RESTIR_VARIANT_COMPILE_OPTION_NAME = "ComputingSpatialDirectionalReuseReSTIRVariant";

	static constexpr float AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE = 0.025f;

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static void resize_common_buffers(GPURenderer* renderer,
									  int new_width,
									  int new_height,
									  ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		directional_spatial_reuse_data.resize(new_width, new_height);
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static bool pre_render_update_common_buffers(const HIPRTRenderData& render_data,
												 ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		return pre_render_update_directional_reuse_buffers<ReSTIRVariant>(render_data, directional_spatial_reuse_data);
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static bool pre_render_update_directional_reuse_buffers(const HIPRTRenderData& render_data,
															ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		ReSTIRCommonSpatialPassSettings spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIRVariant>(render_data);

		bool render_data_invalidated = false;
		if (spatial_pass_settings.do_adaptive_directional_spatial_reuse(render_data.render_settings.accumulate))
		{
			if (directional_spatial_reuse_data.size() == 0)
			{
				directional_spatial_reuse_data.resize(render_data.render_settings.render_resolution.x, render_data.render_settings.render_resolution.y);

				render_data_invalidated = true;
			}
		}
		else
		{
			if (directional_spatial_reuse_data.size() > 0)
			{
				directional_spatial_reuse_data.free();

				render_data_invalidated = true;
			}
		}

		return render_data_invalidated;
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static void reset_common_buffers(ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		directional_spatial_reuse_data.reset();
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static bool free_common_buffers(ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		return directional_spatial_reuse_data.free();
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static void update_render_data_common_buffers(HIPRTRenderData& render_data,
												  ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		directional_spatial_reuse_data.template to_device<ReSTIRVariant>(render_data);
	}
};

#endif
