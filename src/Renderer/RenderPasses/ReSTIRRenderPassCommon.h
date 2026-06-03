/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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
									  ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data,
									  ReSTIRSPMISDataHost<DataContainer>& spmis_data)
	{
		directional_spatial_reuse_data.resize(new_width, new_height);
		spmis_data.resize(new_width, new_height);
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static bool pre_render_update_common_buffers(const HIPRTRenderData& render_data,
												 GPURenderer* renderer,
												 ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data,
												 ReSTIRSPMISDataHost<DataContainer>& spmis_data)
	{
		bool render_data_updated = false;

		render_data_updated |= pre_render_update_directional_reuse_buffers<ReSTIRVariant>(render_data, renderer, directional_spatial_reuse_data);
		render_data_updated |= pre_render_update_spmis_buffers<ReSTIRVariant>(render_data, renderer, spmis_data);

		return render_data_updated;
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static bool pre_render_update_directional_reuse_buffers(const HIPRTRenderData& render_data,
															GPURenderer* renderer,
															ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data)
	{
		ReSTIRCommonSpatialPassSettings spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIRVariant>(render_data);

		// Allocating / deallocating the adaptive directional spatial reuse buffers if the feature
		// isn't used
		bool render_data_invalidated = false;
		if (spatial_pass_settings.do_adaptive_directional_spatial_reuse(render_data.render_settings.accumulate))
		{
			if (directional_spatial_reuse_data.size() == 0)
			{
				directional_spatial_reuse_data.resize(renderer->m_render_resolution.x, renderer->m_render_resolution.y);

				render_data_invalidated = true;
			}
		}
		else
		{
			// We're not using the feature so we can free the buffers

			// Freeing the proper buffer depending on whether we use the 64 bits buffer or not
			if (directional_spatial_reuse_data.size() > 0)
			{
				directional_spatial_reuse_data.free();

				render_data_invalidated = true;
			}
		}

		return render_data_invalidated;
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static bool pre_render_update_spmis_buffers(const HIPRTRenderData& render_data, GPURenderer* renderer, ReSTIRSPMISDataHost<DataContainer>& spmis_data)
	{
		int mis_weight_type;
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			mis_weight_type = renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE);
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			mis_weight_type = renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_TYPE);
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			mis_weight_type = renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE);
		else
			static_assert(ReSTIRVariant == ReSTIR_VARIANT_DI || ReSTIRVariant == ReSTIR_VARIANT_GI || ReSTIRVariant == ReSTIR_VARIANT_PT,
						  "Invalid ReSTIR variant");

		bool render_data_invalidated = false;
		if (mis_weight_type == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS || mis_weight_type == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
		{
			if (spmis_data.size() == 0)
			{
				spmis_data.resize(renderer->m_render_resolution.x, renderer->m_render_resolution.y);

				render_data_invalidated = true;
			}
		}
		else
		{
			if (spmis_data.size() > 0)
			{
				spmis_data.free();

				render_data_invalidated = true;
			}
		}

		return render_data_invalidated;
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static void reset_common_buffers(ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data,
									 ReSTIRSPMISDataHost<DataContainer>& spmis_data)
	{
		directional_spatial_reuse_data.reset();
		spmis_data.reset();
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static bool free_common_buffers(ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data,
									ReSTIRSPMISDataHost<DataContainer>& spmis_data)
	{
		bool render_data_invalidated = false;

		render_data_invalidated |= directional_spatial_reuse_data.free();
		render_data_invalidated |= spmis_data.free();

		return render_data_invalidated;
	}

	template <int ReSTIRVariant, template <typename> typename DataContainer>
	static void update_render_data_common_buffers(HIPRTRenderData& render_data,
												  ReSTIRDirectionalSpatialReuseDataHost<DataContainer>& directional_spatial_reuse_data,
												  ReSTIRSPMISDataHost<DataContainer>& spmis_data)
	{
		ReSTIRCommonSpatialPassSettings& common_spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<ReSTIRVariant>(render_data);
		ReSTIRCommonSPMISSettings& common_spmis_settings			  = ReSTIRSettingsHelper::get_restir_spmis_settings<ReSTIRVariant>(render_data);

		directional_spatial_reuse_data.to_device(render_data);
		spmis_data.to_device(render_data);
	}
};

#endif
