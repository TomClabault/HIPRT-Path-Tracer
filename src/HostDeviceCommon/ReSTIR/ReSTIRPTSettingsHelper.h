/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTIR_PT_SETTINGS_HELPER_H
#define HOST_DEVICE_COMMON_RESTIR_PT_SETTINGS_HELPER_H

#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"

struct ReSTIRPTSettingsHelper
{
	HIPRT_HOST_DEVICE static ReSTIRPTSettings get_restir_pt_settings(const HIPRTRenderData& render_data)
	{
		return render_data.render_settings.restir_pt_settings;
	}

	HIPRT_HOST_DEVICE static const ReSTIRPTCommonSpatialPassSettings& get_restir_pt_spatial_pass_settings(const HIPRTRenderData& render_data)
	{
		return render_data.render_settings.restir_pt_settings.common_spatial_pass;
	}

	HIPRT_HOST_DEVICE static ReSTIRPTCommonSpatialPassSettings& get_restir_pt_spatial_pass_settings(HIPRTRenderData& render_data)
	{
		return render_data.render_settings.restir_pt_settings.common_spatial_pass;
	}

	HIPRT_HOST_DEVICE static ReSTIRPTCommonTemporalPassSettings get_restir_pt_temporal_pass_settings(const HIPRTRenderData& render_data)
	{
		return render_data.render_settings.restir_pt_settings.common_temporal_pass;
	}

	HIPRT_HOST_DEVICE static ReSTIRPTCommonNeighborSimiliaritySettings get_restir_pt_neighbor_similarity_settings(const HIPRTRenderData& render_data)
	{
		return render_data.render_settings.restir_pt_settings.neighbor_similarity_settings;
	}

	/**
	 * Returns the M value of a reservoir from the spatial pass input buffer given its pixel index
	 *
	 * Reads from the ReSTIR PT spatial pass input buffer.
	 */
	HIPRT_HOST_DEVICE static int get_restir_pt_spatial_pass_input_reservoir_confidence(const HIPRTRenderData& render_data, int pixel_index)
	{
		return render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs[pixel_index].confidence;
	}

	HIPRT_HOST_DEVICE static unsigned long long int get_spatial_reuse_direction_mask_ull(const HIPRTRenderData& render_data, int pixel_index)
	{
		return render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull[pixel_index];
	}

	/**
	 * Returns the shading normal or geometric normal of the given surface depending on the rejection heuristics settings
	 */
	HIPRT_HOST_DEVICE static float3_t get_normal_for_rejection_heuristic(const HIPRTRenderData& render_data, const ReSTIRPTSurface& surface)
	{
		if (render_data.render_settings.restir_pt_settings.neighbor_similarity_settings.reject_using_geometric_normals)
			return surface.geometric_normal;
		else
			return surface.shading_normal;
	}
};

#endif // #ifndef HOST_DEVICE_COMMON_RESTIR_PT_SETTINGS_HELPER_H
