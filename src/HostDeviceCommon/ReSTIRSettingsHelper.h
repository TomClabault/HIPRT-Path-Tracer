/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTI_SETTINGS_HELPER_H
#define HOST_DEVICE_COMMON_RESTI_SETTINGS_HELPER_H

#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"

template <bool IsReSTIRGI>
struct ReSTIRSettingsTypeTemplate {};

template <>
struct ReSTIRSettingsTypeTemplate<false>
{
	using Type = ReSTIRDISettings;
};

template <>
struct ReSTIRSettingsTypeTemplate<true>
{
	using Type = ReSTIRGISettings;
};

template <bool IsReSTIRGI>
using ReSTIRSettingsType = typename ReSTIRSettingsTypeTemplate<IsReSTIRGI>::Type;


struct ReSTIRSettingsHelper
{
	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static ReSTIRSettingsType<IsReSTIRGI> get_restir_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (IsReSTIRGI)
			return render_data.render_settings.restir_gi_settings;
		else
			return render_data.render_settings.restir_di_settings;
	}

	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static ReSTIRCommonSpatialPassSettings get_restir_spatial_pass_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (IsReSTIRGI)
			return render_data.render_settings.restir_gi_settings.common_spatial_pass;
		else
			return render_data.render_settings.restir_di_settings.common_spatial_pass;
	}

	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static ReSTIRCommonSpatialPassSettings& get_restir_spatial_pass_settings(HIPRTRenderData& render_data)
	{
		if constexpr (IsReSTIRGI)
			return render_data.render_settings.restir_gi_settings.common_spatial_pass;
		else
			return render_data.render_settings.restir_di_settings.common_spatial_pass;
	}

	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static ReSTIRCommonTemporalPassSettings get_restir_temporal_pass_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (IsReSTIRGI)
			return render_data.render_settings.restir_gi_settings.common_temporal_pass;
		else
			return render_data.render_settings.restir_di_settings.common_temporal_pass;
	}

	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static ReSTIRCommonNeighborSimiliaritySettings get_restir_neighbor_similarity_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (IsReSTIRGI)
			return render_data.render_settings.restir_gi_settings.neighbor_similarity_settings;
		else
			return render_data.render_settings.restir_di_settings.neighbor_similarity_settings;
	}

	/**
	 * Returns the M value of a reservoir from the spatial pass input buffer given its pixel index
	 *
	 * The template argument can be used to select between ReSTIR DI and ReSTIR GI spatial buffers
	 */
	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static int get_restir_spatial_pass_input_reservoir_M(const HIPRTRenderData& render_data, int pixel_index)
	{
		if constexpr (IsReSTIRGI)
			return render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs[pixel_index].M;
		else
			return render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[pixel_index].M;
	}

	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static unsigned long long int get_spatial_reuse_direction_mask_ull(const HIPRTRenderData& render_data, int pixel_index)
	{
		if constexpr (IsReSTIRGI)
		{
#if ReSTIR_GI_SpatialDirectionalReuseBitCount > 32
			return render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull[pixel_index];
#else
			return render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u[pixel_index];
#endif
		}
		else
		{
#if ReSTIR_DI_SpatialDirectionalReuseBitCount > 32
			return render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull[pixel_index];
#else
			return render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_u[pixel_index];
#endif
		}
	}

	/**
	 * Returns the shading normal or geometric normal of the given surface depending on the rejection heuristics settings
	 */
	template <bool IsReSTIRGI>
	HIPRT_HOST_DEVICE static float3 get_normal_for_rejection_heuristic(const HIPRTRenderData& render_data, const ReSTIRSurface& surface)
	{
		if constexpr (IsReSTIRGI)
		{
			if (render_data.render_settings.restir_gi_settings.neighbor_similarity_settings.reject_using_geometric_normals)
				return surface.geometric_normal;
			else
				return surface.shading_normal;
		}
		else
		{
			if (render_data.render_settings.restir_di_settings.neighbor_similarity_settings.reject_using_geometric_normals)
				return surface.geometric_normal;
			else
				return surface.shading_normal;
		}
	}
};

#endif
