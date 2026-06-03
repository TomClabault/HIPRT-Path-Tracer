/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_RESTI_SETTINGS_HELPER_H
#define HOST_DEVICE_COMMON_RESTI_SETTINGS_HELPER_H

#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"

template <int ReSTIRVariant>
struct ReSTIRSettingsTypeTemplate
{
};

template <>
struct ReSTIRSettingsTypeTemplate<ReSTIR_VARIANT_DI>
{
	using Type = ReSTIRDISettings;
};

template <>
struct ReSTIRSettingsTypeTemplate<ReSTIR_VARIANT_GI>
{
	using Type = ReSTIRGISettings;
};

template <>
struct ReSTIRSettingsTypeTemplate<ReSTIR_VARIANT_PT>
{
	using Type = ReSTIRPTSettings;
};

template <int ReSTIRVariant>
using ReSTIRSettingsType = typename ReSTIRSettingsTypeTemplate<ReSTIRVariant>::Type;

struct ReSTIRSettingsHelper
{
	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static ReSTIRSettingsType<ReSTIRVariant> get_restir_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static const ReSTIRCommonSpatialPassSettings& get_restir_spatial_pass_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.common_spatial_pass;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.common_spatial_pass;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.common_spatial_pass;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static ReSTIRCommonSpatialPassSettings& get_restir_spatial_pass_settings(HIPRTRenderData& render_data)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.common_spatial_pass;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.common_spatial_pass;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.common_spatial_pass;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	template <int ReSTIRVariant>
	HIPRT_DEVICE static ReSTIRCommonSPMISSettings& get_restir_spmis_settings(HIPRTRenderData& render_data)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.common_spatial_pass.spmis_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.common_spatial_pass.spmis_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	template <int ReSTIRVariant>
	HIPRT_DEVICE static const ReSTIRCommonSPMISSettings& get_restir_spmis_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.common_spatial_pass.spmis_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.common_spatial_pass.spmis_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.common_spatial_pass.spmis_settings;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static ReSTIRCommonTemporalPassSettings get_restir_temporal_pass_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.common_temporal_pass;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.common_temporal_pass;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.common_temporal_pass;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static ReSTIRCommonNeighborSimiliaritySettings get_restir_neighbor_similarity_settings(const HIPRTRenderData& render_data)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.neighbor_similarity_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.neighbor_similarity_settings;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.neighbor_similarity_settings;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	/**
	 * Returns the M value of a reservoir from the spatial pass input buffer given its pixel index
	 *
	 * The template argument can be used to select between ReSTIR DI and ReSTIR GI spatial buffers
	 */
	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static int get_restir_spatial_pass_input_reservoir_M(const HIPRTRenderData& render_data, int pixel_index)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.spatial_pass.input_reservoirs[pixel_index].M;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs[pixel_index].M;
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs[pixel_index].M;
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static unsigned long long int get_spatial_reuse_direction_mask_ull(const HIPRTRenderData& render_data, int pixel_index)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return render_data.render_settings.restir_di_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull[pixel_index];
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull[pixel_index];
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull[pixel_index];
		else
			static_assert(ReSTIRVariant == 0, "Invalid ReSTIR variant");
	}

	/**
	 * Returns the shading normal or geometric normal of the given surface depending on the rejection heuristics settings
	 */
	template <int ReSTIRVariant>
	HIPRT_HOST_DEVICE static float3_t get_normal_for_rejection_heuristic(const HIPRTRenderData& render_data, const ReSTIRSurface& surface)
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
		{
			if (render_data.render_settings.restir_di_settings.neighbor_similarity_settings.reject_using_geometric_normals)
				return surface.geometric_normal;
			else
				return surface.shading_normal;
		}
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
		{
			if (render_data.render_settings.restir_gi_settings.neighbor_similarity_settings.reject_using_geometric_normals)
				return surface.geometric_normal;
			else
				return surface.shading_normal;
		}
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
		{
			if (render_data.render_settings.restir_pt_settings.neighbor_similarity_settings.reject_using_geometric_normals)
				return surface.geometric_normal;
			else
				return surface.shading_normal;
		}
	}
};

#endif
