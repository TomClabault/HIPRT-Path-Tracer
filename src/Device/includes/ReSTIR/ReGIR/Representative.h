/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int ReGIR_pack_representative_point(const ReGIRSettings& regir_settings, float3 point_to_pack, unsigned int linear_cell_index)
{
	float3 relative_to_grid_cell_origin = point_to_pack - regir_settings.get_cell_origin_from_linear_cell_index(linear_cell_index);
	float3 point_in_grid_cell_0_1 = relative_to_grid_cell_origin / regir_settings.get_cell_size();
	float3 quantized = point_in_grid_cell_0_1 * 1023;
	int3 quantized_int = make_int3(quantized.x, quantized.y, quantized.z);

	unsigned int packed = 0;
	packed |= quantized_int.x;
	packed |= quantized_int.y << 10;
	packed |= quantized_int.z << 20;

	return packed;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_unpack_representative_point(const ReGIRSettings& regir_settings, unsigned int packed_rep_point, unsigned int linear_cell_index)
{
	unsigned int x = (packed_rep_point >> 00) & 0b1111111111;
	unsigned int y = (packed_rep_point >> 10) & 0b1111111111;
	unsigned int z = (packed_rep_point >> 20) & 0b1111111111;

	// This is a point in [0, 1] coordinates in a grid cell
	float3 unpacked_point = make_float3(x / 1023.0f, y / 1023.0f, z / 1023.0f);
	// "Unpacking" from [0, 1] to the grid cell coordinates
	float3 grid_cell_coordinates = unpacked_point * regir_settings.get_cell_size();
	// Bringing the point in world space by adding the origin of the grid cell
	float3 world_space = grid_cell_coordinates + regir_settings.get_cell_origin_from_linear_cell_index(linear_cell_index);

	return world_space;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_point(const HIPRTRenderData& render_data, float3 rep_point, int linear_cell_index)
{
	render_data.render_settings.regir_settings.representative.representative_points[linear_cell_index] = ReGIR_pack_representative_point(render_data.render_settings.regir_settings, rep_point, linear_cell_index);
}

HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_point(const HIPRTRenderData& render_data, float3 rep_point)
{
	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(rep_point);
	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
		// Outside of the grid
		return;

	ReGIR_store_representative_point(render_data, rep_point, linear_cell_index);
}

HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_normal(const HIPRTRenderData& render_data, float3 shading_normal, int linear_cell_index)
{
	render_data.render_settings.regir_settings.representative.representative_normals[linear_cell_index].pack(shading_normal);
}

HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_normal(const HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal)
{
	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
		// Outside of the grid
		return;

	ReGIR_store_representative_normal(render_data, shading_normal, linear_cell_index);
}

HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_primitive(const HIPRTRenderData& render_data, int primitive_index, int linear_cell_index)
{
	render_data.render_settings.regir_settings.representative.representative_primitive[linear_cell_index] = primitive_index;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_primitive(const HIPRTRenderData& render_data, float3 shading_point, int primitive_index)
{
	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
		// Outside of the grid
		return;

	ReGIR_store_representative_primitive(render_data, primitive_index, linear_cell_index);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_get_cell_representative_shading_normal(const HIPRTRenderData& render_data, int linear_cell_index)
{
	return render_data.render_settings.regir_settings.representative.representative_normals[linear_cell_index].unpack();
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_get_cell_representative_point(const HIPRTRenderData& render_data, int linear_cell_index)
{
	unsigned int rep_point_packed = render_data.render_settings.regir_settings.representative.representative_points[linear_cell_index];
	if (rep_point_packed == ReGIRRepresentative::UNDEFINED_POINT)
		return render_data.render_settings.regir_settings.get_cell_center_from_linear_cell_index(linear_cell_index);
	else
		return ReGIR_unpack_representative_point(render_data.render_settings.regir_settings, rep_point_packed, linear_cell_index);
}

HIPRT_HOST_DEVICE HIPRT_INLINE int ReGIR_get_cell_representative_primitive(const HIPRTRenderData& render_data, int linear_cell_index)
{
	return render_data.render_settings.regir_settings.representative.representative_primitive[linear_cell_index];
}

/**
 * If using the feature that "optimizes" representative points to be as close as possible to the cell center:
 * 		- this function stores the given pixel index into the representative point buffer if it is closer to
 *		the grid cell center than the current representative point
 * 
 * Otherwise:
 *		- this function always stores the given pixel index in the grid cell corresponding to the given shading point
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_update_representative_data(HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal, int primitive_index)
{
	if (DirectLightSamplingBaseStrategy != LSS_BASE_REGIR || render_data.render_settings.regir_settings.representative.distance_to_center == nullptr || !render_data.render_settings.regir_settings.use_representative_points)
		return;

	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
	float3 cell_center = render_data.render_settings.regir_settings.get_cell_center_from_linear_cell_index(linear_cell_index);
	float previous_distance_to_center = render_data.render_settings.regir_settings.representative.distance_to_center[linear_cell_index];
	float current_distance_to_center = hippt::length(cell_center - shading_point);
	if (previous_distance_to_center != ReGIRRepresentative::UNDEFINED_DISTANCE && current_distance_to_center < previous_distance_to_center)
	{
		// We have some representative data already, we're going to update it if we're closer to the
		// center of the cell than the previous representative data
		
		if (previous_distance_to_center < render_data.render_settings.regir_settings.get_cell_diagonal_length() * ReGIRRepresentative::OK_DISTANCE_TO_CENTER_FACTOR)
			// We're also only updating if we're not already close enough to the center.
			// 
			// Here, we're close enough to the center so our representative data is good and we don't need to update
			// anymore
			return;

		if (hippt::atomic_compare_exchange(&render_data.render_settings.regir_settings.representative.distance_to_center[linear_cell_index], previous_distance_to_center, ReGIRRepresentative::UNDEFINED_DISTANCE) == previous_distance_to_center)
		{
			ReGIR_store_representative_point(render_data, shading_point, linear_cell_index);
			ReGIR_store_representative_normal(render_data, shading_normal, linear_cell_index);
			ReGIR_store_representative_primitive(render_data, primitive_index, linear_cell_index);
			render_data.render_settings.regir_settings.representative.distance_to_center[linear_cell_index] = current_distance_to_center;

			return;
		}
	}

	if (render_data.render_settings.regir_settings.representative.representative_primitive[linear_cell_index] == ReGIRRepresentative::UNDEFINED_PRIMITIVE)
	{
		if (hippt::atomic_compare_exchange(&render_data.render_settings.regir_settings.representative.representative_primitive[linear_cell_index], ReGIRRepresentative::UNDEFINED_PRIMITIVE, primitive_index) == ReGIRRepresentative::UNDEFINED_PRIMITIVE)
		{
			ReGIR_store_representative_point(render_data, shading_point, linear_cell_index);
			ReGIR_store_representative_normal(render_data, shading_normal, linear_cell_index);

			render_data.render_settings.regir_settings.representative.distance_to_center[linear_cell_index] = current_distance_to_center;
		}
	}
}

#endif
