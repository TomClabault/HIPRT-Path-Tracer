/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "HostDeviceCommon/RenderData.h"

//HIPRT_HOST_DEVICE int ReGIR_get_cell_representative_pixel_index(const HIPRTRenderData& render_data, int linear_cell_index)
//{
//	return render_data.render_settings.regir_settings.grid_fill.representative_points_pixel_index[linear_cell_index];
//}

//HIPRT_HOST_DEVICE void ReGIR_store_representative_point_pixel_index(const HIPRTRenderData& render_data, int linear_cell_index, int pixel_index)
//{
//	render_data.render_settings.regir_settings.grid_fill.representative_points_pixel_index[linear_cell_index] = pixel_index;
//}
//
//HIPRT_HOST_DEVICE void ReGIR_store_representative_point_pixel_index(const HIPRTRenderData& render_data, float3 shading_point, int pixel_index)
//{
//	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
//	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
//		// Outside of the grid
//		return;
//
//	ReGIR_store_representative_point_pixel_index(render_data, linear_cell_index, pixel_index);
//}

HIPRT_HOST_DEVICE void ReGIR_store_representative_point(const HIPRTRenderData& render_data, float3 rep_point, int linear_cell_index)
{
	render_data.render_settings.regir_settings.representative.representative_points[linear_cell_index] = rep_point;
}

HIPRT_HOST_DEVICE void ReGIR_store_representative_point(const HIPRTRenderData& render_data, float3 rep_point)
{
	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(rep_point);
	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
		// Outside of the grid
		return;

	ReGIR_store_representative_point(render_data, rep_point, linear_cell_index);
}

HIPRT_HOST_DEVICE void ReGIR_store_representative_normal(const HIPRTRenderData& render_data, float3 shading_normal, int linear_cell_index)
{
	render_data.render_settings.regir_settings.representative.representative_normals[linear_cell_index] = shading_normal;
}

HIPRT_HOST_DEVICE void ReGIR_store_representative_normal(const HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal)
{
	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
		// Outside of the grid
		return;

	ReGIR_store_representative_normal(render_data, shading_normal, linear_cell_index);
}

HIPRT_HOST_DEVICE void ReGIR_store_representative_primitive(const HIPRTRenderData& render_data, int primitive_index, int linear_cell_index)
{
	render_data.render_settings.regir_settings.representative.representative_primitive[linear_cell_index] = primitive_index;
}

HIPRT_HOST_DEVICE void ReGIR_store_representative_primitive(const HIPRTRenderData& render_data, float3 shading_point, int primitive_index)
{
	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
		// Outside of the grid
		return;

	ReGIR_store_representative_primitive(render_data, primitive_index, linear_cell_index);
}

HIPRT_HOST_DEVICE float3 ReGIR_get_cell_representative_shading_normal(const HIPRTRenderData& render_data, int linear_cell_index)
{
	return render_data.render_settings.regir_settings.representative.representative_normals[linear_cell_index];
}

HIPRT_HOST_DEVICE float3 ReGIR_get_cell_representative_point(const HIPRTRenderData& render_data, int linear_cell_index)
{
	float3 rep_point = render_data.render_settings.regir_settings.representative.representative_points[linear_cell_index];
	if (rep_point.x == ReGIRRepresentative::UNDEFINED_POINT.x)
		return render_data.render_settings.regir_settings.get_cell_center_from_linear_cell_index(linear_cell_index);
	else
		return rep_point;
}

HIPRT_HOST_DEVICE int ReGIR_get_cell_representative_primitive(const HIPRTRenderData& render_data, int linear_cell_index)
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
HIPRT_HOST_DEVICE void ReGIR_update_representative_data(HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal, int primitive_index)
{
	if (render_data.render_settings.regir_settings.representative.distance_to_center == nullptr || !render_data.render_settings.regir_settings.use_representative_points)
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
