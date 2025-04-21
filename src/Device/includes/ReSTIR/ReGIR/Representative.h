/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE int ReGIR_get_cell_representative_pixel_index(const HIPRTRenderData& render_data, int linear_cell_index)
{
	return render_data.render_settings.regir_settings.grid_fill.representative_points_pixel_index[linear_cell_index];
}

HIPRT_HOST_DEVICE void ReGIR_store_representative_point_pixel_index(const HIPRTRenderData& render_data, int linear_cell_index, int pixel_index)
{
	render_data.render_settings.regir_settings.grid_fill.representative_points_pixel_index[linear_cell_index] = pixel_index;
}

HIPRT_HOST_DEVICE void ReGIR_store_representative_point_pixel_index(const HIPRTRenderData& render_data, float3 shading_point, int pixel_index)
{
	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
	if (linear_cell_index < 0 || linear_cell_index >= render_data.render_settings.regir_settings.grid.grid_resolution.x * render_data.render_settings.regir_settings.grid.grid_resolution.y * render_data.render_settings.regir_settings.grid.grid_resolution.z)
		// Outside of the grid
		return;

	ReGIR_store_representative_point_pixel_index(render_data, linear_cell_index, pixel_index);
}

/**
 * Use this function overload if you already have the 'pixel_index_for_representative_point' value.
 */
HIPRT_HOST_DEVICE float3 ReGIR_get_cell_representative_shading_normal(const HIPRTRenderData& render_data, int linear_cell_index, int pixel_index_for_representative_point)
{
	if (pixel_index_for_representative_point < 0 || pixel_index_for_representative_point >= render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y)
		// No representative point yet, using the center of the cell
		return make_float3(0.0f, 0.0f, 0.0f);
	else
		return render_data.g_buffer.shading_normals[pixel_index_for_representative_point].unpack();
}

HIPRT_HOST_DEVICE float3 ReGIR_get_cell_representative_shading_normal(const HIPRTRenderData& render_data, int linear_cell_index)
{
	int pixel_index_for_representative_point = ReGIR_get_cell_representative_pixel_index(render_data, linear_cell_index);
	return ReGIR_get_cell_representative_shading_normal(render_data, linear_cell_index, pixel_index_for_representative_point);
}

/**
 * Use this function overload if you already have the 'pixel_index_for_representative_point' value.
 */
HIPRT_HOST_DEVICE float3 ReGIR_get_cell_representative_point(const HIPRTRenderData& render_data, int linear_cell_index, int pixel_index_for_representative_point)
{
	if (pixel_index_for_representative_point < 0 || pixel_index_for_representative_point >= render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y)
		// No representative point yet, using the center of the cell
		return render_data.render_settings.regir_settings.get_cell_center_from_linear(linear_cell_index);
	else
		return render_data.g_buffer.primary_hit_position[pixel_index_for_representative_point];
}

HIPRT_HOST_DEVICE float3 ReGIR_get_cell_representative_point(const HIPRTRenderData& render_data, int linear_cell_index)
{
	int pixel_index_for_representative_point = ReGIR_get_cell_representative_pixel_index(render_data, linear_cell_index);
	return ReGIR_get_cell_representative_point(render_data, linear_cell_index, pixel_index_for_representative_point);
}

/**
 * Use this function overload if you already have the 'pixel_index_for_representative_point' value.
 */
HIPRT_HOST_DEVICE int ReGIR_get_cell_representative_primitive(const HIPRTRenderData& render_data, int linear_cell_index, int pixel_index_for_representative_point)
{
	if (pixel_index_for_representative_point < 0 || pixel_index_for_representative_point >= render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y)
		// No representative point yet, using the center of the cell
		return -1;
	else
		return render_data.g_buffer.first_hit_prim_index[pixel_index_for_representative_point];
}

HIPRT_HOST_DEVICE int ReGIR_get_cell_representative_primitive(const HIPRTRenderData& render_data, int linear_cell_index)
{
	int pixel_index_for_representative_point = ReGIR_get_cell_representative_pixel_index(render_data, linear_cell_index);
	return ReGIR_get_cell_representative_primitive(render_data, linear_cell_index, pixel_index_for_representative_point);
}

/**
 * If using the feature that "optimizes" representative points to be as close as possible to the cell center:
 * 		- this function stores the given pixel index into the representative point buffer if it is closer to
 *		the grid cell center than the current representative point
 * 
 * Otherwise:
 *		- this function always stores the given pixel index in the grid cell corresponding to the given shading point
 */
HIPRT_HOST_DEVICE void ReGIR_update_representative_point_pixel_index(HIPRTRenderData& render_data, float3 shading_point, int pixel_index)
{
	if (!render_data.render_settings.regir_settings.optimize_representative_points_at_center_of_cell)
	{
		// If we are not aiming at optimizing reprensetative points to be at the center of the grid cells
		// just storing no matter what

		ReGIR_store_representative_point_pixel_index(render_data, shading_point, pixel_index);
		return;
	}

	int linear_cell_index = render_data.render_settings.regir_settings.get_linear_cell_index_from_world_pos(shading_point);
	if (ReGIR_get_cell_representative_pixel_index(render_data, linear_cell_index) != -1)
	{
		// If we already have a representative point for that grid cell, we're going to replace it with our new point
		// if our new point is closer to the center of the grid cell than the point that is currently stored as
		// representative point
		float3 current_point = ReGIR_get_cell_representative_point(render_data, linear_cell_index);
		float3 cell_center = render_data.render_settings.regir_settings.get_cell_center_from_linear(linear_cell_index);

		if (hippt::length2(cell_center - shading_point) < hippt::length2(cell_center - current_point))
			// If the new point is closer to the cell center, storing it
			ReGIR_store_representative_point_pixel_index(render_data, linear_cell_index, pixel_index);
	}
	else
		// If we do not have a representative point already, just storing the current point
		ReGIR_store_representative_point_pixel_index(render_data, linear_cell_index, pixel_index);
}

#endif
