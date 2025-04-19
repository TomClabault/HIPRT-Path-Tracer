/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "HostDeviceCommon/RenderData.h"

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
    int pixel_index_for_representative_point = render_data.render_settings.regir_settings.get_cell_representative_pixel_index(linear_cell_index);
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
    int pixel_index_for_representative_point = render_data.render_settings.regir_settings.get_cell_representative_pixel_index(linear_cell_index);
    return ReGIR_get_cell_representative_point(render_data, linear_cell_index, pixel_index_for_representative_point);
}

/**
 * Use this function overload if you already have the 'pixel_index_for_representative_point' value.
 */
HIPRT_HOST_DEVICE int ReGIR_get_cell_representative_primitive(const HIPRTRenderData& render_data, int linear_cell_index, int pixel_index_for_representative_point)
{
    if (pixel_index_for_representative_point < 0 || pixel_index_for_representative_point   >= render_data.render_settings.render_resolution.x * render_data.render_settings.render_resolution.y)
        // No representative point yet, using the center of the cell
        return -1;
    else
        return render_data.g_buffer.first_hit_prim_index[pixel_index_for_representative_point];
}

HIPRT_HOST_DEVICE int ReGIR_get_cell_representative_primitive(const HIPRTRenderData& render_data, int linear_cell_index)
{
    int pixel_index_for_representative_point = render_data.render_settings.regir_settings.get_cell_representative_pixel_index(linear_cell_index);
    return ReGIR_get_cell_representative_primitive(render_data, linear_cell_index, pixel_index_for_representative_point);
}

#endif
