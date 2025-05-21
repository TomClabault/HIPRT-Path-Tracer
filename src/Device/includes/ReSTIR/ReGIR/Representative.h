/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_get_cell_world_shading_normal(const HIPRTRenderData& render_data, int hash_grid_cell_index)
{
	return render_data.render_settings.regir_settings.hash_cell_data.world_normals[hash_grid_cell_index].unpack();
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_get_cell_world_point(const HIPRTRenderData& render_data, int hash_grid_cell_index)
{
	/*unsigned int rep_point_packed = render_data.render_settings.regir_settings.grid_fill_grid.hash_cell_data.world_points[hash_grid_cell_index];
	if (rep_point_packed == ReGIRHashCellDataSoADevice::UNDEFINED_POINT)
		return render_data.render_settings.regir_settings.get_cell_center_from_hash_grid_cell_index(hash_grid_cell_index);
	else
		return ReGIR_unpack_representative_point(render_data.render_settings.regir_settings, rep_point_packed, hash_grid_cell_index);*/

	return render_data.render_settings.regir_settings.hash_cell_data.world_points[hash_grid_cell_index];
}

HIPRT_HOST_DEVICE HIPRT_INLINE int ReGIR_get_cell_primitive_index(const HIPRTRenderData& render_data, int hash_grid_cell_index)
{
	return render_data.render_settings.regir_settings.hash_cell_data.hit_primitive[hash_grid_cell_index];
}

/**
 * If using the feature that "optimizes" hash_cell_data points to be as close as possible to the cell center:
 * 		- this function stores the given pixel index into the hash_cell_data point buffer if it is closer to
 *		the grid cell center than the current hash_cell_data point
 * 
 * Otherwise:
 *		- this function always stores the given pixel index in the grid cell corresponding to the given shading point
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_update_representative_data(HIPRTRenderData& render_data, float3 shading_point, const HIPRTCamera& current_camera, float3 shading_normal, int primitive_index)
{
	if (DirectLightSamplingBaseStrategy != LSS_BASE_REGIR)
		return;
	else if (primitive_index == -1)
		return;

	render_data.render_settings.regir_settings.insert_hash_cell_data(render_data.render_settings.regir_settings.shading, shading_point, current_camera, shading_normal, primitive_index);
}

#endif
