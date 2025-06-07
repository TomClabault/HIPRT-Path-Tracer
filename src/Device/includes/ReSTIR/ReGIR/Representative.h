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
	return render_data.render_settings.regir_settings.hash_cell_data.world_points[hash_grid_cell_index];
}

HIPRT_HOST_DEVICE HIPRT_INLINE int ReGIR_get_cell_primitive_index(const HIPRTRenderData& render_data, int hash_grid_cell_index)
{
	return render_data.render_settings.regir_settings.hash_cell_data.hit_primitive[hash_grid_cell_index];
}

HIPRT_HOST_DEVICE HIPRT_INLINE float ReGIR_get_cell_roughness(const HIPRTRenderData& render_data, int hash_grid_cell_index)
{
	return render_data.render_settings.regir_settings.hash_cell_data.roughness[hash_grid_cell_index];
}

HIPRT_HOST_DEVICE HIPRT_INLINE float ReGIR_get_cell_metallic(const HIPRTRenderData& render_data, int hash_grid_cell_index)
{
	return render_data.render_settings.regir_settings.hash_cell_data.metallic[hash_grid_cell_index];
}

HIPRT_HOST_DEVICE HIPRT_INLINE float ReGIR_get_cell_specular(const HIPRTRenderData& render_data, int hash_grid_cell_index)
{
	return render_data.render_settings.regir_settings.hash_cell_data.specular[hash_grid_cell_index];
}

/**
 *	Updates the representative point and normal (and other data) of the cell at the given shading point
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_update_representative_data(HIPRTRenderData& render_data, float3 shading_point, const HIPRTCamera& current_camera, float3 shading_normal, int primitive_index, const DeviceUnpackedEffectiveMaterial& material)
{
	if (DirectLightSamplingBaseStrategy != LSS_BASE_REGIR)
		return;
	else if (primitive_index == -1)
		return;

	render_data.render_settings.regir_settings.insert_hash_cell_data(render_data.render_settings.regir_settings.shading, shading_point, current_camera, shading_normal, primitive_index, material);
}

#endif
