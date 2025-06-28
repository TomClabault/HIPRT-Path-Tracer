/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "Device/includes/ReSTIR/ReGIR/GridFillSurface.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_get_cell_world_shading_normal(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).world_normals[hash_grid_cell_index].unpack();
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_get_cell_world_point(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).world_points[hash_grid_cell_index];
}

HIPRT_HOST_DEVICE HIPRT_INLINE int ReGIR_get_cell_primitive_index(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).hit_primitive[hash_grid_cell_index];
}

HIPRT_HOST_DEVICE HIPRT_INLINE float ReGIR_get_cell_roughness(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	// / 255.0f to convert from uchar [0, 255] to float [0, 1]
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).roughness[hash_grid_cell_index] / 255.0f;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float ReGIR_get_cell_metallic(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	// / 255.0f to convert from uchar [0, 255] to float [0, 1]
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).metallic[hash_grid_cell_index] / 255.0f;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float ReGIR_get_cell_specular(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	// / 255.0f to convert from uchar [0, 255] to float [0, 1]
	return render_data.render_settings.regir_settings.get_hash_cell_data_soa(primary_hit).specular[hash_grid_cell_index] / 255.0f;
}

HIPRT_HOST_DEVICE HIPRT_INLINE ReGIRGridFillSurface ReGIR_get_cell_surface(const HIPRTRenderData& render_data, int hash_grid_cell_index, bool primary_hit)
{
	int cell_primitive_index = ReGIR_get_cell_primitive_index(render_data, hash_grid_cell_index, primary_hit);
	float3 cell_point = ReGIR_get_cell_world_point(render_data, hash_grid_cell_index, primary_hit);
	float3 cell_normal = ReGIR_get_cell_world_shading_normal(render_data, hash_grid_cell_index, primary_hit);
	float cell_roughness = ReGIR_get_cell_roughness(render_data, hash_grid_cell_index, primary_hit);
	float cell_metallic = ReGIR_get_cell_metallic(render_data, hash_grid_cell_index, primary_hit);
	float cell_specular = ReGIR_get_cell_specular(render_data, hash_grid_cell_index, primary_hit);

	ReGIRGridFillSurface surface;
	surface.cell_primitive_index = cell_primitive_index;
	surface.cell_point = cell_point;
	surface.cell_normal = cell_normal;
	surface.cell_roughness = cell_roughness;
	surface.cell_metallic = cell_metallic;
	surface.cell_specular = cell_specular;

	return surface;
}

/**
 *	Updates the representative point and normal (and other data) of the cell at the given shading point
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_update_representative_data(HIPRTRenderData& render_data, float3 shading_point, float3 shading_normal, const HIPRTCamera& current_camera, int primitive_index, bool primary_hit, const DeviceUnpackedEffectiveMaterial& material)
{
	if (DirectLightSamplingBaseStrategy != LSS_BASE_REGIR)
		return;
	else if (primitive_index == -1)
		return;

	render_data.render_settings.regir_settings.insert_hash_cell_data(render_data.render_settings.regir_settings.shading, shading_point, shading_normal, current_camera, primary_hit, primitive_index, material);
}

#endif
