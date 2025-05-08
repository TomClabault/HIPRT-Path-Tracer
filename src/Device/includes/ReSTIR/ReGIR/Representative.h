/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"

#include "HostDeviceCommon/RenderData.h"

//HIPRT_HOST_DEVICE HIPRT_INLINE unsigned int ReGIR_pack_representative_point(const ReGIRSettings& regir_settings, float3 point_to_pack, unsigned int hash_grid_cell_index)
//{
//	float3 relative_to_grid_cell_origin = point_to_pack - regir_settings.get_cell_origin_from_hash_grid_cell_index(hash_grid_cell_index);
//	float3 point_in_grid_cell_0_1 = relative_to_grid_cell_origin / regir_settings.get_cell_size();
//	float3 quantized = point_in_grid_cell_0_1 * 1023;
//	int3 quantized_int = make_int3(quantized.x, quantized.y, quantized.z);
//
//	unsigned int packed = 0;
//	packed |= quantized_int.x;
//	packed |= quantized_int.y << 10;
//	packed |= quantized_int.z << 20;
//
//	return packed;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_unpack_representative_point(float3 cell_size, float3 cell_origin, unsigned int packed_rep_point, unsigned int hash_grid_cell_index)
//{
//	unsigned int x = (packed_rep_point >> 00) & 0b1111111111;
//	unsigned int y = (packed_rep_point >> 10) & 0b1111111111;
//	unsigned int z = (packed_rep_point >> 20) & 0b1111111111;
//
//	// This is a point in [0, 1] coordinates in a grid cell
//	float3 unpacked_point = make_float3(x / 1023.0f, y / 1023.0f, z / 1023.0f);
//	// "Unpacking" from [0, 1] to the grid cell coordinates
//	float3 grid_cell_coordinates = unpacked_point * cell_size;
//	// Bringing the point in world space by adding the origin of the grid cell
//	float3 world_space = grid_cell_coordinates + cell_origin;
//
//	return world_space;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE float3 ReGIR_unpack_representative_point(const ReGIRSettings& regir_settings, unsigned int packed_rep_point, unsigned int hash_grid_cell_index)
//{
//	return ReGIR_unpack_representative_point(regir_settings.get_cell_size(), regir_settings.get_cell_origin_from_hash_grid_cell_index(hash_grid_cell_index), packed_rep_point, hash_grid_cell_index);
//}

//HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_point(const HIPRTRenderData& render_data, float3 rep_point, int hash_grid_cell_index)
//{
//	// render_data.render_settings.regir_settings.grid_fill_grid.hash_cell_data.world_points[hash_grid_cell_index] = ReGIR_pack_representative_point(render_data.render_settings.regir_settings, rep_point, hash_grid_cell_index);
//	render_data.render_settings.regir_settings.grid_fill_grid.hash_cell_data.world_points[hash_grid_cell_index] = rep_point;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_point(const HIPRTRenderData& render_data, float3 rep_point, float3 camera_position)
//{
//	int hash_grid_cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos_no_collision_resolve(rep_point, camera_position);
//
//	ReGIR_store_representative_point(render_data, rep_point, hash_grid_cell_index);
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_normal(const HIPRTRenderData& render_data, float3 shading_normal, int hash_grid_cell_index)
//{
//	render_data.render_settings.regir_settings.grid_fill_grid.hash_cell_data.world_normals[hash_grid_cell_index].pack(shading_normal);
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_normal(const HIPRTRenderData& render_data, float3 shading_point, float3 camera_position, float3 shading_normal)
//{
//	int hash_grid_cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos_no_collision_resolve(shading_point, camera_position);
//
//	ReGIR_store_representative_normal(render_data, shading_normal, hash_grid_cell_index);
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_primitive(const HIPRTRenderData& render_data, int primitive_index, int hash_grid_cell_index)
//{
//	render_data.render_settings.regir_settings.grid_fill_grid.hash_cell_data.representative_primitive[hash_grid_cell_index] = primitive_index;
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_representative_primitive(const HIPRTRenderData& render_data, float3 shading_point, float3 camera_position, int primitive_index)
//{
//	int hash_grid_cell_index = render_data.render_settings.regir_settings.get_hash_grid_cell_index_from_world_pos_no_collision_resolve(shading_point, camera_position);
//
//	ReGIR_store_representative_primitive(render_data, primitive_index, hash_grid_cell_index);
//}
//
//HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_store_hash_cell_hash(const HIPRTRenderData& render_data, unsigned int hash)
//{
//	render_data.render_settings.regir_settings.grid_fill_grid.hash_cell_data.hash[hash] = hash;
//}

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
HIPRT_HOST_DEVICE HIPRT_INLINE void ReGIR_update_representative_data(HIPRTRenderData& render_data, float3 shading_point, float3 camera_position, float3 shading_normal, int primitive_index)
{
	if (DirectLightSamplingBaseStrategy != LSS_BASE_REGIR)
		return;
	else if (primitive_index == -1)
		return;

	render_data.render_settings.regir_settings.insert_hash_cell_data<true>(render_data.render_settings.regir_settings.shading, shading_point, camera_position, shading_normal, primitive_index);
}

#endif
