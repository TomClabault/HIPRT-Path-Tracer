/*
* Copyright 2025 Tom Clabault. GNU GPL3 license.
* GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
*/

#ifndef DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
#define DEVICE_KERNELS_REGIR_REPRESENTATIVE_H
 
#include "Device/includes/ReSTIR/ReGIR/GridFillSurface.h"
#include "Device/includes/ReSTIR/ReGIR/HashGridCellData.h"

#include "HostDeviceCommon/RenderData.h"

/**
 *	Updates the representative point and normal (and other data) of the cell at the given shading point
 */
HIPRT_DEVICE void ReGIR_update_representative_data(HIPRTRenderData& render_data, float3 shading_point, float3 surface_normal, const HIPRTCamera& current_camera, int primitive_index, bool primary_hit, const DeviceUnpackedEffectiveMaterial& material)
{
	if (DirectLightSamplingBaseStrategy != LSS_BASE_REGIR)
		return;
	else if (primitive_index == -1)
		return;
	else if (render_data.buffers.emissive_triangles_count == 0)
		return;

	// We're using the packed-unpacked surface normal here because
	// packing/unpacking (as used in the G-Buffer) normals introduces
	// small differences that are enough to shift us from one cell to
	// another.
	//
	// In practice this leads to the cell being inserted into the hash grid
	// with non-packed normals but then when the cell is queried at the first
	// during the path tracing kernels (and thus with the packed + unpacked normal
	// from the G-Buffer we get different hashing results and we can't find our cells
	// back)
	surface_normal = Octahedral24BitNormalPadded32b(surface_normal).unpack();

	render_data.render_settings.regir_settings.insert_hash_cell_data(shading_point, surface_normal, current_camera, primary_hit, primitive_index, material);
}

#endif
