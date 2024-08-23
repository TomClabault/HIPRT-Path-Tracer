/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SURFACE_H
#define DEVICE_RESTIR_DI_SURFACE_H

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Material.h"

struct ReSTIRDISurface
{
	RendererMaterial material;
	RayVolumeState ray_volume_state;
	float3 view_direction;
	float3 shading_normal;
	float3 shading_point;
};

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRDISurface get_pixel_surface(const HIPRTRenderData& render_data, int pixel_index)
{
	ReSTIRDISurface surface;

	surface.material = render_data.g_buffer.materials[pixel_index];
	surface.ray_volume_state = render_data.g_buffer.ray_volume_states[pixel_index];
	surface.view_direction = render_data.g_buffer.view_directions[pixel_index];
	surface.shading_normal = render_data.g_buffer.shading_normals[pixel_index];
	surface.shading_point = render_data.g_buffer.first_hits[pixel_index] + surface.shading_normal * 1.0e-4f;

	return surface;
}

#endif
