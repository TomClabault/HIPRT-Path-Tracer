/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_SURFACE_H
#define DEVICE_RESTIR_DI_SURFACE_H

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"

struct ReSTIRSurface
{
	DeviceUnpackedEffectiveMaterial material;
	RayVolumeState ray_volume_state;
	int primitive_index;

	// Do we need the view direction here? We can probably reconstruct it
	float3 view_direction = { 0.0f, 0.0f, 0.0f};
	float3 shading_normal = { 0.0f, 0.0f, 0.0f};
	float3 geometric_normal = { 0.0f, 0.0f, 0.0f};
	float3 shading_point = { 0.0f, 0.0f, 0.0f };
};

HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRSurface get_pixel_surface(const HIPRTRenderData& render_data, int pixel_index, Xorshift32Generator& random_number_generator)
{
	ReSTIRSurface surface;

	surface.material = render_data.g_buffer.materials[pixel_index].unpack();
	surface.primitive_index = render_data.g_buffer.first_hit_prim_index[pixel_index];
	surface.ray_volume_state.reconstruct_first_hit(
		surface.material,
		render_data.buffers.material_indices,
		surface.primitive_index,
		random_number_generator);

	surface.view_direction = render_data.g_buffer.get_view_direction(render_data.current_camera.position, pixel_index);
	surface.shading_normal = render_data.g_buffer.shading_normals[pixel_index].unpack();
	surface.geometric_normal = render_data.g_buffer.geometric_normals[pixel_index].unpack();
	surface.shading_point = render_data.g_buffer.primary_hit_position[pixel_index];

	return surface;
}

/**
 * Returns the surface at a pixel in the previous frame (so before the camera moved if it is in motion)
 * This is needed for unbiasedness in motion in the temporal reuse pass because when we count the neighbors
 * that could have produced the sample that we picked, we need to consider the neighbors at their previous positions,
 * not the current so we need to read in the last frame's g-buffer.
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRSurface get_pixel_surface_previous_frame(const HIPRTRenderData& render_data, int pixel_index, Xorshift32Generator& random_number_generator)
{
	ReSTIRSurface surface;

	surface.material = render_data.g_buffer_prev_frame.materials[pixel_index].unpack();
	surface.primitive_index = render_data.g_buffer_prev_frame.first_hit_prim_index[pixel_index];
	surface.ray_volume_state.reconstruct_first_hit(
		surface.material,
		render_data.buffers.material_indices,
		surface.primitive_index,
		random_number_generator);

	surface.view_direction = render_data.g_buffer.get_view_direction(render_data.prev_camera.position, pixel_index);
	surface.shading_normal = render_data.g_buffer_prev_frame.shading_normals[pixel_index].unpack();
	surface.geometric_normal = render_data.g_buffer_prev_frame.geometric_normals[pixel_index].unpack();
	surface.shading_point = render_data.g_buffer_prev_frame.primary_hit_position[pixel_index];

	return surface;
}

/**
 * Simple overload of the function to base the 'previous_frame' decision on a boolean instead of on the name of the function
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ReSTIRSurface get_pixel_surface(const HIPRTRenderData& render_data, int pixel_index, bool previous_frame, Xorshift32Generator& random_number_generator)
{
	if (previous_frame)
		return get_pixel_surface_previous_frame(render_data, pixel_index, random_number_generator);
	else
		return get_pixel_surface(render_data, pixel_index, random_number_generator);
}

#endif
