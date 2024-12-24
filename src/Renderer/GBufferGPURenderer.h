/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef G_BUFFER_GPU_RENDERER_H
#define G_BUFFER_GPU_RENDERER_H

#include "Device/includes/RayVolumeState.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/Material/Material.h"
#include "HostDeviceCommon/GBufferDevice.h"

// GBuffer that stores information about the current frame first hit data
struct GBufferGPURenderer
{
	void resize(unsigned int new_element_count, size_t ray_volume_state_byte_size)
	{
		materials.resize(new_element_count);
		geometric_normals.resize(new_element_count);
		shading_normals.resize(new_element_count);
		first_hits.resize(new_element_count);
		first_hit_prim_index.resize(new_element_count);
		cameray_ray_hit.resize(new_element_count);

		// We need to be careful here because the ray volume states contain the nested dielectric stack and the stack size can be changed at runtime through ImGui. However, on the CPU, the stack size is determined at compile time. Changing the stack size through ImGui only resizes the GPU shaders which then adapts to the new stack size thanks to the recompilation. However, on the CPU, we're not recompiling anything. This means that the stack size on the CPU doesn't match the stack size on the GPU anymore and the buffer will not be properly resized --> this is huge undefined behavior.
		// To avoid that, we're manually giving the size here for resizing
		ray_volume_states.resize(new_element_count, ray_volume_state_byte_size);
	}

	void free()
	{
		materials.free();
		geometric_normals.free();
		shading_normals.free();
		first_hits.free();
		first_hit_prim_index.free();
		cameray_ray_hit.free();
		ray_volume_states.free();
	}

	GBufferDevice get_device_g_buffer()
	{
		GBufferDevice out;

		out.materials = materials.get_device_pointer();
		out.geometric_normals = geometric_normals.get_device_pointer();
		out.shading_normals = shading_normals.get_device_pointer();
		out.first_hits = first_hits.get_device_pointer();
		out.first_hit_prim_index = first_hit_prim_index.get_device_pointer();
		out.camera_ray_hit = cameray_ray_hit.get_device_pointer();
		out.ray_volume_states = ray_volume_states.get_device_pointer();

		return out;
	}

	OrochiBuffer<DevicePackedEffectiveMaterial> materials;

	OrochiBuffer<float3> shading_normals;
	OrochiBuffer<float3> geometric_normals;
	OrochiBuffer<float3> first_hits;
	OrochiBuffer<int> first_hit_prim_index;

	OrochiBuffer<unsigned char> cameray_ray_hit;

	OrochiBuffer<RayVolumeState> ray_volume_states;
};

#endif
