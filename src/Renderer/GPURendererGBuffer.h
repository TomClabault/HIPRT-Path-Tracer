/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_G_BUFFER_H
#define GPU_RENDERER_G_BUFFER_H

#include "Device/includes/RayVolumeState.h"

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/Material.h"

// GBuffer that stores information about the current frame first hit data
struct GPURendererGBuffer
{
	void resize(unsigned int new_element_count)
	{
		materials.resize(new_element_count);
		geometric_normals.resize(new_element_count);
		shading_normals.resize(new_element_count);
		view_directions.resize(new_element_count);
		first_hits.resize(new_element_count);
		cameray_ray_hit.resize(new_element_count);
		ray_volume_states.resize(new_element_count);
	}

	void free()
	{
		materials.free();
		geometric_normals.free();
		shading_normals.free();
		view_directions.free();
		first_hits.free();
		cameray_ray_hit.free();
		ray_volume_states.free();
	}

	OrochiBuffer<SimplifiedRendererMaterial> materials;

	OrochiBuffer<float3> shading_normals;
	OrochiBuffer<float3> geometric_normals;
	OrochiBuffer<float3> view_directions;
	OrochiBuffer<float3> first_hits;

	OrochiBuffer<unsigned char> cameray_ray_hit;

	OrochiBuffer<RayVolumeState> ray_volume_states;
};

#endif
