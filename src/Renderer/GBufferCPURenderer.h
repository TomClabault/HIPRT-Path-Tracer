/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef G_BUFFER_CPU_RENDERER_H
#define G_BUFFER_CPU_RENDERER_H

#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Material/Material.h"

#include <vector>

 // GBuffer that stores information about the current frame first hit data
struct GBufferCPURenderer
{
	void resize(unsigned int new_element_count)
	{
		materials.resize(new_element_count);
		geometric_normals.resize(new_element_count);
		shading_normals.resize(new_element_count);
		primary_hits.resize(new_element_count);
		first_hit_prim_index.resize(new_element_count);
		cameray_ray_hit.resize(new_element_count);
		ray_volume_states.resize(new_element_count);
	}

	std::vector<DevicePackedEffectiveMaterial> materials;
	std::vector<Octahedral24BitNormal> geometric_normals;
	std::vector<Octahedral24BitNormal> shading_normals;
	std::vector<float3> primary_hits;
	std::vector<int> first_hit_prim_index;

	std::vector<unsigned char> cameray_ray_hit;

	std::vector<RayVolumeState> ray_volume_states;
};

#endif
