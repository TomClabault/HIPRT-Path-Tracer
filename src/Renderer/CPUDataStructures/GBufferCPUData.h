/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef G_BUFFER_CPU_RENDERER_H
#define G_BUFFER_CPU_RENDERER_H

#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Material/MaterialPacked.h"

#include <vector>

 // GBuffer that stores information about the current frame first hit data
struct GBufferCPUData
{
	void resize(unsigned int new_element_count)
	{
		materials.resize(new_element_count);
		geometric_normals.resize(new_element_count);
		shading_normals.resize(new_element_count);
		primary_hit_position.resize(new_element_count);
		first_hit_prim_index.resize(new_element_count);
		cameray_ray_hit.resize(new_element_count);
		ray_volume_states.resize(new_element_count);
	}

	std::vector<DevicePackedEffectiveMaterial> materials;
	std::vector<Octahedral24BitNormalPadded32b> geometric_normals;
	std::vector<Octahedral24BitNormalPadded32b> shading_normals;
	std::vector<float3> primary_hit_position;
	std::vector<int> first_hit_prim_index;

	std::vector<unsigned char> cameray_ray_hit;

	std::vector<RayVolumeState> ray_volume_states;
};

#endif
