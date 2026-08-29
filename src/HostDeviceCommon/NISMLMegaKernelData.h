/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_NISML_MEGA_KERNEL_DATA_H
#define HOST_DEVICE_COMMON_NISML_MEGA_KERNEL_DATA_H

#include "Device/includes/HitInfo.h"
#include "Device/includes/RayVolumeState.h"
#include "HostDeviceCommon/AtomicType.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

enum class NISMLMegaKernelPathState : unsigned int
{
	UNINITIALIZED,
	WAITING_FOR_NIS,
	READY_TO_RESUME,
	FINISHED
};

struct NISMLMegaKernelPathData
{
	ColorRGB32F throughput = ColorRGB32F(1.0f);
	ColorRGB32F ray_color  = ColorRGB32F(0.0f);

	unsigned int next_ray_state = 0;
	int bounce					= 0;
	float accumulated_roughness = 0.0f;

	DeviceUnpackedEffectiveMaterial material;
	HitInfo closest_hit_info;
	float3_t ray_origin				= make_float3(0.0f, 0.0f, 0.0f);
	float3_t ray_direction			= make_float3(0.0f, 0.0f, 0.0f);
	unsigned int intersection_found = 0;
	unsigned int query_index		= 0;
	unsigned int rng_state			= 0;
};

struct NISQuery
{
	float3_t position			= make_float3(0.0f, 0.0f, 0.0f);
	float3_t outgoing_direction = make_float3(0.0f, 0.0f, 0.0f);
	float3_t normal				= make_float3(0.0f, 0.0f, 0.0f);

	float sg_specular_weight = 0.0f;
	float alpha_x			 = 0.0f;
	float alpha_y			 = 0.0f;

	unsigned int path_index = 0;
	unsigned int rng_state	= 0;
};

struct NISResult
{
	int emissive_triangle_global_index	= -1;
	unsigned int cluster_index			= 0;
	float cluster_probability			= 0.0f;
	float conditional_light_probability = 0.0f;
	float emissive_triangle_pdf			= 0.0f;
	unsigned int rng_state				= 0;
};

struct NISMLMegaKernelDevice
{
	NISMLMegaKernelPathData* path_data	  = nullptr;
	NISMLMegaKernelPathState* path_states = nullptr;
	RayVolumeState* path_volume_states	  = nullptr;

	NISQuery* queries					  = nullptr;
	float* residuals					  = nullptr;
	NISResult* results					  = nullptr;
	AtomicType<unsigned int>* query_count = nullptr;

	unsigned int path_count		 = 0;
	unsigned int query_capacity	 = 0;
	unsigned int residual_stride = 0;
};

#endif // #ifndef HOST_DEVICE_COMMON_NISML_MEGA_KERNEL_DATA_H
