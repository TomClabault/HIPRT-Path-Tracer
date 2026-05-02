/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H

#include "Device/includes/Material.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

template <int NEEEstimator>
struct NEEDeferredMISContextSpecialized
{
	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F weighted_throughput, float bsdf_pdf) {}
};

template <>
struct NEEDeferredMISContextSpecialized<LSS_BSDF>
{
	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta / pdf
	ColorRGB32F last_bsdf_throughput;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
		last_ray_throughput = ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F weighted_throughput, float bsdf_pdf)
	{
		last_bsdf_throughput = weighted_throughput;
	}
};

template <>
struct NEEDeferredMISContextSpecialized<LSS_MIS_LIGHT_BSDF>
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;
	DeviceUnpackedEffectiveMaterial last_material;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta / pdf
	ColorRGB32F last_bsdf_throughput;
	float last_bsdf_sample_pdf;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
		last_view_direction = view_direction;
		last_shading_point	= closest_hit_info.inter_point;
		last_shading_normal = closest_hit_info.shading_normal;
		last_material		= material;
		last_ray_throughput = ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F weighted_throughput, float bsdf_pdf)
	{
		last_bsdf_throughput = weighted_throughput;
		last_bsdf_sample_pdf = bsdf_pdf;
	}
};

using NEEDeferredMISContext = NEEDeferredMISContextSpecialized<DirectLightNEEEstimator>;

#endif
