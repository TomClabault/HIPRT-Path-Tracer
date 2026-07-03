/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H

#include "Device/includes/LightSampling/RIS/RISReservoir.h"
#include "Device/includes/Material.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

template <int NEEEstimator, int PathIntegrator = PathSamplingStrategy>
struct NEEDeferredMISContextSpecialized
{
	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf) {}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir) {}
};

template <int PathIntegrator>
struct NEEDeferredMISContextSpecialized<LSS_BSDF, PathIntegrator>
{
	float3_t last_shading_point;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
		last_shading_point	= closest_hit_info.inter_point;
		last_ray_throughput = ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}
};

template <int PathIntegrator>
struct NEEDeferredMISContextSpecialized<LSS_MIS_LIGHT_BSDF, PathIntegrator>
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;
	DeviceUnpackedEffectiveMaterial last_material;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
		last_view_direction = view_direction;
		last_shading_point	= closest_hit_info.inter_point;
		last_shading_normal = closest_hit_info.shading_normal;
		last_material		= material;
		last_ray_throughput = ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir) {}
};

template <int PathIntegrator>
struct NEEDeferredMISContextSpecialized<LSS_RIS_BSDF_AND_LIGHT, PathIntegrator>
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;
	float3_t last_geometric_normal;
	DeviceUnpackedEffectiveMaterial last_material;
	RayVolumeState last_volume_state;

	int last_primitive_index;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;

	RISReservoir ris_reservoir;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
		last_view_direction	  = view_direction;
		last_shading_point	  = closest_hit_info.inter_point;
		last_shading_normal	  = closest_hit_info.shading_normal;
		last_geometric_normal = closest_hit_info.geometric_normal;
		last_material		  = material;
		last_volume_state	  = volume_state;
		last_primitive_index  = closest_hit_info.primitive_index;
		last_ray_throughput	  = ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir)
	{
		ris_reservoir = reservoir;
	}
};

template <>
struct NEEDeferredMISContextSpecialized<LSS_RIS_BSDF_AND_LIGHT, PATH_SAMPLING_RESTIR_PT>
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;
	float3_t last_geometric_normal;
	RayVolumeState last_volume_state;

	int last_primitive_index;
	float2_t last_texcoords;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta
	ColorRGB32F last_bsdf_x_cos_theta;
	float last_bsdf_sample_pdf;
	BSDFIncidentLightInfo last_bsdf_incident_light_info;

	RISReservoir ris_reservoir;

	HIPRT_DEVICE void fill_last_hit_information(HitInfo& closest_hit_info,
												const float3_t& view_direction,
												const RayVolumeState& volume_state,
												const DeviceUnpackedEffectiveMaterial& material,
												const ColorRGB32F& ray_throughput)
	{
		last_view_direction	  = view_direction;
		last_shading_point	  = closest_hit_info.inter_point;
		last_shading_normal	  = closest_hit_info.shading_normal;
		last_geometric_normal = closest_hit_info.geometric_normal;
		last_volume_state	  = volume_state;
		last_primitive_index  = closest_hit_info.primitive_index;
		last_texcoords		  = closest_hit_info.texcoords;
		last_ray_throughput	  = ray_throughput;
	}

	HIPRT_DEVICE void fill_last_bsdf_information(ColorRGB32F bsdf_cos_theta, float bsdf_pdf)
	{
		last_bsdf_x_cos_theta = bsdf_cos_theta;
		last_bsdf_sample_pdf  = bsdf_pdf;
	}

	HIPRT_DEVICE void fill_ris_reservoir(const RISReservoir& reservoir)
	{
		ris_reservoir = reservoir;
	}

	HIPRT_DEVICE DeviceUnpackedEffectiveMaterial get_last_material(const HIPRTRenderData& render_data) const
	{
		int last_material_index = render_data.buffers.material_indices[last_primitive_index];

		return get_intersection_material(render_data, last_material_index, last_texcoords);
	}
};

using NEEDeferredMISContext = NEEDeferredMISContextSpecialized<DirectLightNEEEstimator>;

#endif
