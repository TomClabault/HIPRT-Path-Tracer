/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_MIS_RAY_REUSE_H
#define DEVICE_MIS_RAY_REUSE_H

#include "Device/includes/RayPayload.h"
#include "HostDeviceCommon/HitInfo.h"
#include "HostDeviceCommon/RenderData.h"

struct MISBSDFRayReuse
{
	HIPRT_HOST_DEVICE void fill(ShadowLightRayHitInfo& shadow_light_ray,
		float3 inter_point, float3 bsdf_sampled_direction, ColorRGB32F& bsdf_color, float bsdf_pdf,
		RayState next_ray_state)
	{
#if ReuseBSDFMISRay
		this->prim_index = shadow_light_ray.hit_prim_index;
		this->material_index = shadow_light_ray.hit_material_index;
			
		this->interpolated_texcoords = shadow_light_ray.hit_interpolated_texcoords;

		this->hit_distance = shadow_light_ray.hit_distance;
		this->inter_point = inter_point;
		this->geometric_normal = shadow_light_ray.hit_geometric_normal;
		this->shading_normal = shadow_light_ray.hit_shading_normal;

		this->bsdf_sampled_direction = bsdf_sampled_direction;
		this->bsdf_color = bsdf_color;
		this->bsdf_pdf = bsdf_pdf;

		this->next_ray_state = next_ray_state;
#endif
	}

	HIPRT_HOST_DEVICE DeviceUnpackedEffectiveMaterial read_material(const HIPRTRenderData& render_data) const
	{
#if ReuseBSDFMISRay
		return get_intersection_material(render_data, this->material_index, this->interpolated_texcoords);
#endif
	}

	HIPRT_HOST_DEVICE void set_bsdf_pdf(float pdf)
	{
#if ReuseBSDFMISRay
		this->bsdf_pdf = pdf;
#endif
	}

	HIPRT_HOST_DEVICE void clear()
	{
#if ReuseBSDFMISRay
		// Setting the BSDF PDF to -1.0f indicates that there is no ray to reuse
		// because such a PDF is impossible to produce.
		//
		// That negative PDF will be catched in the main path tracing loop and the ray will be terminated
		bsdf_pdf = -1.0f;
#endif
	}

	HIPRT_HOST_DEVICE bool has_ray() const
	{
#if ReuseBSDFMISRay
		return bsdf_pdf > -1.0f && ReuseBSDFMISRay == KERNEL_OPTION_TRUE;
#else
		return false;
#endif
	}

#if ReuseBSDFMISRay
	int prim_index = -1;
	int material_index = -1;

	float2 interpolated_texcoords;

	float hit_distance = 0.0f;
	float3 inter_point = make_float3(1.0e35f, 1.0e35f, 1.0e35f);
	float3 geometric_normal = make_float3(-2.0f, -2.0f, -2.0f);
	float3 shading_normal = make_float3(-2.0f, -2.0f, -2.0f);

	float3 bsdf_sampled_direction = make_float3(-2.0f, -2.0f, -2.0f);
	ColorRGB32F bsdf_color;
	float bsdf_pdf = -1.0f;

	RayState next_ray_state;
#endif
};

/**
 * Updates the 'hit_info' and 'ray_payload' structures from a 'mis_reuse' structure
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool reuse_mis_ray(const HIPRTRenderData& render_data, HitInfo& closest_hit_info, RayPayload& ray_payload, float3 view_direction, MISBSDFRayReuse& mis_reuse)
{
#if ReuseBSDFMISRay
	bool intersection_found = false;

	if (mis_reuse.next_ray_state == RayState::MISSED)
		intersection_found = false;
	else
	{
		intersection_found = true;

		// We have a MISBSDFReuse ray, let's reuse all the information rather than tracing a ray
		closest_hit_info.geometric_normal = mis_reuse.geometric_normal;
		closest_hit_info.shading_normal = mis_reuse.shading_normal;
		closest_hit_info.inter_point = mis_reuse.inter_point;
		closest_hit_info.primitive_index = mis_reuse.prim_index;

		ray_payload.material = mis_reuse.read_material(render_data);
		fix_backfacing_normals(ray_payload, closest_hit_info, view_direction);

		if (ray_payload.is_inside_volume())
			ray_payload.volume_state.distance_in_volume += mis_reuse.hit_distance;
		ray_payload.volume_state.interior_stack.push(
			ray_payload.volume_state.incident_mat_index, ray_payload.volume_state.outgoing_mat_index, ray_payload.volume_state.inside_material,
			mis_reuse.material_index, ray_payload.material.get_dielectric_priority());

		// Clearing the MIS ray reuse
		mis_reuse.clear();
	}

	return intersection_found;
#else
	return false;
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F reuse_mis_bsdf_sample(float3& out_bsdf_sampled_direction, float& out_bsdf_pdf, RayPayload& ray_payload, MISBSDFRayReuse& mis_reuse)
{
#if ReuseBSDFMISRay
	// If we do have a MIS BSDF ray to reuse
	out_bsdf_pdf = mis_reuse.bsdf_pdf;
	out_bsdf_sampled_direction = mis_reuse.bsdf_sampled_direction;

	return mis_reuse.bsdf_color;
#else
	return ColorRGB32F();
#endif
}

#endif
