/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LIGHT_SAMPLING_PDF_TRIANGLES_H
#define DEVICE_LIGHT_SAMPLING_PDF_TRIANGLES_H

#include "Device/includes/BSDFs/BSDFSampleHitInfo.h"
#include "Device/includes/LightSampling/LightTree/LightTreeATSSampling.h"
#include "Device/includes/LightSampling/LightTree/LightTreeSGSampling.h"
#include "Device/includes/TriangleLoadUtils.h"

#include "HostDeviceCommon/RenderData.h"

// template <int trianglePointSamplingStrategy = TrianglePointSamplingStrategy>
template <int trianglePointSamplingStrategy>
HIPRT_DEVICE float pdf_of_point_on_triangle_area_measure(const HIPRTRenderData& render_data,
														 float3_t shading_point,
														 float3_t view_direction,
														 float3_t shading_normal,
														 const DeviceUnpackedEffectiveMaterial& material,
														 float3_t point_on_triangle,
														 float3_t triangle_normal,
														 int emissive_triangle_global_index,
														 float light_area)
{
	if constexpr (trianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA)
	{
		return 1.0f / light_area;
	}
	else if constexpr (trianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE)
	{
		float3_t vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 0]];
		float3_t vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 1]];
		float3_t vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 2]];
		ColorRGB32F triangle_emission =
			render_data.buffers.materials_buffer_soa.get_emission(render_data.buffers.material_indices[emissive_triangle_global_index]);

		float3_t to_light_direction = point_on_triangle - shading_point;
		float to_light_distance		= hippt::length(to_light_direction);

		float pdf_solid_angle = solid_angle_triangle_solid_angle_pdf_from_sampled_point(
			render_data, vertex_A, vertex_B, vertex_C, shading_point, view_direction, shading_normal, point_on_triangle,
			ltc_lobe_probas(render_data, vertex_A, vertex_B, vertex_C, shading_point, view_direction, shading_normal, triangle_emission, material), material);

		return solid_angle_to_area_pdf(pdf_solid_angle, to_light_distance,
									   compute_cosine_term_at_light_source(triangle_normal, -to_light_direction / to_light_distance));
	}
	else if constexpr (trianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE)
	{
		float3_t vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 0]];
		float3_t vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 1]];
		float3_t vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[emissive_triangle_global_index * 3 + 2]];
		ColorRGB32F triangle_emission =
			render_data.buffers.materials_buffer_soa.get_emission(render_data.buffers.material_indices[emissive_triangle_global_index]);

		float3_t to_light_direction = point_on_triangle - shading_point;
		float to_light_distance		= hippt::length(to_light_direction);

		float solid_angle = triangle_solid_angle(vertex_A, vertex_B, vertex_C, shading_point);

		bool do_projected_solid_angle_sampling = solid_angle > render_data.render_settings.projected_solid_angle_sampling_threshold;
		if (do_projected_solid_angle_sampling)
		{
			// If the triangle is large enough in solid angle, it may be worth it to compute the heavy projected solid angle
			// stuff
			float pdf_solid_angle = projected_solid_angle_triangle_solid_angle_pdf(
				render_data, vertex_A, vertex_B, vertex_C, shading_point, view_direction, shading_normal, point_on_triangle,
				ltc_lobe_probas(render_data, vertex_A, vertex_B, vertex_C, shading_point, view_direction, shading_normal, triangle_emission, material),
				material);

			return solid_angle_to_area_pdf(pdf_solid_angle, to_light_distance,
										   compute_cosine_term_at_light_source(triangle_normal, -hippt::normalize(point_on_triangle - shading_point)));
		}
		else
		{
			// Otherwise it's not worth it and we can use the cheap solid angle (not projected) sampling
			float pdf_solid_angle = solid_angle_triangle_solid_angle_pdf_from_sampled_point(
				render_data, vertex_A, vertex_B, vertex_C, shading_point, view_direction, shading_normal, point_on_triangle,
				ltc_lobe_probas(render_data, vertex_A, vertex_B, vertex_C, shading_point, view_direction, shading_normal, triangle_emission, material),
				material);

			return solid_angle_to_area_pdf(pdf_solid_angle, to_light_distance,
										   compute_cosine_term_at_light_source(triangle_normal, -to_light_direction / to_light_distance));
		}

		return 0.0f;
	}
}

template <int lightSamplingStrategy = DirectLightSamplingStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle(const HIPRTRenderData& render_data,
											float3_t shading_point,
											float3_t view_direction,
											float3_t shading_normal,
											const DeviceUnpackedEffectiveMaterial& material,
											int emissive_triangle_global_index,
											float light_area)
{
	if (render_data.buffers.emissive_triangles_count == 0)
		return 0.0f;

	if constexpr (lightSamplingStrategy == LSS_BASE_UNIFORM)
	{
		return 1.0f / render_data.buffers.emissive_triangles_count;
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_POWER)
	{
		float sampling_power = render_data.buffers.triangles_average_emissive_power_luminance[emissive_triangle_global_index];
		// TODO EMISSIVE TEXTURE SAMPLING
		// Here we need to use the emission used in the CDF to get the PDF, not the hit emission (which is the light_emission parameter here)
		return sampling_power / render_data.buffers.emissive_triangles_power_alias_table.sum_elements;
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
	{
		return pdf_of_emissive_triangle_light_tree_ats(render_data, shading_point, shading_normal, emissive_triangle_global_index);
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG)
	{
		return pdf_of_emissive_triangle_light_tree_sg(render_data, shading_point, view_direction, shading_normal, material, emissive_triangle_global_index);
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_REGIR)
		// We should never ask that question, we can't get the PDF of ReGIR
		return 1.0e15f;
}

/**
 * Returns the PDF (area measure) of the light sampler for the given triangle_hit_info
 *
 * 'primitive_index' is the index of the emissive triangle hit
 * 'shading_normal' is the shading normal at the intersection point of the emissive triangle hit

 * 'ray_direction' is the direction of the ray that hit the triangle. The direction points towards the triangle.
 */
template <int lightSamplingStrategy = DirectLightSamplingStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data,
															 float3_t shading_point,
															 float3_t view_direction,
															 float3_t shading_normal,
															 const DeviceUnpackedEffectiveMaterial& material,
															 float3_t point_on_triangle,
															 float3_t triangle_normal,
															 int emissive_triangle_global_index,
															 float light_area)
{
	float point_on_light_pdf;

	// Note that for ReGIR, we cannot have the exact light PDF since ReGIR is based on RIS so we're
	// faking it with whatever base strategy ReGIR is using

	if constexpr (lightSamplingStrategy == LSS_BASE_UNIFORM)
	{
		// Surface area PDF of hitting that point on that triangle in the scene
		point_on_light_pdf = pdf_of_point_on_triangle_area_measure(render_data, shading_point, view_direction, shading_normal, material, point_on_triangle,
																   triangle_normal, emissive_triangle_global_index, light_area);
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_POWER)
	{
		point_on_light_pdf = pdf_of_point_on_triangle_area_measure(render_data, shading_point, view_direction, shading_normal, material, point_on_triangle,
																   triangle_normal, emissive_triangle_global_index, light_area);
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
	{
		point_on_light_pdf = pdf_of_point_on_triangle_area_measure(render_data, shading_point, view_direction, shading_normal, material, point_on_triangle,
																   triangle_normal, emissive_triangle_global_index, light_area);
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG)
	{
		point_on_light_pdf = pdf_of_point_on_triangle_area_measure(render_data, shading_point, view_direction, shading_normal, material, point_on_triangle,
																   triangle_normal, emissive_triangle_global_index, light_area);
	}
	else if constexpr (lightSamplingStrategy == LSS_BASE_REGIR)
		// We should never ask that question, we can't get the PDF of ReGIR
		point_on_light_pdf = 1.0e15f;
	else
		// Invalid strategy
		point_on_light_pdf = 1.0e15f;

	float light_pdf = pdf_of_emissive_triangle<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal, material,
																	  emissive_triangle_global_index, light_area);

	float full_pdf = point_on_light_pdf * light_pdf;

	return full_pdf;
}

template <int lightSamplingStrategy = DirectLightSamplingStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data,
															 float3_t shading_point,
															 float3_t view_direction,
															 float3_t shading_normal,
															 const DeviceUnpackedEffectiveMaterial& material,
															 float3_t point_on_triangle,
															 float3_t triangle_normal,
															 int emissive_triangle_global_index)
{
	return pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal, material,
																			point_on_triangle, triangle_normal, emissive_triangle_global_index,
																			triangle_load_area(render_data, emissive_triangle_global_index));
}

template <int lightSamplingStrategy = DirectLightSamplingStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_area_measure(const HIPRTRenderData& render_data,
															 float3_t shading_point,
															 float3_t view_direction,
															 float3_t shading_normal,
															 const DeviceUnpackedEffectiveMaterial& material,
															 float3_t point_on_triangle,
															 const BSDFLightSampleRayHitInfo& light_hit_info)
{
	return pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal, material,
																			point_on_triangle, light_hit_info.hit_geometric_normal,
																			light_hit_info.hit_prim_index);
}

/**
 * Returns the PDF (solid angle measure) of the light sampler for the given 'light_hit_info'
 *
 * Note that for light samplers that cannot be point-evaluated (ReGIR for example: we cannot compute a RIS PDF),
 * the returned PDF is an approximation
 *
 * 'primitive_index' is the index of the emissive triangle hit
 * 'shading_normal' is the shading normal at the intersection point of the emissive triangle hit
 * 'hit_distance' is the distance to the intersection point on the hit triangle
 * 'to_light_direction' is the direction of the ray that hit the triangle. The direction points towards the triangle.
 */
template <int lightSamplingStrategy = DirectLightSamplingStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data,
															float3_t shading_point,
															float3_t view_direction,
															float3_t shading_normal,
															const DeviceUnpackedEffectiveMaterial& material,
															int emissive_triangle_global_index,
															float light_area,
															float3_t light_surface_normal,
															float hit_distance,
															float3_t to_light_direction)
{
	// abs() here to allow backfacing lights
	// Without abs() here:
	//  - We could be hitting the back of an emissive triangle (think of quad light hanging in the air)
	//  --> triangle normal not facing the same way
	//  --> cos_angle negative
	float cosine_light_source = compute_cosine_term_at_light_source(light_surface_normal, -to_light_direction);
	if (cosine_light_source < 1.0e-8f)
		return 0.0f;

	float pdf_area_measure = pdf_of_emissive_triangle_hit_area_measure<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal,
																							  material, shading_point + hit_distance * to_light_direction,
																							  light_surface_normal, emissive_triangle_global_index, light_area);

	return area_to_solid_angle_pdf(pdf_area_measure, hit_distance, cosine_light_source);
}

template <int lightSamplingStrategy = DirectLightSamplingStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data,
															float3_t shading_point,
															float3_t view_direction,
															float3_t shading_normal,
															const DeviceUnpackedEffectiveMaterial& material,
															int emissive_triangle_global_index,
															float3_t light_surface_normal,
															float hit_distance,
															float3_t to_light_direction)
{
	return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(
		render_data, shading_point, view_direction, shading_normal, material, emissive_triangle_global_index,
		triangle_load_area(render_data, emissive_triangle_global_index), light_surface_normal, hit_distance, to_light_direction);
}

template <int lightSamplingStrategy = DirectLightSamplingStrategy>
HIPRT_DEVICE float pdf_of_emissive_triangle_hit_solid_angle(const HIPRTRenderData& render_data,
															float3_t shading_point,
															float3_t view_direction,
															float3_t shading_normal,
															const DeviceUnpackedEffectiveMaterial& material,
															const BSDFLightSampleRayHitInfo& light_hit_info,
															float3_t to_light_direction)
{
	return pdf_of_emissive_triangle_hit_solid_angle<lightSamplingStrategy>(render_data, shading_point, view_direction, shading_normal, material,
																		   light_hit_info.hit_prim_index, light_hit_info.hit_geometric_normal,
																		   light_hit_info.hit_distance, to_light_direction);
}

#endif
