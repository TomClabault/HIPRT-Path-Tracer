/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_H

#include "Device/includes/LightSampling/LightSampleInformation.h"
#include "Device/includes/LightSampling/TriangleSamplingProjectedSolidAngle.h"
#include "Device/includes/LightSampling/TriangleSamplingSolidAngle.h"
#include "Device/includes/TriangleLoadUtils.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

/**
 * Reference: [A Low-Distortion Map Between Triangle and Square, Heitz, 2019]
 *
 * Maps a point in a square to a point in an arbitrary triangle
 */
HIPRT_DEVICE float2_t square_to_triangle(float& x, float& y)
{
	if (y > x)
	{
		x *= 0.5f;
		y -= x;
	}
	else
	{
		y *= 0.5f;
		x -= y;
	}

	return make_float2(x, y);
}

HIPRT_DEVICE float2_t sample_uv_on_triangle_uniform_area(float triangle_area, Xorshift32Generator& rng, float& out_uv_pdf)
{
	float rand_1 = rng();
	float rand_2 = rng();

#if TrianglePointSamplingUniformAreaStrategy == TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_TURK_1990
	float sqrt_r1 = sqrt(rand_1);
	float u		  = 1.0f - sqrt_r1;
	float v		  = (1.0f - rand_2) * sqrt_r1;
#elif TrianglePointSamplingUniformAreaStrategy == TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_HEITZ_2019
	float2_t remapped = square_to_triangle(rand_1, rand_2);

	float u = remapped.x;
	float v = remapped.y;
#endif // #if TrianglePointSamplingUniformAreaStrategy == TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_TURK_1990

	out_uv_pdf = 1.0f / triangle_area;

	return make_float2(u, v);
}

HIPRT_DEVICE float3_t sample_point_on_triangle_uniform_area(
	float3_t vertex_A, float3_t edge_AB, float3_t edge_AC, float triangle_area, Xorshift32Generator& rng, float& out_point_pdf, float2_t& out_point_uvs)
{
	out_point_uvs = sample_uv_on_triangle_uniform_area(triangle_area, rng, out_point_pdf);

	return vertex_A + edge_AB * out_point_uvs.x + edge_AC * out_point_uvs.y;
}

// HIPRT_DEVICE float2_t compute_uvs_of_point_on_triangle(float3_t vertex_A, float3_t edge_AB, float3_t edge_AC, float3_t point_on_triangle)
//{
//	float3_t AP = point_on_triangle - vertex_A;
//
//	// We can compute the uvs of the sampled point by projecting the AP vector on the AB and AC edges
//	// and dividing by the length of the edges squared (because we want to get the barycentrics coordinates)
//	float u = hippt::dot(AP, edge_AB) / hippt::dot(edge_AB, edge_AB);
//	float v = hippt::dot(AP, edge_AC) / hippt::dot(edge_AC, edge_AC);
//
//	return make_float2(u, v);
// }

HIPRT_DEVICE float2_t compute_uvs_of_point_on_triangle(float3_t vertex_A, float3_t edge_AB, float3_t edge_AC, float3_t point_on_triangle)
{
	float3_t AP = point_on_triangle - vertex_A;

	float d00 = hippt::dot(edge_AB, edge_AB);
	float d01 = hippt::dot(edge_AB, edge_AC);
	float d11 = hippt::dot(edge_AC, edge_AC);
	float d20 = hippt::dot(AP, edge_AB);
	float d21 = hippt::dot(AP, edge_AC);

	float denom = d00 * d11 - d01 * d01;

	// Triangle is degenerate if denom is zero or very close to zero
	if (fabsf(denom) < 1e-8f)
		return make_float2(0.0f, 0.0f);

	float u = (d11 * d20 - d01 * d21) / denom;
	float v = (d00 * d21 - d01 * d20) / denom;

	return make_float2(u, v);
}

/**
 * Samples a point uniformly on the given triangle (given with the triangle index)
 *
 * Returns true if the sampling was successful, false otherwise (can fail if the triangle is way too small or degenerate)
 */
HIPRT_DEVICE bool sample_point_on_generic_triangle(const HIPRTRenderData& render_data,
												   float3_t shading_point,
												   float3_t view_direction,
												   float3_t shading_normal,
												   const DeviceUnpackedEffectiveMaterial& material,
												   int global_triangle_index,
												   ColorRGB32F triangle_emission,
												   Xorshift32Generator& rng,
												   float3_t& out_sample_point,
												   float2_t& out_sample_point_uvs,
												   float3_t& out_sampled_triangle_normal,
												   float& out_triangle_area,
												   float& out_point_pdf)
{
	// TODO can we pack light triangles data to avoid loading vertices positions again? Have a buffer of packed, contiguous vertices just for light triangles to
	// get better memory coherence?
	//
	// EDIT: Nope, this is slower for some reason. More cache misses it seems but why? See commit 1e350ea6
	float3_t vertex_A = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[global_triangle_index * 3 + 0]];
	float3_t vertex_B = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[global_triangle_index * 3 + 1]];
	float3_t vertex_C = render_data.buffers.vertices_positions[render_data.buffers.triangles_indices[global_triangle_index * 3 + 2]];

	float3_t AB		= vertex_B - vertex_A;
	float3_t AC		= vertex_C - vertex_A;
	float3_t normal = hippt::cross(AB, AC);

	float length_normal = hippt::length(normal);
	if (length_normal <= TriangleSamplingNormalLengthRejectionThreshold)
		return false;

	normal /= length_normal;

	out_sampled_triangle_normal = normal;
	out_triangle_area			= 0.5f * length_normal;

#if TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA
	out_sample_point = sample_point_on_triangle_uniform_area(vertex_A, AB, AC, out_triangle_area, rng, out_point_pdf, out_sample_point_uvs);
#elif TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE

	out_sample_point = sample_point_on_triangle_solid_angle_peters_2021(render_data, vertex_A, vertex_B, vertex_C, normal, shading_point, view_direction,
																		shading_normal, triangle_emission, material, out_point_pdf, rng);

	out_sample_point_uvs = compute_uvs_of_point_on_triangle(vertex_A, AB, AC, out_sample_point);
#elif TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE // #if TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA
	float solid_angle = triangle_solid_angle(vertex_A, vertex_B, vertex_C, shading_point);

	bool do_projected_solid_angle_sampling = solid_angle > render_data.render_settings.projected_solid_angle_sampling_threshold;
	if (do_projected_solid_angle_sampling)
	{
		// If the triangle is large enough in solid angle, it may be worth it to compute the heavy projected solid angle
		// stuff
		out_sample_point = sample_point_on_triangle_projected_solid_angle_peters_2021(
			render_data, vertex_A, vertex_B, vertex_C, normal, shading_point, view_direction, shading_normal, triangle_emission, material, out_point_pdf, rng);

		out_sample_point_uvs = compute_uvs_of_point_on_triangle(vertex_A, AB, AC, out_sample_point);
	}
	else
		out_sample_point = sample_point_on_triangle_uniform_area(vertex_A, AB, AC, out_triangle_area, rng, out_point_pdf, out_sample_point_uvs);
#endif // #if TrianglePointSamplingStrategy == TRIANGLE_POINT_SAMPLING_STRATEGY_UNIFORM_AREA

	return out_point_pdf != 0.0f;
}

HIPRT_DEVICE ColorRGB32F sample_triangle_emission_at_point(const HIPRTRenderData& render_data, int global_triangle_index, float3_t point_on_triangle)
{
	TriangleIndices triangle_vertex_indices = load_triangle_vertex_indices(render_data.buffers.triangles_indices, global_triangle_index);

	float3_t vertex_A = render_data.buffers.vertices_positions[triangle_vertex_indices.x];
	float3_t vertex_B = render_data.buffers.vertices_positions[triangle_vertex_indices.y];
	float3_t vertex_C = render_data.buffers.vertices_positions[triangle_vertex_indices.z];

	float2_t uvs = compute_uvs_of_point_on_triangle(vertex_A, vertex_B - vertex_A, vertex_C - vertex_A, point_on_triangle);

	TriangleTexcoords triangle_texcoords = load_triangle_texcoords(render_data.buffers.texcoords, triangle_vertex_indices);
	float2_t texcoords					 = uv_interpolate(triangle_texcoords, uvs);

	unsigned short int emissive_texture_index =
		render_data.buffers.materials_buffer_soa.get_emission_texture_index(render_data.buffers.material_indices[global_triangle_index]);
	unsigned short int base_color_texture_index =
		render_data.buffers.materials_buffer_soa.get_base_color_texture_index(render_data.buffers.material_indices[global_triangle_index]);

	ColorRGBA32F rgba_emission = sample_texture_rgba(render_data.buffers.material_textures, texcoords, emissive_texture_index, false);
	ColorRGBA32F base_color	   = ColorRGBA32F(1.0f);
	if (base_color_texture_index != MaterialConstants::NO_TEXTURE)
		base_color = sample_texture_rgba(render_data.buffers.material_textures, texcoords, base_color_texture_index, false);

	if (!render_data.render_settings.do_alpha_testing)
	{
		// Doing this so that this match what we see visually
		rgba_emission.a = 1.0f;
		base_color.a	= 1.0f;
	}

	return ColorRGB32F(rgba_emission.r, rgba_emission.g, rgba_emission.b) * rgba_emission.a * base_color.a;
}

HIPRT_DEVICE ColorRGB32F get_triangle_emission_at_point(const HIPRTRenderData& render_data, int global_triangle_index, float3_t point_on_triangle)
{
	float emission_strength = render_data.buffers.materials_buffer_soa.get_emission_strength(render_data.buffers.material_indices[global_triangle_index]);

	bool emissive_texture_used =
		render_data.buffers.materials_buffer_soa.get_emissive_texture_used(render_data.buffers.material_indices[global_triangle_index]);
	if (!emissive_texture_used)
	{
		ColorRGB32F emisison = render_data.buffers.materials_buffer_soa.get_emission(render_data.buffers.material_indices[global_triangle_index]);

		return emisison * emission_strength;
	}
	else
	{
		ColorRGB32F emission = sample_triangle_emission_at_point(render_data, global_triangle_index, point_on_triangle);

		return emission * emission_strength;
	}
}

template <int trianglePointSamplingStrategy = TrianglePointSamplingStrategy>
HIPRT_DEVICE float pdf_of_point_on_triangle_area_measure(const HIPRTRenderData& render_data,
														 float3_t shading_point,
														 float3_t view_direction,
														 float3_t shading_normal,
														 const DeviceUnpackedEffectiveMaterial& material,
														 float3_t point_on_triangle,
														 float3_t triangle_normal,
														 int emissive_triangle_global_index,
														 float light_area);

/**
 * From a triangle index, samples uniformly a point on the triangle and fills a LightSamplePointInformation
 * structure with the information (normal, area, emission, ...) of the triangle
 *
 * The PDF field of the LightSamplePointInformation is only field with the probability of sampling the
 * point on the triangle. The rest of the PDF must be computed by the caller
 */
HIPRT_DEVICE LightSamplePointInformation sample_point_on_light_and_fill_light_sample_information(const HIPRTRenderData& render_data,
																								 float3_t shading_point,
																								 float3_t view_direction,
																								 float3_t shading_normal,
																								 const DeviceUnpackedEffectiveMaterial& material,
																								 int global_triangle_index,
																								 Xorshift32Generator& rng)
{
	if (global_triangle_index == -1)
		return LightSamplePointInformation();

	LightSamplePointInformation light_sample;

	float sampled_point_pdf;
	float sampled_triangle_area;
	float3_t sampled_triangle_normal;
	float3_t random_point_on_triangle;
	float2_t random_point_on_triangle_uvs;
	ColorRGB32F triangle_emission = ColorRGB32F(render_data.buffers.triangles_average_emissive_luminance[global_triangle_index]);
	if (!sample_point_on_generic_triangle(render_data, shading_point, view_direction, shading_normal, material, global_triangle_index, triangle_emission, rng,
										  random_point_on_triangle, random_point_on_triangle_uvs, sampled_triangle_normal, sampled_triangle_area,
										  sampled_point_pdf))
		return LightSamplePointInformation();

	light_sample.emissive_triangle_global_index = global_triangle_index;
	light_sample.light_source_normal			= sampled_triangle_normal;
	light_sample.light_area						= sampled_triangle_area;
	light_sample.emission						= get_triangle_emission_at_point(render_data, global_triangle_index, random_point_on_triangle);
	light_sample.point_on_light					= random_point_on_triangle;
	light_sample.area_measure_pdf				= sampled_point_pdf;

	return light_sample;
}

#endif // #ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_TRIANGLE_SAMPLING_H
