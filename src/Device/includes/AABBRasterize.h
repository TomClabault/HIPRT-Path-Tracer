/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_AABB_RASTERIZE_H
#define DEVICE_INCLUDES_AABB_RASTERIZE_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/HIPRTCamera.h"

HIPRT_DEVICE bool aabb_rasterize_ray_is_near_segment(
	const hiprtRay& ray, float3_t segment_start, float3_t segment_end, float pixel_angular_radius, float& out_ray_parameter)
{
	const float3_t segment_direction   = segment_end - segment_start;
	const float segment_length_squared = hippt::dot(segment_direction, segment_direction);
	if (segment_length_squared <= 1.0e-12f)
		return false;

	const float3_t origin_to_segment_start = ray.origin - segment_start;
	const float ray_segment_direction_dot  = hippt::dot(ray.direction, segment_direction);
	const float ray_origin_offset		   = hippt::dot(ray.direction, origin_to_segment_start);
	const float segment_origin_offset	   = hippt::dot(segment_direction, origin_to_segment_start);
	const float denominator				   = segment_length_squared - ray_segment_direction_dot * ray_segment_direction_dot;

	float ray_parameter		= 0.0f;
	float segment_parameter = 0.0f;
	if (denominator > 1.0e-12f)
	{
		ray_parameter	  = (ray_segment_direction_dot * segment_origin_offset - segment_length_squared * ray_origin_offset) / denominator;
		segment_parameter = (segment_origin_offset - ray_segment_direction_dot * ray_origin_offset) / denominator;
	}
	else
		segment_parameter = segment_origin_offset / segment_length_squared;

	if (segment_parameter < 0.0f)
	{
		segment_parameter = 0.0f;
		ray_parameter	  = -ray_origin_offset;
	}
	else if (segment_parameter > 1.0f)
	{
		segment_parameter = 1.0f;
		ray_parameter	  = ray_segment_direction_dot - ray_origin_offset;
	}

	if (ray_parameter < 0.0f)
	{
		ray_parameter	  = 0.0f;
		segment_parameter = hippt::clamp(0.0f, 1.0f, segment_origin_offset / segment_length_squared);
	}

	const float3_t closest_ray_point		= ray.origin + ray_parameter * ray.direction;
	const float3_t closest_segment_point	= segment_start + segment_parameter * segment_direction;
	const float3_t closest_point_difference = closest_ray_point - closest_segment_point;
	const float world_line_radius			= ray_parameter * pixel_angular_radius * 1.5f;

	out_ray_parameter = ray_parameter;
	return ray_parameter > 0.0f && hippt::dot(closest_point_difference, closest_point_difference) <= world_line_radius * world_line_radius;
}

HIPRT_DEVICE bool aabb_rasterize_pixel_is_on_aabb_edge(
	const HIPRTCamera& camera, int2_t render_resolution, int pixel_index, float3_t bounds_min, float3_t bounds_max)
{
	const unsigned int image_width	= render_resolution.x;
	const unsigned int image_height = render_resolution.y;
	const unsigned int pixel_x		= pixel_index % image_width;
	const unsigned int pixel_y		= pixel_index / image_width;

	const float pixel_center_x = pixel_x + 0.5f;
	const float pixel_center_y = pixel_y + 0.5f;
	const hiprtRay center_ray  = camera.get_camera_ray(pixel_center_x, pixel_center_y, render_resolution);

	const float neighbor_x						= pixel_x + 1 < image_width ? pixel_center_x + 1.0f : pixel_center_x - 1.0f;
	const float neighbor_y						= pixel_y + 1 < image_height ? pixel_center_y + 1.0f : pixel_center_y - 1.0f;
	const hiprtRay horizontal_neighbor_ray		= camera.get_camera_ray(neighbor_x, pixel_center_y, render_resolution);
	const hiprtRay vertical_neighbor_ray		= camera.get_camera_ray(pixel_center_x, neighbor_y, render_resolution);
	const float horizontal_pixel_angular_radius = hippt::length(horizontal_neighbor_ray.direction - center_ray.direction);
	const float vertical_pixel_angular_radius	= hippt::length(vertical_neighbor_ray.direction - center_ray.direction);
	const float pixel_angular_radius			= hippt::max(horizontal_pixel_angular_radius, vertical_pixel_angular_radius);

	static constexpr int edge_corner_indices[12][2] = { { 0, 1 }, { 0, 2 }, { 0, 4 }, { 1, 3 }, { 1, 5 }, { 2, 3 },
														{ 2, 6 }, { 3, 7 }, { 4, 5 }, { 4, 6 }, { 5, 7 }, { 6, 7 } };

	float3_t corners[8];
	for (int corner_index = 0; corner_index < 8; corner_index++)
		corners[corner_index] = make_float3((corner_index & 1) ? bounds_min.x : bounds_max.x, (corner_index & 2) ? bounds_min.y : bounds_max.y,
											(corner_index & 4) ? bounds_min.z : bounds_max.z);

	for (int edge_index = 0; edge_index < 12; edge_index++)
	{
		float ray_parameter;
		if (aabb_rasterize_ray_is_near_segment(center_ray, corners[edge_corner_indices[edge_index][0]], corners[edge_corner_indices[edge_index][1]],
											   pixel_angular_radius, ray_parameter))
			return true;
	}

	return false;
}

#endif
