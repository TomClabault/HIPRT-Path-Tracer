/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INTERSECT_H
#define DEVICE_INTERSECT_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Material.h"
#include "Device/includes/ONB.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Texture.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Math.h"

/* References:
 * 
 * [1] [Foundations of Game Engine Development: Rendering - Tangent/Bitangent calculation] http://foundationsofgameenginedev.com/#fged2
 */
HIPRT_HOST_DEVICE HIPRT_INLINE float3 normal_mapping(const HIPRTRenderData& render_data, int normal_map_texture_index, int primitive_index, const float2& interpolated_texcoords, const float3& surface_normal)
{
    int vertex_A_index = render_data.buffers.triangles_indices[primitive_index * 3 + 0];
    int vertex_B_index = render_data.buffers.triangles_indices[primitive_index * 3 + 1];
    int vertex_C_index = render_data.buffers.triangles_indices[primitive_index * 3 + 2];

    // Calculating tangents and bitangents aligned with texture U and V coordinates
    float2 P0_texcoords = render_data.buffers.texcoords[vertex_A_index];
    float2 P1_texcoords = render_data.buffers.texcoords[vertex_B_index];
    float2 P2_texcoords = render_data.buffers.texcoords[vertex_C_index];

    float2 delta_P1P0_texcoords = P1_texcoords - P0_texcoords;
    float2 delta_P2P0_texcoords = P2_texcoords - P0_texcoords;

    float det_inverse = 1.0f / (delta_P1P0_texcoords.x * delta_P2P0_texcoords.y - delta_P1P0_texcoords.y * delta_P2P0_texcoords.x);

    float3 P0 = render_data.buffers.vertices_positions[vertex_A_index];
    float3 P1 = render_data.buffers.vertices_positions[vertex_B_index];
    float3 P2 = render_data.buffers.vertices_positions[vertex_C_index];

    float3 edge_P0P1 = P1 - P0;
    float3 edge_P0P2 = P2 - P0;

    float3 T = (edge_P0P1 * delta_P2P0_texcoords.y - edge_P0P2 * delta_P1P0_texcoords.y) * det_inverse;
    float3 B = (edge_P0P2 * delta_P1P0_texcoords.x - edge_P0P1 * delta_P2P0_texcoords.x) * det_inverse;

    ColorRGB32F normal = sample_texture_rgb_8bits(render_data.buffers.material_textures, normal_map_texture_index, render_data.buffers.textures_dims[normal_map_texture_index], /* is_srgb */ false, interpolated_texcoords);
    // Bringing the normal in [-x, x]. x doesn't really matter since we normalize the result anyway
    normal -= ColorRGB32F(0.5f);

    float3 normal_tangent_space = hippt::normalize(make_float3(normal.r, normal.g, normal.b));

    return local_to_world_frame(hippt::normalize(T), hippt::normalize(B), surface_normal, normal_tangent_space);
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 get_shading_normal(const HIPRTRenderData& render_data, const float3& geometric_normal, int primitive_index, const float2& uv, const float2& interpolated_texcoords)
{
    int mat_index = render_data.buffers.material_indices[primitive_index];
    RendererMaterial& material = render_data.buffers.materials_buffer[mat_index];

    // Do smooth shading first if we have vertex normals
    float3 surface_normal;
    int vertex_A_index = render_data.buffers.triangles_indices[primitive_index * 3 + 0];
    if (render_data.buffers.has_vertex_normals[vertex_A_index])
        // Smooth normal available for the triangle
        surface_normal = hippt::normalize(uv_interpolate(render_data.buffers.triangles_indices, primitive_index, render_data.buffers.vertex_normals, uv));
    else
        surface_normal = geometric_normal;

    // Do normal mapping if we have a normal map
    if (material.normal_map_texture_index != -1)
        surface_normal = normal_mapping(render_data, material.normal_map_texture_index, primitive_index, interpolated_texcoords, surface_normal);

    return surface_normal;
}

#ifndef __KERNELCC__
#include "Renderer/BVH.h"
HIPRT_HOST_DEVICE HIPRT_INLINE hiprtHit intersect_scene_cpu(const HIPRTRenderData& render_data, const hiprtRay& ray)
{
    hiprtHit hiprtHit;
    HitInfo closest_hit_info;
    closest_hit_info.t = -1.0f;

    if (render_data.cpu_only.bvh->intersect(ray, closest_hit_info))
    {
        hiprtHit.primID = closest_hit_info.primitive_index;
        hiprtHit.normal = closest_hit_info.geometric_normal;
        hiprtHit.t = closest_hit_info.t;
        hiprtHit.uv = closest_hit_info.uv;
    }

    return hiprtHit;
}
#endif

/**
 * Returns true if a hit was found, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool trace_ray(const HIPRTRenderData& render_data, hiprtRay ray, RayPayload& in_out_ray_payload, HitInfo& out_hit_info)
{
    hiprtHit hit;
    bool skipping_volume_boundary = false;
    bool skipping_intersection_alpha_test = false;
    do
    {
    #ifdef __KERNELCC__
        hiprtGeomTraversalClosest traversal(render_data.geom, ray);

        hit = traversal.getNextHit();
    #else
        hit = intersect_scene_cpu(render_data, ray);
    #endif

        if (!hit.hasHit())
            return false;

        out_hit_info.inter_point = ray.origin + hit.t * ray.direction;
        out_hit_info.primitive_index = hit.primID;
        out_hit_info.texcoords = uv_interpolate(render_data.buffers.triangles_indices, out_hit_info.primitive_index, render_data.buffers.texcoords, hit.uv);
        // TODO hit.normal is in object space, this simple approach will not work if using
        // multiple-levels BVH (TLAS/BLAS)
        out_hit_info.geometric_normal = hippt::normalize(hit.normal);
        out_hit_info.shading_normal = get_shading_normal(render_data, out_hit_info.geometric_normal, out_hit_info.primitive_index, hit.uv, out_hit_info.texcoords);

        out_hit_info.t = hit.t;
        out_hit_info.uv = hit.uv;

        if (in_out_ray_payload.is_inside_volume())
            in_out_ray_payload.volume_state.distance_in_volume += hit.t;

        float base_color_alpha = 1.0f;
        int material_index = render_data.buffers.material_indices[hit.primID];
        in_out_ray_payload.material = get_intersection_material(render_data, material_index, out_hit_info.texcoords, base_color_alpha);

        skipping_intersection_alpha_test = base_color_alpha < 1.0f && render_data.render_settings.do_alpha_testing;
        if (skipping_intersection_alpha_test)
        {
            // Skipping the intersection now and not doing the volume stack handling
            ray.origin = out_hit_info.inter_point + ray.direction * 3.0e-3f;

            continue;
        }

        if (!in_out_ray_payload.is_inside_volume() || in_out_ray_payload.material.specular_transmission == 0.0f)
        {
            // If we're not in a volume, there's no reason for the normals not to be facing us so we're flipping
            // if they were wrongly oriented
            // 
            // Same thing with objects that do not let the rays pass through (non transmissive):
            // because we can never be inside of these objects, we're always outside.
            // If we're always outside, there's no reason to have the normals of these objects inverted.
            out_hit_info.geometric_normal *= hippt::dot(out_hit_info.geometric_normal, -ray.direction) < 0 ? -1 : 1;
            out_hit_info.shading_normal *= hippt::dot(out_hit_info.shading_normal, -ray.direction) < 0 ? -1 : 1;
        }

        skipping_volume_boundary = in_out_ray_payload.volume_state.interior_stack.push(in_out_ray_payload.volume_state.incident_mat_index, in_out_ray_payload.volume_state.outgoing_mat_index, in_out_ray_payload.volume_state.leaving_mat, material_index, in_out_ray_payload.material.dielectric_priority);

        if (skipping_volume_boundary)
        {
            // If we're skipping, the boundary, the ray just keeps going on its way
            ray.origin = out_hit_info.inter_point + ray.direction * 3.0e-3f;

            // Don't forget to increment the distance traveled
            // TODO: Are we not double counting the distance here and a few lines above (where we set the .t, .uv, .geometric_normal, ...)
            in_out_ray_payload.volume_state.distance_in_volume += hit.t;
        }

    } while ((skipping_volume_boundary && hit.hasHit()) || skipping_intersection_alpha_test);

    return hit.hasHit();
}

/**
 * Returns true if in shadow, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool evaluate_shadow_ray(const HIPRTRenderData& render_data, hiprtRay ray, float t_max)
{
#ifdef __KERNELCC__
    ray.maxT = t_max - 1.0e-4f;

    hiprtHit shadow_ray_hit;
    float alpha;

    do
    {
        hiprtGeomTraversalClosest traversal(render_data.geom, ray);

        shadow_ray_hit = traversal.getNextHit();
        if (!shadow_ray_hit.hasHit())
            return false;

        if (render_data.render_settings.do_alpha_testing)
            alpha = get_hit_base_color_alpha(render_data, shadow_ray_hit);
        else
            alpha = 1.0f;

        float3 inter_point = ray.origin + ray.direction * shadow_ray_hit.t;
        ray.origin = inter_point + ray.direction * 3.0e-3f;
        ray.maxT -= shadow_ray_hit.t;
    } while (alpha < 1.0f);

    // If we're here, this means that we found a hit that is not
    // alpha-transparent with a distance < t_max so that's a hit and we're shadowed.
    return true;
#else
    float alpha = 1.0f;
    // The total distance of our ray. Incremented after each hit
    // (we may find multiple hits if we hit transparent texture
    // and keep intersecting the scene)
    float cumulative_t = 0.0f;

    hiprtHit hit;
    do
    {
        // We should use ray tracing fitler functions here instead of re-tracing new rays
        hit = intersect_scene_cpu(render_data, ray);
        if (!hit.hasHit())
            return false;

        if (render_data.render_settings.do_alpha_testing)
            alpha = get_hit_base_color_alpha(render_data, hit);
        else
            alpha = 1.0f;

        float3 inter_point = ray.origin + ray.direction * hit.t;
        ray.origin = inter_point + ray.direction * 3.0e-3f;
        cumulative_t += hit.t;

        // We keep going as long as the alpha is < 1.0f meaning that we hit texture transparency
    } while (alpha < 1.0f && cumulative_t < t_max - 1.0e-4f);

    // If we found a hit and that it is close enough
    return hit.hasHit() && cumulative_t < t_max - 1.0e-4f;
#endif // __KERNELCC__
}

/**
 * Returns true if in shadow, false otherwise.
 * 
 * Also, if a hit was found, outputs the emission of the material at the hit point in 'out_hit_emission'
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool evaluate_shadow_light_ray(const HIPRTRenderData& render_data, hiprtRay ray, float t_max, ShadowLightRayHitInfo& out_light_hit_info)
{
#ifdef __KERNELCC__
    ray.maxT = t_max - 1.0e-4f;

    hiprtHit shadow_ray_hit;
    float alpha;
    out_light_hit_info.hit_distance = 0.0f;

    do
    {
        hiprtGeomTraversalClosest traversal(render_data.geom, ray);

        shadow_ray_hit = traversal.getNextHit();
        if (!shadow_ray_hit.hasHit())
            return false;

        if (render_data.render_settings.do_alpha_testing)
            alpha = get_hit_base_color_alpha(render_data, shadow_ray_hit);
        else
            alpha = 1.0f;

        float3 inter_point = ray.origin + ray.direction * shadow_ray_hit.t;
        ray.origin = inter_point + ray.direction * 3.0e-3f;
        ray.maxT -= shadow_ray_hit.t;
        out_light_hit_info.hit_distance += shadow_ray_hit.t;
    } while (alpha < 1.0f);


    // If we're here, this means that we found a hit that is not
    // alpha-transparent with a distance < t_max so that's a hit and we're shadowed.

    // Reading the emission of the material
    int material_index = render_data.buffers.material_indices[shadow_ray_hit.primID];
    int emission_texture_index = render_data.buffers.materials_buffer[material_index].emission_texture_index;

    if (emission_texture_index != -1)
    {
        float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, shadow_ray_hit.primID, render_data.buffers.texcoords, shadow_ray_hit.uv);
        get_material_property(render_data, out_light_hit_info.hit_emission, false, texcoords, emission_texture_index);

        // Using the already computed texcoords to get the shading normal
        out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), shadow_ray_hit.primID, shadow_ray_hit.uv, texcoords);
    }
    else
    {
        out_light_hit_info.hit_emission = render_data.buffers.materials_buffer[material_index].emission;

        float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, shadow_ray_hit.primID, render_data.buffers.texcoords, shadow_ray_hit.uv);
        out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), shadow_ray_hit.primID, shadow_ray_hit.uv, texcoords);
    }

    out_light_hit_info.hit_prim_index = shadow_ray_hit.primID;

    return true;
#else
    float alpha = 1.0f;
    // The total distance of our ray. Incremented after each hit
    // (we may find multiple hits if we hit transparent texture
    // and keep intersecting the scene)
    float cumulative_t = 0.0f;

    hiprtHit shadow_ray_hit;
    do
    {
        // We should use ray tracing fitler functions here instead of re-tracing new rays
        shadow_ray_hit = intersect_scene_cpu(render_data, ray);
        if (!shadow_ray_hit.hasHit())
            return false;

        if (render_data.render_settings.do_alpha_testing)
            alpha = get_hit_base_color_alpha(render_data, shadow_ray_hit);
        else
            alpha = 1.0f;

        float3 inter_point = ray.origin + ray.direction * shadow_ray_hit.t;
        ray.origin = inter_point + ray.direction * 3.0e-3f;
        cumulative_t += shadow_ray_hit.t;

        // We keep going as long as the alpha is < 1.0f meaning that we hit texture transparency
    } while (alpha < 1.0f && cumulative_t < t_max - 1.0e-4f);

    bool hit_found = shadow_ray_hit.hasHit() && cumulative_t < t_max - 1.0e-4f;

    if (hit_found)
    {
        // If we found a hit and that it is close enough (hit_found conditions)

        int material_index = render_data.buffers.material_indices[shadow_ray_hit.primID];
        int emission_texture_index = render_data.buffers.materials_buffer[material_index].emission_texture_index;

        if (emission_texture_index != -1)
        {
            float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, shadow_ray_hit.primID, render_data.buffers.texcoords, shadow_ray_hit.uv);
            get_material_property(render_data, out_light_hit_info.hit_emission, false, texcoords, emission_texture_index);

            // Using the already computed texcoords to get the shading normal
            out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), shadow_ray_hit.primID, shadow_ray_hit.uv, texcoords);
        }
        else
        {
            out_light_hit_info.hit_emission = render_data.buffers.materials_buffer[material_index].emission;

            float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, shadow_ray_hit.primID, render_data.buffers.texcoords, shadow_ray_hit.uv);
            out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), shadow_ray_hit.primID, shadow_ray_hit.uv, texcoords);
        }

        out_light_hit_info.hit_prim_index = shadow_ray_hit.primID;
        out_light_hit_info.hit_distance = cumulative_t;

        return true;
    }
    else
        return false;
#endif // __KERNELCC__
}

#endif
