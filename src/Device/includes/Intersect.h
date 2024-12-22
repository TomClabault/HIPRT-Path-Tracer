/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INTERSECT_H
#define DEVICE_INTERSECT_H

#include "Device/includes/Dispersion.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Material.h"
#include "Device/includes/ONB.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Texture.h"
#include "Device/functions/FilterFunction.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Math.h"

#if SharedStackBVHTraversalSize > 0
// This if is necessary to avoid declaring 0 size arrays if the
// shared stack traversal sizes are 0
__shared__ static int shared_stack_cache[SharedStackBVHTraversalSize * SharedStackBVHTraversalBlockSize];
#endif

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

    float3 P0 = render_data.buffers.vertices_positions[vertex_A_index];
    float3 P1 = render_data.buffers.vertices_positions[vertex_B_index];
    float3 P2 = render_data.buffers.vertices_positions[vertex_C_index];

    float3 edge_P0P1 = P1 - P0;
    float3 edge_P0P2 = P2 - P0;

    float det = delta_P1P0_texcoords.x * delta_P2P0_texcoords.y - delta_P1P0_texcoords.y * delta_P2P0_texcoords.x;
    // Check if the det isn't too low to avoid degenerate geometries that can then cause NaNs
    if (det > 1.0e-6f)
    {
        float det_inverse = 1.0f / det;
        float3 T = (edge_P0P1 * delta_P2P0_texcoords.y - edge_P0P2 * delta_P1P0_texcoords.y) * det_inverse;
        float3 B = (edge_P0P2 * delta_P1P0_texcoords.x - edge_P0P1 * delta_P2P0_texcoords.x) * det_inverse;

        ColorRGB32F normal = sample_texture_rgb_8bits(render_data.buffers.material_textures, normal_map_texture_index, render_data.buffers.textures_dims[normal_map_texture_index], /* is_srgb */ false, interpolated_texcoords);
        // Bringing the normal in [-x, x]. x doesn't really matter since we normalize the result anyway
        normal -= ColorRGB32F(0.5f);

        float3 normal_tangent_space = hippt::normalize(make_float3(normal.r, normal.g, normal.b));

        return local_to_world_frame(hippt::normalize(T), hippt::normalize(B), surface_normal, normal_tangent_space);
    }
    else
        // No surface derivatives basically, can't compute the normal mapping
        return surface_normal;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 get_shading_normal(const HIPRTRenderData& render_data, const float3& geometric_normal, int primitive_index, const float2& uv, const float2& interpolated_texcoords)
{
    int mat_index = render_data.buffers.material_indices[primitive_index];
    DevicePackedTexturedMaterial& material = render_data.buffers.materials_buffer[mat_index];

    // Do smooth shading first if we have vertex normals
    float3 surface_normal;
    int vertex_A_index = render_data.buffers.triangles_indices[primitive_index * 3 + 0];
    if (render_data.buffers.has_vertex_normals[vertex_A_index])
        // Smooth normal available for the triangle
        surface_normal = hippt::normalize(uv_interpolate(render_data.buffers.triangles_indices, primitive_index, render_data.buffers.vertex_normals, uv));
    else
        surface_normal = geometric_normal;

    // Do normal mapping if we have a normal map
    if (material.get_normal_map_texture_index() != MaterialUtils::NO_TEXTURE)
        surface_normal = normal_mapping(render_data, material.get_normal_map_texture_index(), primitive_index, interpolated_texcoords, surface_normal);

    return surface_normal;
}

#ifndef __KERNELCC__
#include "Renderer/BVH.h"
HIPRT_HOST_DEVICE HIPRT_INLINE hiprtHit intersect_scene_cpu(const HIPRTRenderData& render_data, const hiprtRay& ray, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
    hiprtHit hiprtHit;
    HitInfo closest_hit_info;
    closest_hit_info.t = -1.0f;

    FilterFunctionPayload filter_function_payload;
    filter_function_payload.render_data = &render_data;
    filter_function_payload.random_number_generator = &random_number_generator;
    // Filling the payload with the last hit primitive index to avoid self intersections
    // (avoid that the ray intersects the triangle it is currently sitting on)
    filter_function_payload.last_hit_primitive_index = last_hit_primitive_index;
    if (render_data.cpu_only.bvh->intersect(ray, closest_hit_info, &filter_function_payload))
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
HIPRT_HOST_DEVICE HIPRT_INLINE bool trace_ray(const HIPRTRenderData& render_data, hiprtRay ray, RayPayload& in_out_ray_payload, HitInfo& out_hit_info, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
    hiprtHit hit;
    bool skipping_volume_boundary = false;
    do
    {
#ifdef __KERNELCC__
        // Payload for the alpha testing filter function
        FilterFunctionPayload payload;
        payload.render_data = &render_data;
        payload.random_number_generator = &random_number_generator;
        // Filling the payload with the last hit primitive index to avoid self intersections
        // (avoid that the ray intersects the triangle it is currently sitting on)
        payload.last_hit_primitive_index = last_hit_primitive_index;

#if UseSharedStackBVHTraversal == KERNEL_OPTION_TRUE
#if SharedStackBVHTraversalSize > 0
        hiprtSharedStackBuffer shared_stack_buffer { SharedStackBVHTraversalSize, shared_stack_cache };
#else
        hiprtSharedStackBuffer shared_stack_buffer{ 0, nullptr };
#endif
        hiprtGlobalStack global_stack(render_data.global_traversal_stack_buffer, shared_stack_buffer);

        hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> traversal(render_data.geom, ray, global_stack, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#else
        hiprtGeomTraversalClosest traversal(render_data.geom, ray, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#endif

        hit = traversal.getNextHit();
    #else
        hit = intersect_scene_cpu(render_data, ray, last_hit_primitive_index, random_number_generator);
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

        int material_index = render_data.buffers.material_indices[hit.primID];
        in_out_ray_payload.material = get_intersection_material(render_data, material_index, out_hit_info.texcoords);

        if ((!in_out_ray_payload.is_inside_volume() || in_out_ray_payload.material.get_specular_transmission() == 0.0f) && !in_out_ray_payload.material.get_thin_walled())
        {
            // If we're not in a volume, there's no reason for the normals not to be facing us so we're flipping
            // if they were wrongly oriented
            // 
            // Same thing with objects that do not let the rays pass through (non transmissive):
            // because we can never be inside of these objects, we're always outside.
            // If we're always outside, there's no reason to have the normals of these objects inverted.
            //
            // Thin objects though can let the rays go inside them. But because they are thin, we won't know that
            // we're inside "a volume". So the above condition will trigger while we're hitting a thin material
            // from the inside which will flip the normal the wrong direction. So we're adding an exception for
            // thin materials here

            out_hit_info.geometric_normal *= hippt::dot(out_hit_info.geometric_normal, -ray.direction) < 0.0f ? -1.0f : 1.0f;

            // If that's not enough and the shading normal is below the geometric surface, this will lead to light leaking
            // because BSDF samples may now be able to sample below the surface so we're flipping the normal to be in the same
            // hemisphere as the geometric normal
            out_hit_info.shading_normal *= hippt::dot(out_hit_info.shading_normal, out_hit_info.geometric_normal) < 0.0f ? -1.0f : 1.0f;

            // Reflecting the shading normal about the view direction so that the shading 
            // normal is in the same hemisphere as the view direction
            float NdotV = hippt::dot(out_hit_info.shading_normal, -ray.direction);
            out_hit_info.shading_normal += (2.0f * hippt::clamp(0.0f, 1.0f, -NdotV)) * -ray.direction;
        }

        skipping_volume_boundary = in_out_ray_payload.volume_state.interior_stack.push(
            in_out_ray_payload.volume_state.incident_mat_index, in_out_ray_payload.volume_state.outgoing_mat_index, in_out_ray_payload.volume_state.inside_material, material_index, in_out_ray_payload.material.get_dielectric_priority());

        if (skipping_volume_boundary)
        {
            // If we're skipping, the boundary, the ray just keeps going on its way
            ray.origin = out_hit_info.inter_point;

            // Don't forget to increment the distance traveled
            // TODO: Are we not double counting the distance here and a few lines above (where we set the .t, .uv, .geometric_normal, ...)
            in_out_ray_payload.volume_state.distance_in_volume += hit.t;
        }

    } while ((skipping_volume_boundary && hit.hasHit()));

    if (in_out_ray_payload.material.get_dispersion_scale() > 0.0f && in_out_ray_payload.material.get_specular_transmission() > 0.0f && in_out_ray_payload.volume_state.sampled_wavelength == 0.0f)
        // If we hit a dispersive material, we sample the wavelength that will be used
        // for computing the wavelength dependent IORs used for dispersion
        //
        // We're also not re-doing the sampling if a wavelength has already been sampled for that path
        //
        // Negating the wavelength to indicate that the throughput filter of the wavelength
        // hasn't been applied yet (applied in principled_glass_eval())
        in_out_ray_payload.volume_state.sampled_wavelength = -sample_wavelength_uniformly(random_number_generator);

    return hit.hasHit();
}

/**
 * Returns true if in shadow, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool evaluate_shadow_ray(const HIPRTRenderData& render_data, hiprtRay ray, float t_max, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
#ifdef __KERNELCC__
    ray.maxT = t_max - 1.0e-4f;

    // Payload for the alpha testing filter function
    FilterFunctionPayload payload;
    payload.render_data = &render_data;
    payload.random_number_generator = &random_number_generator;
    // Filling the payload with the last hit primitive index to avoid self intersections
    // (avoid that the ray intersects the triangle it is currently sitting on)
    payload.last_hit_primitive_index = last_hit_primitive_index;

#if UseSharedStackBVHTraversal == KERNEL_OPTION_TRUE
#if SharedStackBVHTraversalSize > 0
    hiprtSharedStackBuffer shared_stack_buffer{ SharedStackBVHTraversalSize, shared_stack_cache };
#else
    hiprtSharedStackBuffer shared_stack_buffer{ 0, nullptr };
#endif
    hiprtGlobalStack global_stack(render_data.global_traversal_stack_buffer, shared_stack_buffer);

    hiprtGeomTraversalAnyHitCustomStack<hiprtGlobalStack> traversal(render_data.geom, ray, global_stack, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
    //hiprtGeomTraversalAnyHitCustomStack<hiprtGlobalStack> traversal()
#else
    hiprtGeomTraversalClosest traversal(render_data.geom, ray, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#endif

    hiprtHit shadow_ray_hit = traversal.getNextHit();
    if (!shadow_ray_hit.hasHit())
        return false;

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
        hit = intersect_scene_cpu(render_data, ray, last_hit_primitive_index, random_number_generator);
        if (!hit.hasHit())
            return false;

        if (render_data.render_settings.do_alpha_testing)
            alpha = get_hit_base_color_alpha(render_data, hit);
        else
            alpha = 1.0f;

        // Next ray origin
        ray.origin = ray.origin + ray.direction * hit.t;
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
HIPRT_HOST_DEVICE HIPRT_INLINE bool evaluate_shadow_light_ray(const HIPRTRenderData& render_data, hiprtRay ray, float t_max, ShadowLightRayHitInfo& out_light_hit_info, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
#ifdef __KERNELCC__
    ray.maxT = t_max - 1.0e-4f;

    // Payload for the alpha testing filter function
    FilterFunctionPayload payload;
    payload.render_data = &render_data;
    payload.random_number_generator = &random_number_generator;
    // Filling the payload with the last hit primitive index to avoid self intersections
    // (avoid that the ray intersects the triangle it is currently sitting on)
    payload.last_hit_primitive_index = last_hit_primitive_index;

#if UseSharedStackBVHTraversal == KERNEL_OPTION_TRUE
#if SharedStackBVHTraversalSize > 0
    hiprtSharedStackBuffer shared_stack_buffer{ SharedStackBVHTraversalSize, shared_stack_cache };
#else
    hiprtSharedStackBuffer shared_stack_buffer{ 0, nullptr };
#endif
    hiprtGlobalStack global_stack(render_data.global_traversal_stack_buffer, shared_stack_buffer);

    hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> traversal(render_data.geom, ray, global_stack, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#else
    hiprtGeomTraversalClosest traversal(render_data.geom, ray, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#endif

    hiprtHit shadow_ray_hit = traversal.getNextHit();
    if (!shadow_ray_hit.hasHit())
        return false;

    // If we're here, this means that we found a hit that is not
    // alpha-transparent with a distance < t_max so that's a hit and we're shadowed.

    // Reading the emission of the material
    int material_index = render_data.buffers.material_indices[shadow_ray_hit.primID];
    int emission_texture_index = render_data.buffers.materials_buffer[material_index].get_emission_texture_index();

    if (emission_texture_index != MaterialUtils::NO_TEXTURE)
    {
        float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, shadow_ray_hit.primID, render_data.buffers.texcoords, shadow_ray_hit.uv);
        out_light_hit_info.hit_emission = get_material_property<ColorRGB32F>(render_data, false, texcoords, emission_texture_index);

        // Using the already computed texcoords to get the shading normal
        out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), shadow_ray_hit.primID, shadow_ray_hit.uv, texcoords);
    }
    else
    {
        out_light_hit_info.hit_emission = render_data.buffers.materials_buffer[material_index].get_emission();

        float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, shadow_ray_hit.primID, render_data.buffers.texcoords, shadow_ray_hit.uv);
        out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), shadow_ray_hit.primID, shadow_ray_hit.uv, texcoords);
    }
    
    out_light_hit_info.hit_distance = shadow_ray_hit.t;
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
        shadow_ray_hit = intersect_scene_cpu(render_data, ray, last_hit_primitive_index, random_number_generator);
        if (!shadow_ray_hit.hasHit())
            return false;

        if (render_data.render_settings.do_alpha_testing)
            alpha = get_hit_base_color_alpha(render_data, shadow_ray_hit);
        else
            alpha = 1.0f;

        // Next ray origin
        ray.origin = ray.origin + ray.direction * shadow_ray_hit.t;
        cumulative_t += shadow_ray_hit.t;

        // We keep going as long as the alpha is < 1.0f meaning that we hit texture transparency
    } while (alpha < 1.0f && cumulative_t < t_max - 1.0e-4f);

    bool hit_found = shadow_ray_hit.hasHit() && cumulative_t < t_max - 1.0e-4f;

    if (hit_found)
    {
        // If we found a hit and that it is close enough (hit_found conditions)

        int material_index = render_data.buffers.material_indices[shadow_ray_hit.primID];
        int emission_texture_index = render_data.buffers.materials_buffer[material_index].get_emission_texture_index();

        if (emission_texture_index != MaterialUtils::NO_TEXTURE)
        {
            float2 texcoords = uv_interpolate(render_data.buffers.triangles_indices, shadow_ray_hit.primID, render_data.buffers.texcoords, shadow_ray_hit.uv);
            out_light_hit_info.hit_emission = get_material_property<ColorRGB32F>(render_data, false, texcoords, emission_texture_index);

            // Using the already computed texcoords to get the shading normal
            out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), shadow_ray_hit.primID, shadow_ray_hit.uv, texcoords);
        }
        else
        {
            out_light_hit_info.hit_emission = render_data.buffers.materials_buffer[material_index].get_emission();

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
