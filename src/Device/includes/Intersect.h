/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INTERSECT_H
#define DEVICE_INTERSECT_H

#include "Device/includes/Dispersion.h"
#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Material.h"
#include "Device/includes/ONB.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Sampling.h"
#include "Device/includes/Texture.h"
#include "Device/includes/TriangleStructures.h"
#include "Device/functions/FilterFunction.h"

#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Math.h"

#if SharedStackBVHTraversalSize > 0
// This if is necessary to avoid declaring 0 size arrays if the
// shared stack traversal sizes are 0
__shared__ static int shared_stack_cache[SharedStackBVHTraversalSize * KernelWorkgroupThreadCount];
#endif

//#define __KERNELCC__

#ifdef __KERNELCC__

#if SharedStackBVHTraversalSize > 0
#define DECLARE_SHARED_STACK_BUFFER shared_stack_buffer{ SharedStackBVHTraversalSize, shared_stack_cache }
#else
#define DECLARE_SHARED_STACK_BUFFER shared_stack_buffer{ 0, nullptr }
#endif

#if UseSharedStackBVHTraversal == KERNEL_OPTION_TRUE
#define CONSTRUCT_HIPRT_CLOSEST_HIT_TRAVERSAL(traversal_variable_name) hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> traversal_variable_name(render_data.GPU_BVH, ray, global_stack, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0)
#define CONSTRUCT_HIPRT_ANY_HIT_TRAVERSAL(traversal_variable_name) hiprtGeomTraversalAnyHitCustomStack<hiprtGlobalStack> traversal_variable_name(render_data.GPU_BVH, ray, global_stack, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0)
#else
#define CONSTRUCT_HIPRT_CLOSEST_HIT_TRAVERSAL(traversal_variable_name) hiprtGeomTraversalClosest traversal_variable_name(render_data.GPU_BVH, ray, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#define CONSTRUCT_HIPRT_ANY_HIT_TRAVERSAL(traversal_variable_name) hiprtGeomTraversalAnyHit traversal_variable_name(render_data.GPU_BVH, ray, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#endif

#define DECLARE_HIPRT_CLOSEST_ANY_HIT_COMMON(render_data, ray, last_hit_primitive_index, random_number_generator) \
  /* Payload for the alpha testing filter function */                                                             \
  FilterFunctionPayload payload;                                                                                  \
  payload.render_data = &render_data;                                                                             \
  payload.random_number_generator = &random_number_generator;                                                     \
  /* Filling the payload with the last hit primitive index to avoid self intersections */                         \
  /* (avoid that the ray intersects the triangle it is currently sitting on) */                                   \
  payload.last_hit_primitive_index = last_hit_primitive_index;                                                    \
  payload.bounce = bounce;                                                                            \
                                                                                                                  \
  hiprtSharedStackBuffer DECLARE_SHARED_STACK_BUFFER;                                                             \
  hiprtGlobalStack global_stack(render_data.global_traversal_stack_buffer, shared_stack_buffer);




#define DECLARE_HIPRT_CLOSEST_HIT_TRAVERSAL(traversal_variable_name, render_data, ray, last_hit_primitive_index, random_number_generator) \
  DECLARE_HIPRT_CLOSEST_ANY_HIT_COMMON(render_data, ray, last_hit_primitive_index, random_number_generator);                              \
  CONSTRUCT_HIPRT_CLOSEST_HIT_TRAVERSAL(traversal_variable_name);

#define DECLARE_HIPRT_ANY_HIT_TRAVERSAL(traversal_variable_name, render_data, ray, last_hit_primitive_index, random_number_generator) \
  DECLARE_HIPRT_CLOSEST_ANY_HIT_COMMON(render_data, ray, last_hit_primitive_index, random_number_generator);                          \
  CONSTRUCT_HIPRT_ANY_HIT_TRAVERSAL(traversal_variable_name);

#endif

/* References:
 * 
 * [1] [Foundations of Game Engine Development: Rendering - Tangent/Bitangent calculation] http://foundationsofgameenginedev.com/#fged2
 */
HIPRT_DEVICE HIPRT_INLINE float3 normal_mapping(const HIPRTRenderData& render_data, int normal_map_texture_index, TriangleIndices triangle_vertex_indices, TriangleTexcoords& texcoords, const float2& interpolated_texcoords, const float3& surface_normal)
{
    // Calculating tangents and bitangents aligned with texture U and V coordinates
    float2 P0_texcoords = texcoords.x;
    float2 P1_texcoords = texcoords.y;
    float2 P2_texcoords = texcoords.z;

    float2 delta_P1P0_texcoords = P1_texcoords - P0_texcoords;
    float2 delta_P2P0_texcoords = P2_texcoords - P0_texcoords;

    float3 P0 = render_data.buffers.vertices_positions[triangle_vertex_indices.x];
    float3 P1 = render_data.buffers.vertices_positions[triangle_vertex_indices.y];
    float3 P2 = render_data.buffers.vertices_positions[triangle_vertex_indices.z];

    float3 edge_P0P1 = P1 - P0;
    float3 edge_P0P2 = P2 - P0;

    // To counter degenerate UVs
    constexpr float det_bias = 1.0e-6f;
    float det = delta_P1P0_texcoords.x * delta_P2P0_texcoords.y - delta_P1P0_texcoords.y * delta_P2P0_texcoords.x + det_bias;
    // Check if the det isn't too low to avoid degenerate geometries that can then cause NaNs
    float det_inverse = 1.0f / det;
    float3 T = (edge_P0P1 * delta_P2P0_texcoords.y - edge_P0P2 * delta_P1P0_texcoords.y) * det_inverse;
    float3 B = (edge_P0P2 * delta_P1P0_texcoords.x - edge_P0P1 * delta_P2P0_texcoords.x) * det_inverse;
    if (hippt::length2(T) < 1.0e-6f || hippt::length2(B) < 1.0e-6f)
        // The tangent or the bitangent is degenerate
        return surface_normal;

    ColorRGB32F normal = sample_texture_rgb_8bits(render_data.buffers.material_textures, normal_map_texture_index, /* is_srgb */ false, interpolated_texcoords);
    // Bringing the normal in [-x, x]. x doesn't really matter since we normalize the result anyway
    normal -= ColorRGB32F(0.5f);

    float3 normal_tangent_space = hippt::normalize(make_float3(normal.r, normal.g, normal.b));

    return local_to_world_frame(hippt::normalize(T), hippt::normalize(B), surface_normal, normal_tangent_space);
}

HIPRT_DEVICE HIPRT_INLINE float3 get_shading_normal(const HIPRTRenderData& render_data, const float3& geometric_normal, TriangleIndices triangle_vertex_indices, TriangleTexcoords triangle_texcoords, int primitive_index, const float2& uv, const float2& interpolated_texcoords)
{
    if (!render_data.render_settings.do_normal_mapping)
        return geometric_normal;

    // Do smooth shading first if we have vertex normals
    float3 surface_normal;
    if (render_data.buffers.has_vertex_normals[triangle_vertex_indices.x])
        // Smooth normal available for the triangle
        surface_normal = hippt::normalize(uv_interpolate(triangle_vertex_indices, render_data.buffers.vertex_normals, uv));
    else
        surface_normal = geometric_normal;

    // Do normal mapping if we have a normal map
    int material_index = render_data.buffers.material_indices[primitive_index];
    unsigned short int normal_map_texture_index = render_data.buffers.materials_buffer.get_normal_map_texture_index(material_index);
    if (normal_map_texture_index != MaterialConstants::NO_TEXTURE)
        surface_normal = normal_mapping(render_data, normal_map_texture_index, triangle_vertex_indices, triangle_texcoords, interpolated_texcoords, surface_normal);

    return surface_normal;
}

/**
 * Flips the surface normals if necessary such that they are facing us. 
 * 
 * The normals are only flipped if some conditions are met, read the 
 * comment in the function for more details
 */
HIPRT_DEVICE HIPRT_INLINE void fix_backfacing_normals(HitInfo& hit_info, const float3& view_direction)
{
    if (hippt::dot(view_direction, hit_info.geometric_normal) < 0.0f)
    {
        // The geometry isn't front-facing

        hit_info.geometric_normal *= -1.0f;
        hit_info.shading_normal *= -1.0f;
    }

    if (hippt::dot(view_direction, hit_info.shading_normal) < 0.0f)
        // Flipping the normal such that the view direction isn't below the shading hemisphere anymore
        hit_info.shading_normal *= -1.0f;

    // Now ensuring that a perfectly reflected direction (about the shading normal) doesn't go below the *geometric* surface
    float3 perfect_reflected_direction = reflect_ray(view_direction, hit_info.shading_normal);
    if (hippt::dot(perfect_reflected_direction, hit_info.geometric_normal) <= 0.0f)
    {
        // The perfectly reflected direction *is* below the geometric normal,
        // we're going to pull the shading normal towards the geometric normal such that
        // the perfectly reflected direction now is just an epsilon above the surface
        //
        // This is done by first computing a new reflected direction that is just above the surface
        // and then recomputing the new shading normal as the half vector between the new reflect direction
        // and the view direction

        constexpr float epsilon = 0.01;

        perfect_reflected_direction -= hippt::normalize((hippt::dot(perfect_reflected_direction, hit_info.geometric_normal) - epsilon) * hit_info.geometric_normal);

        // The new shading normal is the half vector between the pulled up reflected direction
        // and the view direction
        hit_info.shading_normal = hippt::normalize(view_direction + perfect_reflected_direction);
    }
}

#ifndef __KERNELCC__
#include "Renderer/BVH.h"
HIPRT_DEVICE HIPRT_INLINE hiprtHit intersect_scene_cpu(const HIPRTRenderData& render_data, const hiprtRay& ray, int last_hit_primitive_index, Xorshift32Generator& random_number_generator)
{
    FilterFunctionPayload filter_function_payload;
    filter_function_payload.render_data = &render_data;
    filter_function_payload.random_number_generator = &random_number_generator;
    // Filling the payload with the last hit primitive index to avoid self intersections
    // (avoid that the ray intersects the triangle it is currently sitting on)
    filter_function_payload.last_hit_primitive_index = last_hit_primitive_index;

    hiprtHit hiprtHit;
    render_data.cpu_only.bvh->intersect(ray, hiprtHit, &filter_function_payload);

    return hiprtHit;
}
#endif

/**
 * Returns true if a hit was found, false otherwise
 */
HIPRT_DEVICE HIPRT_INLINE bool trace_main_path_ray(const HIPRTRenderData& render_data, hiprtRay ray, RayPayload& in_out_ray_payload, HitInfo& out_hit_info, int last_hit_primitive_index, int bounce, Xorshift32Generator& random_number_generator)
{
#ifdef __KERNELCC__
    if (render_data.GPU_BVH == nullptr)
        // Empty scene --> no intersection
        return false;
#endif

    hiprtHit hit;
    bool skipping_volume_boundary = false;
    do
    {
#ifdef __KERNELCC__
        DECLARE_HIPRT_CLOSEST_HIT_TRAVERSAL(traversal, render_data, ray, last_hit_primitive_index, random_number_generator);
        
        hit = traversal.getNextHit();
#else
        hit = intersect_scene_cpu(render_data, ray, last_hit_primitive_index, random_number_generator);
#endif

        if (!hit.hasHit())
            return false;

        TriangleIndices triangle_vertex_indices = load_triangle_vertex_indices(render_data.buffers.triangles_indices, hit.primID);
        TriangleTexcoords triangle_texcoords = load_triangle_texcoords(render_data.buffers.texcoords, triangle_vertex_indices);

        out_hit_info.inter_point = ray.origin + hit.t * ray.direction;
        out_hit_info.primitive_index = hit.primID;
        out_hit_info.texcoords = uv_interpolate(triangle_texcoords, hit.uv);
        // TODO hit.normal is in object space, this simple approach will not work if using
        // multiple-levels BVH (TLAS/BLAS). We'll have to  transform by the BLAS transform
        out_hit_info.geometric_normal = hippt::normalize(hit.normal);
        out_hit_info.shading_normal = get_shading_normal(render_data, out_hit_info.geometric_normal, triangle_vertex_indices, triangle_texcoords, hit.primID, hit.uv, out_hit_info.texcoords);
        out_hit_info.t = hit.t;

        int material_index = render_data.buffers.material_indices[hit.primID];
        in_out_ray_payload.material = get_intersection_material(render_data, material_index, out_hit_info.texcoords);

        skipping_volume_boundary = in_out_ray_payload.volume_state.interior_stack.push(
            in_out_ray_payload.volume_state.incident_mat_index, in_out_ray_payload.volume_state.outgoing_mat_index, in_out_ray_payload.volume_state.inside_material, material_index, in_out_ray_payload.material.get_dielectric_priority());

        if (in_out_ray_payload.volume_state.inside_material)
            // If we're traveling inside a volume, accumulating the distance for Beer's law
            in_out_ray_payload.volume_state.distance_in_volume += hit.t;

        fix_backfacing_normals(out_hit_info, -ray.direction);

        if (skipping_volume_boundary)
        {
            // If we're skipping, the boundary, the ray just keeps going on its way
            ray.origin = out_hit_info.inter_point;

            // Don't forget to increment the distance traveled
            // TODO: Are we not double counting the distance here and a few lines above (where we set the .t, .uv, .geometric_normal, ...)
            in_out_ray_payload.volume_state.distance_in_volume += hit.t;
        }

    } while ((skipping_volume_boundary && hit.hasHit()));

    if (in_out_ray_payload.material.dispersion_scale > 0.0f && in_out_ray_payload.material.specular_transmission > 0.0f && in_out_ray_payload.volume_state.sampled_wavelength == 0.0f)
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
 * Returns true if in shadow (a hit was found before 't_max' distance)
 * Returns false if unoccluded
 */
HIPRT_DEVICE HIPRT_INLINE bool evaluate_shadow_ray_occluded(const HIPRTRenderData& render_data, hiprtRay ray, float t_max, int last_hit_primitive_index, int bounce, Xorshift32Generator& random_number_generator)
{
#ifdef __KERNELCC__
    if (render_data.GPU_BVH == nullptr)
        // Empty scene --> no intersection
        return false;
#endif

#ifdef __KERNELCC__
    ray.maxT = t_max - 1.0e-4f;

    DECLARE_HIPRT_ANY_HIT_TRAVERSAL(traversal, render_data, ray, last_hit_primitive_index, random_number_generator);

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
        // We should use ray tracing filter functions here instead of re-tracing new rays
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
 * Returns true if in shadow (a hit was found before 't_max' distance
 * Returns false if unoccluded
 * 
 * This function also uses NEE++ if enabled in the kernel options and this
 * function can update the visibility map of NEE++ if enabled in 'render_data.nee_plus_plus'
 */
HIPRT_DEVICE HIPRT_INLINE bool evaluate_shadow_ray_nee_plus_plus(HIPRTRenderData& render_data, hiprtRay ray, float t_max, int last_hit_primitive_index, NEEPlusPlusContext& nee_plus_plus_context, Xorshift32Generator& random_number_generator, int bounce)
{
#if DirectLightUseNEEPlusPlusRR == KERNEL_OPTION_TRUE && DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE
    bool shadow_ray_discarded = false;
    bool shadow_ray_occluded = false;

    if (render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics)
        // Updating the statistics
        hippt::atomic_fetch_add(render_data.nee_plus_plus.total_shadow_ray_queries, 1ull);

    bool nee_plus_plus_envmap_rr_disabled = nee_plus_plus_context.envmap && !render_data.nee_plus_plus.m_enable_nee_plus_plus_RR_for_envmap;
    bool nee_plus_plus_emissives_rr_disabled = !nee_plus_plus_context.envmap && !render_data.nee_plus_plus.m_enable_nee_plus_plus_RR_for_emissives;
    if (nee_plus_plus_envmap_rr_disabled || nee_plus_plus_emissives_rr_disabled)
    {
        // This is NEE++ RR for envmap sampling but envmap NEE++ RR is disabled
        nee_plus_plus_context.unoccluded_probability = 1.0f;

        if (render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics)
            // Updating the statistics
            hippt::atomic_fetch_add(render_data.nee_plus_plus.shadow_rays_actually_traced, 1ull);

        shadow_ray_occluded = evaluate_shadow_ray_occluded(render_data, ray, t_max, last_hit_primitive_index, bounce, random_number_generator);
        shadow_ray_discarded = false;
    }

    // Getting the matrix index from 'estimate_visibility_probability' in case we need to accumulate
    // visibility in the visibility map with 'accumulate_visibility'. If we do need to do that,
    // then that matrix index can be reused instead of being recomputed automatically by 'accumulate_visibility'
    // to save a little bit of computations
    unsigned int seed_before = random_number_generator.m_state.seed;

    unsigned int nee_plus_plus_hash_grid_cell_index;
    float visible_probability = nee_plus_plus_context.unoccluded_probability = render_data.nee_plus_plus.estimate_visibility_probability(nee_plus_plus_context, render_data.current_camera, nee_plus_plus_hash_grid_cell_index);
    bool likely_visible = random_number_generator() < visible_probability;

    if (likely_visible)
    {
        if (render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics)
            // Updating the statistics
            hippt::atomic_fetch_add(render_data.nee_plus_plus.shadow_rays_actually_traced, 1ull);

        // The shadow ray is likely visible, testing with a shadow ray
        shadow_ray_occluded = evaluate_shadow_ray_occluded(render_data, ray, t_max, last_hit_primitive_index, bounce, random_number_generator);
        shadow_ray_discarded = false;

        if (render_data.nee_plus_plus.m_update_visibility_map)
            render_data.nee_plus_plus.accumulate_visibility(!shadow_ray_occluded, nee_plus_plus_hash_grid_cell_index);
    }
    else
    {
        shadow_ray_discarded = true;

        // NEE++ tells us that these two points are going to be occluded so we're not testing
        // the shadow ray and assuming occluded instead
        shadow_ray_occluded = true;
    }
#else
    // Setting this to 1.0f if not using NEE++ so that is has no effect when the caller
    // divides by it
    nee_plus_plus_context.unoccluded_probability = 1.0f;

    bool shadow_ray_occluded = evaluate_shadow_ray_occluded(render_data, ray, t_max, last_hit_primitive_index, bounce, random_number_generator);

    // We may still want to update the visibility map
    if (render_data.nee_plus_plus.m_update_visibility_map && DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE)
    {
        unsigned int nee_plus_plus_hash_grid_cell_index = render_data.nee_plus_plus.get_visibility_map_index<true>(nee_plus_plus_context, render_data.current_camera);
        
        render_data.nee_plus_plus.accumulate_visibility(!shadow_ray_occluded, nee_plus_plus_hash_grid_cell_index);
    }
#endif

#if DirectLightNEEPlusPlusDisplayShadowRaysDiscarded == KERNEL_OPTION_TRUE
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t seed = blockIdx.x + blockIdx.y * gridDim.x + 1 + (threadIdx.y >= 4) * 1;
    uint32_t pixel_index = x + y * render_data.render_settings.render_resolution.x;

    Xorshift32Generator color_random(wang_hash(seed));

    ColorRGB32F block_color = ColorRGB32F(color_random(), color_random(), color_random()) * (render_data.render_settings.sample_number + 1);

    if (bounce == DirectLightNEEPlusPlusDisplayShadowRaysDiscardedBounce)
    {
        if (shadow_ray_discarded)
            render_data.buffers.accumulated_ray_colors[pixel_index] = ColorRGB32F();
        else
            render_data.buffers.accumulated_ray_colors[pixel_index] = block_color;
    }
#endif

    return shadow_ray_occluded;
}

/**
 * Returns true if in shadow, false otherwise.
 * 
 * Also, if a hit was found, outputs the emission of the material at the hit point in 'out_hit_emission'
 */
HIPRT_DEVICE HIPRT_INLINE bool evaluate_bsdf_light_sample_ray(const HIPRTRenderData& render_data, hiprtRay ray, float t_max, BSDFLightSampleRayHitInfo& out_light_hit_info, int last_hit_primitive_index, int bounce, Xorshift32Generator& random_number_generator)
{
#ifdef __KERNELCC__
    if (render_data.GPU_BVH == nullptr)
        // Empty scene --> no intersection
        return false;
#endif

#ifdef __KERNELCC__
    ray.maxT = t_max - 1.0e-4f;

    DECLARE_HIPRT_CLOSEST_HIT_TRAVERSAL(traversal, render_data, ray, last_hit_primitive_index, random_number_generator);

    hiprtHit shadow_ray_hit = traversal.getNextHit();
    if (!shadow_ray_hit.hasHit())
        return false;

    // If we're here, this means that we found a hit that is not
    // alpha-transparent with a distance < t_max so that's a hit and we're shadowed.

    // Reading the emission of the material
    int material_index = render_data.buffers.material_indices[shadow_ray_hit.primID];
    int emission_texture_index = render_data.buffers.materials_buffer.get_emission_texture_index(material_index);

    TriangleIndices triangle_vertex_indices = load_triangle_vertex_indices(render_data.buffers.triangles_indices, shadow_ray_hit.primID);
    TriangleTexcoords triangle_texcoords = load_triangle_texcoords(render_data.buffers.texcoords, triangle_vertex_indices);
    float2 interpolated_texcoords = uv_interpolate(triangle_texcoords, shadow_ray_hit.uv);

    if (emission_texture_index != MaterialConstants::NO_TEXTURE)
        out_light_hit_info.hit_emission = get_material_property<ColorRGB32F>(render_data, false, interpolated_texcoords, emission_texture_index);
        // Getting the shading normal
    else
        out_light_hit_info.hit_emission = render_data.buffers.materials_buffer.get_emission(material_index);

    out_light_hit_info.hit_interpolated_texcoords = interpolated_texcoords;
    out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), triangle_vertex_indices, triangle_texcoords, shadow_ray_hit.primID, shadow_ray_hit.uv, interpolated_texcoords);
    out_light_hit_info.hit_geometric_normal = hippt::normalize(shadow_ray_hit.normal);
    out_light_hit_info.hit_prim_index = shadow_ray_hit.primID;
    out_light_hit_info.hit_material_index = material_index;
    out_light_hit_info.hit_distance = shadow_ray_hit.t;

    return true;
#else
    float alpha = 1.0f;
    // The total distance of our ray. Incremented after each hit
    // (we may find multiple hits if we hit transparent texture
    // and keep intersecting the scene)
    float cumulative_t = 0.0f;

    // TODO DEBUG REMOVE THIS minT
    ray.minT = 1.0e-5f;

    hiprtHit shadow_ray_hit;
    do
    {
        // We should use ray tracing filter functions here instead of re-tracing new rays
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
        int emission_texture_index = render_data.buffers.materials_buffer.get_emission_texture_index(material_index);

        TriangleIndices triangle_vertex_indices = load_triangle_vertex_indices(render_data.buffers.triangles_indices, shadow_ray_hit.primID);
        TriangleTexcoords triangle_texcoords = load_triangle_texcoords(render_data.buffers.texcoords, triangle_vertex_indices);
        float2 interpolated_texcoords = uv_interpolate(triangle_texcoords, shadow_ray_hit.uv);

        if (emission_texture_index != MaterialConstants::NO_TEXTURE)
            out_light_hit_info.hit_emission = get_material_property<ColorRGB32F>(render_data, false, interpolated_texcoords, emission_texture_index);
        else 
            out_light_hit_info.hit_emission = render_data.buffers.materials_buffer.get_emission(material_index);

        out_light_hit_info.hit_interpolated_texcoords = interpolated_texcoords;
        out_light_hit_info.hit_shading_normal = get_shading_normal(render_data, hippt::normalize(shadow_ray_hit.normal), triangle_vertex_indices, triangle_texcoords, shadow_ray_hit.primID, shadow_ray_hit.uv, interpolated_texcoords);
        out_light_hit_info.hit_geometric_normal = hippt::normalize(shadow_ray_hit.normal);
        out_light_hit_info.hit_prim_index = shadow_ray_hit.primID;
        out_light_hit_info.hit_material_index = material_index;
        out_light_hit_info.hit_distance = cumulative_t;

        return true;
    }
    else
        return false;
#endif // __KERNELCC__
}

HIPRT_DEVICE hiprtHit simple_closest_hit(const HIPRTRenderData& render_data, hiprtRay ray, int last_primitive_index, Xorshift32Generator& random_number_generator)
{
    hiprtHit hit;

#ifdef __KERNELCC__
    // Payload for the alpha testing filter function
    FilterFunctionPayload payload;
    payload.render_data = &render_data;
    payload.random_number_generator = &random_number_generator;
    payload.last_hit_primitive_index = last_primitive_index;

#if UseSharedStackBVHTraversal == KERNEL_OPTION_TRUE
#if SharedStackBVHTraversalSize > 0
    hiprtSharedStackBuffer shared_stack_buffer{ SharedStackBVHTraversalSize, shared_stack_cache };
#else
    hiprtSharedStackBuffer shared_stack_buffer{ 0, nullptr };
#endif
    hiprtGlobalStack global_stack(render_data.global_traversal_stack_buffer, shared_stack_buffer);

    hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack> traversal(render_data.GPU_BVH, ray, global_stack, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#else
    hiprtGeomTraversalClosest traversal(render_data.GPU_BVH, ray, hiprtTraversalHintDefault, &payload, render_data.hiprt_function_table, 0);
#endif

    hit = traversal.getNextHit();
#else
    hit = intersect_scene_cpu(render_data, ray, last_primitive_index, random_number_generator);
#endif

    return hit;
}

#endif
