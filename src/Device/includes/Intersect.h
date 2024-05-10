/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INTERSECT_H
#define DEVICE_INTERSECT_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/HitInfo.h"
#include "Device/includes/Texture.h"
#include "HostDeviceCommon/RenderData.h"

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

HIPRT_HOST_DEVICE HIPRT_INLINE bool trace_ray(const HIPRTRenderData& render_data, hiprtRay ray, HitInfo& hit_info)
{
#ifdef __KERNELCC__
    hiprtGeomTraversalClosest tr(render_data.geom, ray);
    hiprtHit				  hit = tr.getNextHit();
#else
    hiprtHit hit = intersect_scene_cpu(render_data, ray);
#endif

    if (hit.hasHit())
    {
        hit_info.inter_point = ray.origin + hit.t * ray.direction;
        hit_info.primitive_index = hit.primID;
        // hit.normal is in object space, this simple approach will not work if using
        // multiple-levels BVH (TLAS/BLAS)
        hit_info.geometric_normal = hippt::normalize(hit.normal);

        int vertex_A_index = render_data.buffers.triangles_indices[hit_info.primitive_index * 3 + 0];
        if (render_data.buffers.has_vertex_normals[vertex_A_index])
            // Smooth normal available for the triangle
            hit_info.shading_normal = hippt::normalize(uv_interpolate(render_data.buffers.triangles_indices, hit_info.primitive_index, render_data.buffers.vertex_normals, hit.uv));
        else
            hit_info.shading_normal = hit_info.geometric_normal;

        hit_info.texcoords = uv_interpolate(render_data.buffers.triangles_indices, hit_info.primitive_index, render_data.buffers.texcoords, hit.uv);

        hit_info.t = hit.t;
        hit_info.uv = hit.uv; // TODO remove ?

        return true;
    }
    else
        return false;
}

/**
 * Returns true if in shadow, false otherwise
 */
HIPRT_HOST_DEVICE HIPRT_INLINE bool evaluate_shadow_ray(const HIPRTRenderData& render_data, hiprtRay ray, float t_max)
{
#ifdef __KERNELCC__
    ray.maxT = t_max - 1.0e-4f;

    hiprtGeomTraversalAnyHit traversal(render_data.geom, ray);
    hiprtHit aoHit = traversal.getNextHit();

    return aoHit.hasHit();
#else
    hiprtHit hit = intersect_scene_cpu(render_data, ray);
    return hit.hasHit() && hit.t < t_max - 1.0e-4f;
#endif // __KERNELCC__
}

#endif
