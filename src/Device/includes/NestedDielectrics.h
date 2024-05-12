/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef NESTED_DIELECTRICS_H
#define NESTED_DIELECTRICS_H

#include "Device/includes/RayPayload.h"
#include "HostDeviceCommon/RenderData.h"

/** References:
 * 
 * [1] [Simple Nested Dielectrics in Ray Traced Images, Schmidt 2002]
 */

// TODO remove
//struct NestedDielectricsPayload
//{
//    const HIPRTRenderData* render_data;
//    RayPayload* ray_payload;
//};

/**
 * Returns true if the intersection needs to be ignored, false if the intersection is not filtered (i.e. valid, accepted)
 */
#ifdef __KERNELCC__
HIPRT_DEVICE HIPRT_INLINE bool dielectric_priorities_cutout_filter(const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit)
#else
HIPRT_HOST HIPRT_INLINE bool dielectric_priorities_cutout_filter(const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit)
#endif
{
    //NestedDielectricsPayload* ray_payload = reinterpret_cast<NestedDielectricsPayload*>(payload);
    //if (!ray_payload->ray_payload->inside_volume)
    //    // Not even inside a dielectric, de facto accepting the intersection
    //    return false;

    //RendererMaterial material = ray_payload->render_data->buffers.materials_buffer[ray_payload->render_data->buffers.material_indices[hit.primID]];

    //if (material.dielectric_priority == 1)
    //    return false;
    //else
    //    return true;
}

#endif
