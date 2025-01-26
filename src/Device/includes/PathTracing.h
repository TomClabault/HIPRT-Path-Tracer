/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PATH_TRACING_H
#define DEVICE_INCLUDES_PATH_TRACING_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/MISBSDFRayReuse.h"

#include "HostDeviceCommon/RenderData.h"

HIPRT_HOST_DEVICE bool path_tracing_find_indirect_bounce_intersection(HIPRTRenderData& render_data, hiprtRay ray, RayPayload& out_ray_payload, HitInfo& out_closest_hit_info, int bounce, MISBSDFRayReuse mis_reuse, Xorshift32Generator& random_number_generator)
{
	if (mis_reuse.has_ray())
		// Reusing a BSDF MIS ray if there is one available
		return reuse_mis_ray(render_data, -ray.direction, out_ray_payload, out_closest_hit_info, mis_reuse);
	else
		// Not tracing for the primary ray because this has already been done in the camera ray pass
		return trace_ray(render_data, ray, out_ray_payload, out_closest_hit_info, out_closest_hit_info.primitive_index, bounce, random_number_generator);
}

#endif
