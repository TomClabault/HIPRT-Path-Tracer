/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_UTILS_H
#define DEVICE_RESTIR_GI_UTILS_H 

#define RESTIR_GI_RECONNECTION_SURFACE_NORMAL_ENVMAP_VALUE -42.0f

HIPRT_HOST_DEVICE bool ReSTIR_GI_is_envmap_path(float3 reconnection_surface_normal)
{
	return reconnection_surface_normal.x == RESTIR_GI_RECONNECTION_SURFACE_NORMAL_ENVMAP_VALUE;
}

#endif
