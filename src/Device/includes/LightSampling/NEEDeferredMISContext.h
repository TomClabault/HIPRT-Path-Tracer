/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H
#define DEVICE_INCLUDES_LIGHT_SAMPLING_NEE_DEFERRED_MIS_CONTEXT_H

#include "Device/includes/Material.h"

struct NEEDeferredMISContext
{
	float3_t last_view_direction;
	float3_t last_shading_point;
	float3_t last_shading_normal;
	DeviceUnpackedEffectiveMaterial last_material;

	// Ray throughput before multiplication with last_bsdf_throughput
	ColorRGB32F last_ray_throughput;

	// BSDF * cos_theta / pdf
	ColorRGB32F last_bsdf_throughput;
	float last_bsdf_sample_pdf;
};

#endif
