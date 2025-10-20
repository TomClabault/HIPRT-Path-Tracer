/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H
#define HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"
#include "Device/includes/LightSampling/PDFConversion.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

struct LightSampleInformation
{
    float3 light_source_normal = { 0.0f, 1.0f, 0.0f };

    // Index of the triangle in the whole scene (not just in the emissive triangles buffer)
    int emissive_triangle_global_index = -1;

    ColorRGB32F emission;
    float light_area = 1.0f;

    float3 point_on_light = make_float3(0.0f, 0.0f, 0.0f);
    float area_measure_pdf = 0.0f;

#if DirectLightSamplingBaseStrategy == LSS_BASE_REGIR
    // Used for replaying that random sample when necessary
    unsigned int sample_random_seed;
    // The light sample may come from BSDF sampling with ReGIR and so we may have
	// information about the lobe that was sampled.
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
#endif
};

#endif
