/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H
#define HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"
#include "Device/includes/LightSampling/PDFConversion.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

/**
 * Information about a light sample taken from an emissive triangle
 */
struct LightSamplePointInformation
{
	float3_t light_source_normal = { 0.0f, 1.0f, 0.0f };

	// Index of the triangle in the whole scene (not just in the emissive triangles buffer)
	int emissive_triangle_global_index = -1;

	ColorRGB32F emission;
	float light_area = 1.0f;

	float3_t point_on_light = make_float3(0.0f, 0.0f, 0.0f);
	float area_measure_pdf	= 0.0f;

#if DirectLightSamplingStrategy == LSS_BASE_REGIR
	// The light sample may come from BSDF sampling with ReGIR and so we may have
	// information about the lobe that was sampled.
	BSDFIncidentLightInfo incident_light_info = BSDFIncidentLightInfo::NO_INFO;
#endif
};

template <int size>
struct LightSamplePointArray
{
	LightSamplePointInformation samples[size];

	HIPRT_DEVICE LightSamplePointInformation& operator[](int index)
	{
		return samples[index];
	}
};

/**
 * Information about a sampled emissive triangle
 */
struct LightSampleInformation
{
	int emissive_triangle_global_index = -1;

	float pdf = 0.0f;
};

template <int size>
struct LightSampleArray
{
	LightSampleInformation samples[size];

	HIPRT_DEVICE LightSampleInformation& operator[](int index)
	{
		return samples[index];
	}
};

#endif // #ifndef HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H
