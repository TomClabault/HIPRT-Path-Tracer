/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PG_SPLATTING_SAMPLE_SOA_DEVICE_H
#define DEVICE_INCLUDES_RESTIR_PG_SPLATTING_SAMPLE_SOA_DEVICE_H

#include "Device/includes/ReSTIR/PG/SplattingSample.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

struct ReSTIRPGSplattingSampleSoADevice
{
	float3_t* position			 = nullptr;
	float3_t* normal			 = nullptr;
	float3_t* incident_direction = nullptr;

	HIPRT_DEVICE unsigned int get_soa_index(int2_t render_resolution, unsigned int pixel_x, unsigned int pixel_y, unsigned int bounce) const
	{
		unsigned int pixel_count  = render_resolution.x * render_resolution.y;
		unsigned int sample_index = pixel_x + pixel_y * render_resolution.x + bounce * pixel_count;

		return sample_index;
	}

	HIPRT_DEVICE ReSTIRPGSplattingSample read_sample(int2_t render_resolution, unsigned int pixel_x, unsigned int pixel_y, unsigned int bounce) const
	{
		unsigned int sample_index = get_soa_index(render_resolution, pixel_x, pixel_y, bounce);

		ReSTIRPGSplattingSample sample;
		sample.position			  = position[sample_index];
		sample.normal			  = normal[sample_index];
		sample.incident_direction = incident_direction[sample_index];

		return sample;
	}

	HIPRT_DEVICE void store_sample(
		const ReSTIRPGSplattingSample& sample, int2_t render_resolution, unsigned int pixel_x, unsigned int pixel_y, unsigned int bounce) const
	{
		unsigned int sample_index = get_soa_index(render_resolution, pixel_x, pixel_y, bounce);

		position[sample_index]			 = sample.position;
		normal[sample_index]			 = sample.normal;
		incident_direction[sample_index] = sample.incident_direction;
	}

	HIPRT_DEVICE void invalidate_sample(int2_t render_resolution, unsigned int pixel_x, unsigned int pixel_y, unsigned int bounce) const
	{
		unsigned int pixel_count  = render_resolution.x * render_resolution.y;
		unsigned int sample_index = pixel_x + pixel_y * render_resolution.x + bounce * pixel_count;

		normal[sample_index] = make_float3(0.0f, 0.0f, 0.0f);
	}
};

#endif
