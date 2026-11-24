/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GBUFFER_DEVICE_H
#define GBUFFER_DEVICE_H

#include "Device/includes/RayVolumeState.h"

#include "HostDeviceCommon/Material/MaterialPacked.h"

/**
 * Lobe record of a pixel for the target function BSDF approximation of the union domain reuse of ReSTIR PT (see
 * SpatialTargetFunctionBSDFApproximation.h), two words loaded together. 'coefficients': bits 0..15 Cs (unorm of Cs / 4), bits 16..31 Cd0 (unorm, rounded
 * up). 'parameters': bits 0..15 Cd1 (unorm, rounded up), bits 16..23 alpha (log code), bits 24..30 Fresnel code, bit 31 metal
 */
struct alignas(8) ReSTIRPTTargetFunctionBSDFApproximationPixelLobeRecord
{
	unsigned int coefficients = 0;
	unsigned int parameters	  = 0;
};

// Structure of arrays for the data contained in the pixels of the GBuffer
//
// If you want the roughness of the pixel (X, Y) = [50, 0] for example,
// get it at materials[50].get_roughness()
struct GBufferDevice
{
	HIPRT_HOST_DEVICE float3_t get_view_direction(float3_t camera_position, int pixel_index) const
	{
		return hippt::normalize(camera_position - primary_hit_position[pixel_index]);
	}

	DevicePackedEffectiveMaterial* materials = nullptr;

	int* first_hit_prim_index	   = nullptr;
	float3_t* primary_hit_position = nullptr;

	// We need both normals to correct the black fringes from the microfacet
	// model when used with smooth normals / normal mapping
	Octahedral24BitNormalPadded32b* shading_normals	  = nullptr;
	Octahedral24BitNormalPadded32b* geometric_normals = nullptr;

	// Analytic lobe of the pixel material for the target function BSDF approximation of the union domain reuse of ReSTIR PT with the principled BSDF
	// (SpatialTargetFunctionBSDFApproximation.h), written by CameraRays, read with one load by the MIS weights. The padding byte of 'shading_normals' holds
	// the flags of the pixel for that approximation. Only allocated on the GPU while the approximation is used with the principled BSDF, nullptr otherwise
	ReSTIRPTTargetFunctionBSDFApproximationPixelLobeRecord* target_function_bsdf_approximation_lobe_records = nullptr;
};

#endif // #ifndef GBUFFER_DEVICE_H
