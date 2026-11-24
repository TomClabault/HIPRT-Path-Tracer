/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PT_WAVEFRONT_RESTIR_PT_WAVEFRONT_DATA_DEVICE_H
#define DEVICE_INCLUDES_RESTIR_PT_WAVEFRONT_RESTIR_PT_WAVEFRONT_DATA_DEVICE_H

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

/**
 * Per-path state of the wavefront ReSTIR PT initial candidates that the generic wavefront data (WavefrontDataDevice) does not hold: the
 * variables of the path that the kernels of one bounce pass to the kernels of the next bounce.
 *
 * All the buffers are indexed by the pixel index of the path
 */
struct ReSTIRPTWavefrontDataDevice
{
	// BSDF * cosine product of the bounces up to the reconnection vertex, see ReSTIR_PT_add_candidate()
	ColorRGB32F* path_unweighted_throughputs_up_to_rc_vertex = nullptr;
	// BSDF * cosine / PDF product of the bounces after the reconnection vertex, see ReSTIR_PT_add_candidate()
	ColorRGB32F* path_weighted_throughputs_after_rc_vertex = nullptr;
	// Throughput of the last sampled bounce. It is only multiplied into one of the two throughputs above once the deferred MIS of the next vertex
	// has added its candidate with the throughputs of the start of the bounce
	ColorRGB32F* path_pending_throughputs = nullptr;

	// Resampling weight sum of the initial candidates reservoir of the pixel, accumulated over the path trees of the pixel
	float* path_reservoir_weight_sums = nullptr;

	// Full precision direction of the next ray of the path. The generic wavefront data only keeps an octahedral encoding of the direction: the
	// direction would then not be the one that the BSDF sampled, with which the BSDF throughput of the bounce and the reconnection data of the
	// sample were computed
	float3_t* path_ray_directions = nullptr;
};

#endif // #ifndef DEVICE_INCLUDES_RESTIR_PT_WAVEFRONT_RESTIR_PT_WAVEFRONT_DATA_DEVICE_H
