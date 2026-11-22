/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_DATA_H
#define DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_DATA_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterTrainingSample.h"
#include "HostDeviceCommon/Color.h"

static constexpr unsigned int WAVEFRONT_INVALID_SHADOW_QUEUE_SLOT = 0xFFFFFFFFu;
// The bits of WavefrontShadowWork::flags below this shift hold the WavefrontShadowFlags, the bits above hold the deterministic RNG key
static constexpr unsigned int WAVEFRONT_SHADOW_RNG_KEY_SHIFT = 5;

enum NEEShadowMode
{
	NEE_SHADOW_INLINE,
	NEE_SHADOW_WAVEFRONT,
};

enum WavefrontShadowFlags : unsigned int
{
	WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS = 1u << 0,
	WAVEFRONT_SHADOW_IS_ENVMAP		   = 1u << 1,
	// The contribution already includes the throughput of the path vertex it was sampled from and is not multiplied by the throughput of
	// the shadow sink
	WAVEFRONT_SHADOW_FINAL_COLOR				= 1u << 2,
	WAVEFRONT_SHADOW_UPDATE_LEARNING_TO_CLUSTER = 1u << 3,
};

enum WavefrontShadowPathFlags : unsigned int
{
	WAVEFRONT_SHADOW_PATH_PENDING_FINALIZE = 1u << 0,
};

enum WavefrontShadowPhase : unsigned int
{
	WAVEFRONT_SHADOW_PHASE_SHADE,
	WAVEFRONT_SHADOW_PHASE_DEFERRED_COMPLETION,
};

struct WavefrontShadowWork
{
	unsigned int path_index = 0;

	float3_t origin;
	// Finite shadow segments must retain the exact direction used to compute t_max. Quantization can move the endpoint onto the sampled light geometry.
	float3_t direction;
	float t_max = 0.0f;

	int last_hit_primitive_index = -1;
	unsigned int flags			 = 0;

	ColorRGB32F contribution_if_visible;
	float3_t point_on_light;
};

struct WavefrontLearningToClusterShadowSidecar
{
	unsigned int spatial_training_sample_index	  = 0;
	unsigned int learning_to_cluster_sample_index = 0;
	float spatial_radiance_weight_if_visible	  = 0.0f;
	float cluster_reward_if_visible				  = 0.0f;
};

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_DATA_H
