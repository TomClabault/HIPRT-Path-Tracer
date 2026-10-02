/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_DATA_H
#define DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_DATA_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeDirectIlluminationTrainingSample.h"
#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeLearningToClusterTrainingSample.h"
#include "Device/includes/Neural/NISML/NISMLTrainingSample.h"
#include "HostDeviceCommon/Color.h"

static constexpr unsigned int WAVEFRONT_INVALID_SHADOW_QUEUE_SLOT = 0xFFFFFFFFu;

enum NEEShadowMode
{
	NEE_SHADOW_INLINE,
	NEE_SHADOW_WAVEFRONT,
};

enum WavefrontShadowFlags : unsigned int
{
	WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS			= 1u << 0,
	WAVEFRONT_SHADOW_IS_ENVMAP					= 1u << 1,
	WAVEFRONT_SHADOW_EMISSIVE_RAW				= 1u << 2,
	WAVEFRONT_SHADOW_ENVMAP_RAW					= 1u << 3,
	WAVEFRONT_SHADOW_FINAL_COLOR				= 1u << 4,
	WAVEFRONT_SHADOW_UPDATE_LEARNING_TO_CLUSTER = 1u << 5,
	WAVEFRONT_SHADOW_UPDATE_NISML				= 1u << 6,
};

enum WavefrontShadowPathFlags : unsigned int
{
	WAVEFRONT_SHADOW_PATH_PRIMARY		   = 1u << 0,
	WAVEFRONT_SHADOW_PATH_PENDING_FINALIZE = 1u << 1,
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

struct WavefrontNISMLShadowSidecar
{
	IlluminationAwareKDTreeDirectIlluminationTrainingSample spatial_training_sample{};
	NISMLTrainingSample training_record{};

	float spatial_radiance_weight_if_visible = 0.0f;
	float point_on_light_pdf_if_visible		 = 0.0f;
	float contribution_luminance_if_visible	 = 0.0f;

	unsigned int append_spatial_training_sample = 0;
	unsigned int append_training_record			= 0;
	unsigned int training_rng_seed				= 0;
};

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_DATA_H
