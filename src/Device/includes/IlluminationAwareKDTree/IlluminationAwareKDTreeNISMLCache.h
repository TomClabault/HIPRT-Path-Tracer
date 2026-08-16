/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NISML_CACHE_H
#define DEVICE_INCLUDES_ILLUMINATION_AWARE_KD_TREE_ILLUMINATION_AWARE_KD_TREE_NISML_CACHE_H

#include "Device/includes/IlluminationAwareKDTree/IlluminationAwareKDTreeSurfaceNormalFace.h"
#include "HostDeviceCommon/Maths/VecTypes.h"

static constexpr unsigned int ILLUMINATION_AWARE_KD_TREE_NISML_CLUSTER_COUNT		   = 64;
static constexpr unsigned int ILLUMINATION_AWARE_KD_TREE_NISML_NORMAL_FACE_COUNT	   = static_cast<unsigned int>(SurfaceNormalFace_Count);
static constexpr unsigned int ILLUMINATION_AWARE_KD_TREE_NISML_HASH_ENTRY_EMPTY		   = 0;
static constexpr unsigned int ILLUMINATION_AWARE_KD_TREE_NISML_HASH_ENTRY_INITIALIZING = 1;
static constexpr unsigned int ILLUMINATION_AWARE_KD_TREE_NISML_HASH_ENTRY_READY		   = 2;

struct IlluminationAwareKDTreeNISMLCache
{
	float log_importances[ILLUMINATION_AWARE_KD_TREE_NISML_CLUSTER_COUNT] = {};

	float3_t representative_position	   = float3_t(0.0f, 0.0f, 0.0f);
	float3_t representative_view_direction = float3_t(0.0f, 0.0f, 0.0f);
	float3_t representative_normal		   = float3_t(0.0f, 0.0f, 0.0f);

	float representative_sg_specular_weight = 0.0f;
	float representative_alpha_x			= 0.0f;
	float representative_alpha_y			= 0.0f;
};

#endif
