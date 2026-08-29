/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDF_CONTEXT_H
#define DEVICE_INCLUDES_BSDF_CONTEXT_H

#include "Device/includes/BSDFs/BSDFIncidentLightInfo.h"
#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "Device/includes/RayVolumeState.h"

struct BSDFContext
{
	DeviceUnpackedEffectiveMaterial& material;
	RayVolumeState& volume_state;

	float3_t view_direction		= make_float3(-1.0f, -1.0f, -1.0f);
	float3_t shading_normal		= make_float3(-1.0f, -1.0f, -1.0f);
	float3_t geometric_normal	= make_float3(-1.0f, -1.0f, -1.0f);
	float3_t to_light_direction = make_float3(-1.0f, -1.0f, -1.0f);

	BSDFIncidentLightInfo& incident_light_info;

	float accumulated_path_roughness = 0.0f;

	// Whether or not to modify the volume state of the ray as the BSDF is sampled / evaluated.
	//
	// For example, if the ray is currently refracting out of a glass material, and 'update_ray_volume_state' == true,
	// the ray volume state of the ray will be updated and the glass object will be popped out of the
	// nested dielectrics stack
	bool update_ray_volume_state = false;

	// Whether or not to regularize the BSDF when sampling/evaluating it
	MicrofacetRegularization::RegularizationMode bsdf_regularization_mode = MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION;

	/**
	 * 'to_light_direction' is only needed if evaluating the BSDF // TODO create a separate eval context and sampling context
	 * 'incident_light_info' should be passed as BSDFIncidentLightInfo::NO_INFO if you don't care about what lobe the BSDF sampled of if you don't have the
	 * information about what lobe the 'to_light_direction' comes from (during NEE light sampling for example)
	 */
	HIPRT_HOST_DEVICE BSDFContext(
							const float3_t& view_direction_,
							const float3_t& shading_normal_,
							const float3_t& geometric_normal_,
							const float3_t& to_light_direction_,
							BSDFIncidentLightInfo& incident_light_info_,
							RayVolumeState& ray_volume_state_,
							bool update_ray_volume_state_,
							DeviceUnpackedEffectiveMaterial& material_,
							float accumulated_path_roughness_,
							MicrofacetRegularization::RegularizationMode regularize_bsdf = MicrofacetRegularization::RegularizationMode::NO_REGULARIZATION)
		: material(material_), volume_state(ray_volume_state_), view_direction(view_direction_), shading_normal(shading_normal_),
		  geometric_normal(geometric_normal_), to_light_direction(to_light_direction_), incident_light_info(incident_light_info_),
		  accumulated_path_roughness(accumulated_path_roughness_), update_ray_volume_state(update_ray_volume_state_), bsdf_regularization_mode(regularize_bsdf)
	{
	}
};

#endif // #ifndef DEVICE_INCLUDES_BSDF_CONTEXT_H
