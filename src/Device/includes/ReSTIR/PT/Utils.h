/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_PT_UTILS_H
#define DEVICE_RESTIR_PT_UTILS_H

#include "Device/includes/BSDFs/BSDFDispatcher.h"
#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/Envmap.h"
#include "Device/includes/ReSTIR/NeighborSimilarity.h"
#include "Device/includes/ReSTIR/Surface.h"

#include "HostDeviceCommon/RenderData.h"

/**
 * Tests the visibility of the sample containde in 'reservoir' from the given shading point and kills the reservoir
 * if the visibility is occluded
 *
 * Returns true if the reservoir was killed, false otherwise
 */
HIPRT_DEVICE bool ReSTIR_PT_visibility_validation(const HIPRTRenderData& render_data,
												  ReSTIRPTReservoir& reservoir,
												  float3_t shading_point,
												  int last_hit_primitive_index,
												  Xorshift32Generator& random_number_generator)
{
	// A zero-contribution reservoir has no sample that needs a visibility ray.
	if (reservoir.UCW <= 0.0f)
		return false;

	float distance_to_sample_point;
	float3_t sample_direction;

	if (reservoir.sample.is_envmap_path())
	{
		// For envmap path, the direction is stored in the 'rc_vertex' value
		sample_direction		 = reservoir.sample.rc_vertex;
		distance_to_sample_point = 1.0e35f;
	}
	else
	{
		// Not an envmap path, the direction is the difference between the current shading
		// point and the reconnection point

		sample_direction = reservoir.sample.rc_vertex - shading_point;

		distance_to_sample_point = hippt::length(sample_direction);
		if (distance_to_sample_point <= 1.0e-6f)
		{
			// To avoid numerical instabilities, killing the reservoir
			reservoir.UCW = 0.0f;

			return true;
		}

		sample_direction /= distance_to_sample_point;
	}

	hiprtRay shadow_ray;
	shadow_ray.origin	 = shading_point;
	shadow_ray.direction = sample_direction;

	Xorshift32Generator random_number_generator_alpha_test(reservoir.sample.visible_to_sample_point_alpha_test_random_seed);

	bool visible =
		!evaluate_shadow_ray_occluded(render_data, shadow_ray, distance_to_sample_point, last_hit_primitive_index, random_number_generator_alpha_test);
	if (!visible)
	{
		reservoir.UCW = 0.0f;

		return true;
	}

	return false;
}

/**
 * Brings 'volume_state', the volume state at the visible point as reconstruct_first_hit() leaves it (visible point material pushed), to the volume
 * state that the path tracer has when it evaluates the BSDF at the sample point: the stack update of the BSDF sample at the visible point, then
 * the push of the material of the sample point by trace_main_path_ray()
 *
 * The material pushed is the one of the sample point, not of the visible point. Through a glass surface that does not close a volume (a pane in
 * front of a picture), a second push of the glass of the visible point would mark the ray as exiting the glass at the sample point: the BSDF
 * evaluation of an opaque sample point then only keeps its glass lobe and returns 0
 */
// Forced inline: out of line, the volume state and the material passed by reference live in scratch memory in the caller
HIPRT_DEVICE HIPRT_FORCEINLINE void ReSTIR_PT_update_volume_state_for_sample_point(const HIPRTRenderData& render_data,
																				   RayVolumeState& volume_state,
																				   DeviceUnpackedPrincipledFullMaterial& material_at_visible_point,
																				   BSDFIncidentLightInfo incident_light_info_visible_point,
																				   int sample_point_material_index)
{
	// The BSDF sampling pops the visible point off the stack when it reflects and when it refracts through a thin walled glass (both interfaces of
	// a thin walled glass are in the one surface, the ray leaves the glass right away)
	bool refracted			= bsdf_incident_light_info_transmission_lobe(incident_light_info_visible_point);
	bool refracted_by_glass = (incident_light_info_visible_point & BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE) != 0;
	if (!refracted || (refracted_by_glass && material_at_visible_point.thin_walled))
		volume_state.interior_stack.pop(false);

	// If we refracted at the visible point, the material of the visible point stays in the interior stack of the ray volume state so that the BSDF
	// evaluation at the sample point has the expected ray volume state (incident medium of the sample point)
	unsigned char sample_point_dielectric_priority = get_intersection_dielectric_priority(render_data, sample_point_material_index);
	volume_state.interior_stack.push(sample_point_material_index, sample_point_dielectric_priority, volume_state.incident_mat_index,
									 volume_state.outgoing_mat_index, volume_state.inside_material);
}

#endif // #ifndef DEVICE_RESTIR_PT_UTILS_H
