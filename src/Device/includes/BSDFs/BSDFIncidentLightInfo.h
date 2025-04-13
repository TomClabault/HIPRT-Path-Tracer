/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_BSDF_EVAL_INCIDENT_LIGHT_INFO_H
#define DEVICE_BSDF_EVAL_INCIDENT_LIGHT_INFO_H

enum BSDFIncidentLightInfo
{
	// Default value: nothing is assumed about the incident light direction
	NO_INFO = 0,

	// The additional information below can be used by the bsdf_eval() function to
	// avoid evaluating delta lobes (such as a perfectly smooth clearcoat lobe,
	// glass lobe, specualr lobe, ...) and save some performance.
	//
	// In such scenarios, the BSDF evaluation will still be correct because delta distribution
	// lobes will evaluate to 0 anyways if they are evaluated with a direction that
	// was not sampled from the lobe itself.
	//
	// For example, consider a clearcoat diffuse lobe. If bsdf_eval() is called with an
	// incident light direction that was sampled from the diffuse lobe, the perfectly smooth clearcoat lobe
	// is going to have its contribution evaluate to 0 because there is no chance that the sampled
	// diffuse direction perfectly aligns with the delta of the smooth clearcoat lobe
	//
	// Same with all the other lobes that can be delta distributions
	//
	// Using bit shifts for the values here so that it can be used easily by ReSTIR DI
	LIGHT_DIRECTION_SAMPLED_FROM_COAT_LOBE = 1 << 1,
	LIGHT_DIRECTION_SAMPLED_FROM_FIRST_METAL_LOBE = 1 << 2,
	LIGHT_DIRECTION_SAMPLED_FROM_SECOND_METAL_LOBE = 1 << 3,
	LIGHT_DIRECTION_SAMPLED_FROM_SPECULAR_LOBE = 1 << 4,
	LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFLECT_LOBE = 1 << 5,
	LIGHT_DIRECTION_SAMPLED_FROM_GLASS_REFRACT_LOBE = 1 << 6,
	LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE = 1 << 7,
	LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_TRANSMISSION_LOBE = 1 << 8,

	// This can be used if the incident light direction comes from sampling a light in the scene
	// from example
	LIGHT_DIRECTION_NOT_SAMPLED_FROM_BSDF
};

#endif
