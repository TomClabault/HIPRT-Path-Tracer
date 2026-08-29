/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_BSDF_LIGHT_SAMPLE_RAY_HIT_INFO_H
#define HOST_DEVICE_COMMON_BSDF_LIGHT_SAMPLE_RAY_HIT_INFO_H

/**
 * Information returned by a shadow ray cast from a BSDF sample.
 *
 * This structure is filled by the 'evaluate_bsdf_light_sample_ray()'
 * function that is usually called for testing if a BSDF ray
 * (used by MIS) sees some emissive geometry or not.
 */
struct BSDFLightSampleRayHitInfo
{
	// TODO do we use this only for the area of the light? In which case we can just store the area of the light
	int hit_prim_index;
	// TODO is this used?
	int hit_material_index;
	float hit_distance;

	float2_t hit_interpolated_texcoords;
	float3_t hit_shading_normal;
	float3_t hit_geometric_normal;

	ColorRGB32F hit_emission;
};

#endif // #ifndef HOST_DEVICE_COMMON_BSDF_LIGHT_SAMPLE_RAY_HIT_INFO_H
