/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_LAMBERTIAN_H
#define DEVICE_LAMBERTIAN_H

#include "Device/includes/ONB.h"
#include "Device/includes/Sampling.h"
#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"

HIPRT_DEVICE static ColorRGB32F lambertian_brdf_eval(const DeviceUnpackedEffectiveMaterial& material, float NoL, float& pdf)
{
	pdf = 0.0f;

	if (NoL <= 0.0f)
		return ColorRGB32F(0.0f);

	pdf = NoL * hippt::M_INV_PI;

	return material.base_color * hippt::M_INV_PI;
}

HIPRT_DEVICE static float lambertian_brdf_pdf(float NoL)
{
	float pdf = 0.0f;

	if (NoL <= 0.0f)
		return 0.0f;

	return NoL * hippt::M_INV_PI;
}

/**
 * If sampleDirectionOnly is 'true',, this function samples only the BSDF without
 * evaluating the contribution or the PDF of the BSDF. This function will then always return
 * ColorRGB32F(0.0f) and the 'pdf' out parameter will always be set to 0.0f
 */
template <bool sampleDirectionOnly = false>
HIPRT_DEVICE static ColorRGB32F lambertian_brdf_sample(const DeviceUnpackedEffectiveMaterial& material,
													   const float3_t& geometric_normal,
													   const float3_t& shading_normal,
													   float3_t& sampled_direction,
													   float& pdf,
													   Xorshift32Generator& random_number_generator,
													   BSDFIncidentLightInfo& out_sampled_light_info)
{
	sampled_direction = cosine_weighted_sample_around_normal_world_space(shading_normal, random_number_generator);
	if (hippt::dot(sampled_direction, geometric_normal) <= 0.0f)
	{
		// Sampling below the geometry is going to lead to light leaks, invalidating
		pdf = 0.0f;

		return ColorRGB32F(0.0f);
	}

	out_sampled_light_info = BSDFIncidentLightInfo::LIGHT_DIRECTION_SAMPLED_FROM_DIFFUSE_LOBE;

	if constexpr (sampleDirectionOnly)
	{
		pdf = 0.0f;

		return ColorRGB32F(0.0f);
	}
	else
		return lambertian_brdf_eval(material, hippt::dot(shading_normal, sampled_direction), pdf);
}

#endif
