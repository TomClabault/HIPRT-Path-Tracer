/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDFS_SHEEN_LTC
#define DEVICE_INCLUDES_BSDFS_SHEEN_LTC

#include "Device/includes/BSDFs/LTCsData/ZeltnerSheenLTCFitData.h"
#include "Device/includes/ONB.h"
#include "Device/includes/Texture.h"

#include "HostDeviceCommon/Color.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/RenderData.h"

/**
 * Reference:
 *
 * [1] [Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines] https://tizianzeltner.com/projects/Zeltner2022Practical/
 * [2] [Real-Time Polygonal-Light Shading with Linearly Transformed Cosines] https://eheitzresearch.wordpress.com/415-2/
 * [3] [Blender's Cycles Implementation] https://github.com/blender/cycles/blob/main/src/kernel/closure/bsdf_sheen.h
 */

HIPRT_DEVICE static float sheen_eval_ltc(const float3_t& to_light_direction_standard, const ColorRGB32F& AiBiRi)
{
	// AiBiRi are the parameters of the LTC such that
	//        { Ai 0  Bi }
	// M^-1 = { 0  Ai 0  }
	//		  { 0  0  1  }
	//
	// Bringing the to_light_direction into the "LTC space",
	// with identity transformation is thus done by multiplying
	// the direction by the M^-1 matrix
	float3_t light_dir_original = make_float3(to_light_direction_standard.x * AiBiRi.r + to_light_direction_standard.z * AiBiRi.g,
											  to_light_direction_standard.y * AiBiRi.r, to_light_direction_standard.z);

	float length = hippt::length(light_dir_original);
	light_dir_original /= length; // Normalization

	// Determinant of M^-1
	float M_inv_determinant = AiBiRi.r * AiBiRi.r;
	float jacobian			= M_inv_determinant / (length * length * length);

	return light_dir_original.z * hippt::M_INV_PI * jacobian;
}

HIPRT_DEVICE static ColorRGB32F read_LTC_parameters(const HIPRTRenderData& render_data, float roughness, float cos_theta)
{
	const void* ltc_parameters_texture_pointer;
#ifdef __KERNELCC__
	ltc_parameters_texture_pointer = &render_data.bsdfs_data.ltcs_data.sheen_zeltner_texture_ltc_params;
#else
	ltc_parameters_texture_pointer = render_data.bsdfs_data.ltcs_data.sheen_zeltner_texture_ltc_params;
#endif

	float2_t parameters_uv = make_float2(cos_theta, hippt::clamp(0.0f, 1.0f, roughness));
	return sample_texture_rgb_32bits(ltc_parameters_texture_pointer, parameters_uv, 0, false, false);
}

/**
 * Returns the phi angle of a direction given in a canonical frame with Z up
 */
HIPRT_DEVICE static float get_phi(const float3_t& direction)
{
	float p = atan2(direction.y, direction.x);
	if (p < 0.0f)
		p += hippt::M_TWO_PI;

	return p;
}

HIPRT_DEVICE static float get_sheen_ltc_reflectance(const HIPRTRenderData& render_data,
													const DeviceUnpackedEffectiveMaterial& material,
													const float3_t& local_view_direction)
{
	return read_LTC_parameters(render_data, material.sheen_roughness, local_view_direction.z).b;
}

HIPRT_DEVICE static ColorRGB32F sheen_ltc_eval(const HIPRTRenderData& render_data,
											   const DeviceUnpackedEffectiveMaterial& material,
											   const float3_t& local_to_light_direction,
											   const float3_t& local_view_direction,
											   float& out_pdf,
											   float& out_sheen_reflectance)
{
	if (local_view_direction.z <= 0.0f || local_to_light_direction.z <= 0.0f)
	{
		out_pdf = 0.0f;
		if (local_view_direction.z > 0.0f)
			out_sheen_reflectance = get_sheen_ltc_reflectance(render_data, material, local_view_direction);
		else
			out_sheen_reflectance = 0.0f;

		return ColorRGB32F(0.0f);
	}

	// The LTC needs to be evaluated in a Z-up coordinate frame with view direction aligned
	// with phi=0 (so no rotation on the X/Y plane).
	//
	// We're thus computing the phi angle and then rotating the to light direction
	// on that phi angle so that the view direction is at phi=0.
	float phi = get_phi(local_view_direction);

	// Rotating the to light direction around z axis such that the view direction is aligned
	// with phi=0 (because we computed the rotation angle, phi, from the view direction)
	float3_t to_light_standard_frame = rotate_vector(local_to_light_direction, make_float3(0.0f, 0.0f, 1.0f), -phi);

	ColorRGB32F AiBiRi = read_LTC_parameters(render_data, material.sheen_roughness, local_view_direction.z);
	float Do		   = sheen_eval_ltc(to_light_standard_frame, AiBiRi);

	out_pdf				  = Do;
	out_sheen_reflectance = AiBiRi.b;
	// The cosine term is included in the LTC distribution but the renderer expects that
	// the cosine term isn't included in the BSDFs so we cancel it here.
	return material.sheen_color * AiBiRi.b * Do / local_to_light_direction.z;
}

HIPRT_DEVICE static float sheen_ltc_pdf(const HIPRTRenderData& render_data,
										const DeviceUnpackedEffectiveMaterial& material,
										const float3_t& local_to_light_direction,
										const float3_t& local_view_direction)
{
	if (local_view_direction.z <= 0.0f || local_to_light_direction.z <= 0.0f)
		return 0.0f;

	// The LTC needs to be evaluated in a Z-up coordinate frame with view direction aligned
	// with phi=0 (so no rotation on the X/Y plane).
	//
	// We're thus computing the phi angle and then rotating the to light direction backwards
	// on that phi angle so that the view direction is at phi=0.
	float phi = get_phi(local_view_direction);

	// Rotating the to light direction around z axis such that the view direction is aligned
	// with phi=0 (because we computed the rotation angle, phi, from the view direction)
	float3_t to_light_standard_frame = rotate_vector(local_to_light_direction, make_float3(0.0f, 0.0f, 1.0f), -phi);

	ColorRGB32F AiBiRi = read_LTC_parameters(render_data, material.sheen_roughness, local_view_direction.z);
	float Do		   = sheen_eval_ltc(to_light_standard_frame, AiBiRi);

	return Do;
}

HIPRT_DEVICE static float3_t sheen_ltc_sample(const HIPRTRenderData& render_data,
											  const DeviceUnpackedEffectiveMaterial& material,
											  const float3_t& local_view_direction,
											  const float3_t& shading_normal,
											  Xorshift32Generator& random_number_generator)
{
	// Sampling a direction in the original space of the LTC
	float3_t cosine_sample = cosine_weighted_sample_z_up_frame(random_number_generator);

	ColorRGB32F AiBiRi = read_LTC_parameters(render_data, material.sheen_roughness, local_view_direction.z);

	// And then from the transformation matrix of the LTC, we're going to bring that
	// sampled direction back to the local space of the BSDF (shading/tangent space)
	// For that, we need to multiply that standard sampled direction by the matrix M
	// which is (M^-1)^-1 and we already have M^-1 from AiRiBi, we just to invert it
	// and its inverse actually is
	//
	//      { 1/Ai       0      -Bi/Ai }
	//  M = { 0         1/Ai       0   }
	//		{ 0          0         1   }
	//

	float Ai_inv = 1.0f / AiBiRi.r;
	float Bi	 = AiBiRi.g;

	// Creating the sampled direction in a space at phi=0
	float3_t sampled_direction_ltc_space =
		hippt::normalize(make_float3(cosine_sample.x * Ai_inv - cosine_sample.z * Bi * Ai_inv, cosine_sample.y * Ai_inv, cosine_sample.z));

	// Bringing out of the phi=0 configuration by rotating
	return rotate_vector(sampled_direction_ltc_space, make_float3(0.0f, 0.0f, 1.0f), get_phi(local_view_direction));
}

#endif
