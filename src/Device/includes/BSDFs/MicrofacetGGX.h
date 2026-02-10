/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDF_MICROFACET_GGX_H
#define DEVICE_INCLUDES_BSDF_MICROFACET_GGX_H

#include "Device/includes/BSDFs/MicrofacetCommon.h"

#include "HostDeviceCommon/Xorshift.h"

/**
 * Reference: [Sampling the GGX Distribution of Visible Normals, Unity: Heitz ; 2018]
 */
HIPRT_DEVICE static float3 GGX_VNDF_sample(const float3 local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
	float r1 = random_number_generator();
	float r2 = random_number_generator();

	// Stretching the ellipsoid to the hemisphere configuration
	float3 Vh = hippt::normalize(float3{ alpha_x * local_view_direction.x, alpha_y * local_view_direction.y, local_view_direction.z });

	// Orthonormal basis construction
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 T1	= lensq > 0.0f ? float3{ -Vh.y, Vh.x, 0 } / hippt::sqrt(lensq) : float3{ 1.0f, 0.0f, 0.0f };
	float3 T2	= hippt::cross(Vh, T1);

	// Parametrization of the projected area of the hemisphere
	float r	  = hippt::sqrt(r1);
	float phi = hippt::M_TWO_PI * r2;
	float t1  = r * hippt::intrin_cosf(phi);
	float t2  = r * hippt::intrin_sinf(phi);
	float s	  = 0.5f * (1.0f + Vh.z);
	t2		  = (1.0f - s) * hippt::sqrt(1.0f - t1 * t1) + s * t2;

	// Sampling the hemisphere
	float3 Nh = t1 * T1 + t2 * T2 + hippt::sqrt(hippt::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

	// Un-stretching back to our ellipsoid
	return hippt::normalize(float3{ alpha_x * Nh.x, alpha_y * Nh.y, hippt::max(0.0f, Nh.z) });
}

/**
 * Sample the distribution anisotropic GGX of visible normals using
 * the spherical caps formulation which is slightly faster than the traditional
 * VNDF sampling by Heitz 2018.
 *
 * Reference: [Sampling Visible GGX Normals with Spherical Caps, Dupuy, Benyoub, 2023]
 */
HIPRT_DEVICE static float3 GGX_VNDF_spherical_caps_sample(const float3 local_view_direction,
														  float alpha_x,
														  float alpha_y,
														  Xorshift32Generator& random_number_generator)
{
	float r1 = random_number_generator();
	float r2 = random_number_generator();

	// Stretching the ellipsoid to the hemisphere configuration
	float3 Vh = hippt::normalize(make_float3(alpha_x * local_view_direction.x, alpha_y * local_view_direction.y, local_view_direction.z));

	// Sample a spherical cap in (-wi.z, 1]
	float phi	   = hippt::M_TWO_PI * r1;
	float z		   = (1.0f - r2) * (1.0f + Vh.z) - Vh.z;
	float sinTheta = hippt::sqrt(hippt::clamp(0.0f, 1.0f, 1.0f - z * z));
	float x		   = sinTheta * hippt::intrin_cosf(phi);
	float y		   = sinTheta * hippt::intrin_sinf(phi);
	float3 c	   = make_float3(x, y, z);

	// Compute microfacet normal
	float3 Nh = c + Vh;

	// Un-stretching back to our ellipsoid
	return hippt::normalize(make_float3(alpha_x * Nh.x, alpha_y * Nh.y, Nh.z));
}

/**
 * Samples a microfacet normal from the distribution of visible normals of
 * the GGX normal function distribution
 */
HIPRT_DEVICE static float3 GGX_anisotropic_sample_microfacet(const float3& local_view_direction,
															 float roughness,
															 float anisotropy,
															 Xorshift32Generator& random_number_generator)
{
	float alpha_x, alpha_y;
	MaterialUtils::get_alphas(roughness, anisotropy, alpha_x, alpha_y);

	if (alpha_x <= MaterialConstants::ROUGHNESS_CLAMP && alpha_y <= MaterialConstants::ROUGHNESS_CLAMP)
		// For delta GGX distribution, the sampled normal is always the same as the surface normal
		// (so (0, 0, 1) in local space
		//
		// This is basically a small optimization to avoid the whole sampling routine
		return make_float3(0.0f, 0.0f, 1.0f);

#if PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_SAMPLING
	return GGX_VNDF_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_SPHERICAL_CAPS
	return GGX_VNDF_spherical_caps_sample(local_view_direction, alpha_x, alpha_y, random_number_generator);
#elif PrincipledBSDFAnisotropicGGXSampleFunction == GGX_VNDF_BOUNDED
	// TODO
#else
	// Not implemented
	return make_float3(0.0f, 0.0f, 0.0f);
#endif // PrincipledBSDFAnisotropicGGXSampleFunction
}

// Forward declaration
HIPRT_DEVICE float3 microfacet_GGX_multiple_scattering_invariance_sample_reflection(const float3& local_view_direction,
																					float material_roughness,
																					float material_anisotropy,
																					Xorshift32Generator& rng);

/*
 * Samples a microfacet normal from the distribution of visible normals of
 * the GGX normal function distribution and reflects the given view direction
 * about that microfacet normal to produce a 'to_light_direction' in local
 * shading space that is then returned by that function
 */
template <bool multipleScatteringAllowed = true>
HIPRT_DEVICE static float3 microfacet_GGX_sample_reflection(float roughness,
															float anisotropy,
															const float3& local_view_direction,
															Xorshift32Generator& random_number_generator,
															bool flip_view_direction_below_surface = true)
{
	// if constexpr (PrincipledBSDFEnergyCompensationMode == ENERGY_COMPENSATION_MODE_INVARIANCE_CUI && PrincipledBSDFDoEnergyCompensation == KERNEL_OPTION_TRUE
	// && 			  multipleScatteringAllowed)
	//	// The && multipleScatteringAllowed check is used by microfacet_GGX_multiple_scattering_invariance_sample_reflect itself because that function samples
	//	// the VNDF at each bounce in the microsurface. Without this check, we would be sampling the VNDF with multiple scattering infinitely recursively: at
	//	// each bounce in the microsurface, we want to sample the simple VNDF, not the multiple scattering BRDF
	//	return microfacet_GGX_multiple_scattering_invariance_sample_reflection(local_view_direction, roughness, anisotropy, random_number_generator);
	// else
	{
		// The view direction can sometimes be below the shading normal hemisphere
		// because of normal mapping / smooth normals
		float below_normal = (local_view_direction.z < 0.0f && flip_view_direction_below_surface) ? -1.0f : 1.0f;

		float3 microfacet_normal = GGX_anisotropic_sample_microfacet(local_view_direction * below_normal, roughness, anisotropy, random_number_generator);
		float3 sampled_direction = reflect_ray(local_view_direction, microfacet_normal * below_normal);

		// Should already be normalized but float imprecisions...
		return hippt::normalize(sampled_direction);
	}
}

#endif
