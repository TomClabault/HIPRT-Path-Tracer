/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDE_MICROFACET_REGULARIZATION_H
#define DEVICE_INCLUDE_MICROFACET_REGULARIZATION_H

#include "HostDeviceCommon/KernelOptions/PrincipledBSDFKernelOptions.h"
#include "HostDeviceCommon/Math.h"
#include "HostDeviceCommon/MicrofacetRegularizationSettings.h"

struct MicrofacetRegularization
{
	enum class RegularizationMode : unsigned char
	{
		NO_REGULARIZATION = 0,
		REGULARIZATION_CLASSIC = 1, // Should be used when the regularized BSDF PDF isn't going to be used in a MIS weight
		REGULARIZATION_MIS = 2, // Should be used when the regularized BSDF PDF ** is ** going to be used in a MIS weight or if this is for evaluating a BSDF whose sample comes from MIS sampling
	};

	HIPRT_HOST_DEVICE static float regularize_reflection(const MicrofacetRegularizationSettings& regularization_settings, RegularizationMode regularization_mode, float initial_roughness, float accumulated_path_roughness, int sample_number)
	{
#if PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_FALSE
		return initial_roughness;
#endif

		if (regularization_mode == RegularizationMode::NO_REGULARIZATION)
			return initial_roughness;

		float consistent_tau = MicrofacetRegularization::consistent_tau(regularization_settings.tau_0, sample_number);
		// Note that the diffusion heuristic that we're using here is not the one proposed in the paper
		// because the one of the paper requires the mean curvature of the surface and this requires additional
		// per vertex data to be computed... Sounds a bit heavy just for path regularization
		//
		// So instead, we're just using the maximum roughness found on the path so far (which is
		// 'accumulated_path_roughness') to decide whether or not we should use a strong regularization
		// or not.
		//
		// Caustics only happen on diffuse surfaces (roughness 1). So for such a surface, tau should be
		// unchanged i.e., we use the full regularization.
		// 
		// But for smooth surfaces (mirrors, clear glass), we shouldn't regularize anything to keep the sharpness
		// of the glossy reflections.
		//
		// By dividing by a roughness close to 0, tau skyrockets and regularization is essentially disabled 
		float path_diffusion_tau = consistent_tau / hippt::max(hippt::square(accumulated_path_roughness), 1.0e-7f);

#if PrincipledBSDFMicrofacetRegularizationDiffusionHeuristic == KERNEL_OPTION_TRUE
		float final_tau = path_diffusion_tau;
#else
		float final_tau = consistent_tau;
#endif

		float regularized_roughness = sqrtf(sqrtf(1.0f / (final_tau * M_PI)));

		return hippt::max(regularization_settings.min_roughness, hippt::max(initial_roughness, regularized_roughness));
	}

	HIPRT_HOST_DEVICE static float regularize_refraction(const MicrofacetRegularizationSettings& regularization_settings, RegularizationMode regularization_mode, float initial_roughness, float accumulated_path_roughness, float eta_i, float eta_t, int sample_number)
	{
#if PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_FALSE
		return initial_roughness;
#endif

		if (regularization_mode == RegularizationMode::NO_REGULARIZATION)
			return initial_roughness;

		float consistent_tau = MicrofacetRegularization::consistent_tau(regularization_settings.tau_0, sample_number + 1);
		// Note that the diffusion heuristic that we're using here is not the one proposed in the paper
		// because the one of the paper requires the mean curvature of the surface and this requires additional
		// per vertex data to be computed... Sounds a bit heavy just for path regularization
		//
		// So instead, we're just using the maximum roughness found on the path so far (which is
		// 'accumulated_path_roughness') to decide whether or not we should use a strong regularization
		// or not.
		//
		// Caustics only happen on diffuse surfaces (roughness 1). So for such a surface, tau should be
		// unchanged i.e., we use the full regularization.
		// 
		// But for smooth surfaces (mirrors, clear glass), we shouldn't regularize anything to keep the sharpness
		// of the glossy reflections.
		//
		// By dividing by a roughness close to 0, tau skyrockets and regularization is essentially disabled
		float path_diffusion_tau = consistent_tau / hippt::max(hippt::square(accumulated_path_roughness), 1.0e-7f);

#if PrincipledBSDFMicrofacetRegularizationDiffusionHeuristic == KERNEL_OPTION_TRUE
		float final_tau = path_diffusion_tau;
#else
		float final_tau = consistent_tau;
#endif

		float regularized_roughness = sqrtf(sqrtf(1.0f / (final_tau * M_PI * hippt::square(eta_i - eta_t) / (4.0f * hippt::square(hippt::max(eta_i, eta_t))))));

		return hippt::max(regularization_settings.min_roughness, hippt::max(initial_roughness, regularized_roughness));
	}

	HIPRT_HOST_DEVICE static float regularize_mix_reflection_refraction(const MicrofacetRegularizationSettings& regularization_settings, RegularizationMode regularization_mode, float initial_roughness, float accumulated_path_roughness, float eta_i, float eta_t, int sample_number)
	{
#if PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_FALSE
		return initial_roughness;
#endif

		if (regularization_mode == RegularizationMode::NO_REGULARIZATION)
			return initial_roughness;

		float consistent_tau = MicrofacetRegularization::consistent_tau(regularization_settings.tau_0, sample_number);
		// Note that the diffusion heuristic that we're using here is not the one proposed in the paper
		// because the one of the paper requires the mean curvature of the surface and this requires additional
		// per vertex data to be computed... Sounds a bit heavy just for path regularization
		//
		// So instead, we're just using the maximum roughness found on the path so far (which is
		// 'accumulated_path_roughness') to decide whether or not we should use a strong regularization
		// or not.
		//
		// Caustics only happen on diffuse surfaces (roughness 1). So for such a surface, tau should be
		// unchanged i.e., we use the full regularization.
		// 
		// But for smooth surfaces (mirrors, clear glass), we shouldn't regularize anything to keep the sharpness
		// of the glossy reflections.
		//
		// By dividing by a roughness close to 0, tau skyrockets and regularization is essentially disabled
		float path_diffusion_tau = consistent_tau / hippt::max(hippt::square(accumulated_path_roughness), 1.0e-6f);

#if PrincipledBSDFMicrofacetRegularizationDiffusionHeuristic == KERNEL_OPTION_TRUE
		float final_tau = path_diffusion_tau;
#else
		float final_tau = consistent_tau;
#endif

		float regularized_roughness_reflection = sqrtf(sqrtf(1.0f / (final_tau * M_PI)));
		float regularized_roughness_refraction = sqrtf(sqrtf(1.0f / (final_tau * M_PI * hippt::square(eta_i - eta_t) / (4.0f * hippt::square(hippt::max(eta_i, eta_t))))));

		// Mixing both reflection and refraction regularized roughnesses.
		// Refraction regularization tends to be stronger (higher resulting roughness).
		//
		// We're biasing (75%) towards refraction to bias towards higher regularization to conservatively
		// reduce variance
		return hippt::max(regularization_settings.min_roughness, hippt::max(initial_roughness, regularized_roughness_refraction * 0.75f + regularized_roughness_reflection * 0.25f));
	}

	/**
	 * 'sample_number" should be >= 1
	 */
	HIPRT_HOST_DEVICE static float consistent_tau(float tau_0, int sample_number)
	{
#if PrincipledBSDFDoMicrofacetRegularizationConsistentParameterization == KERNEL_OPTION_FALSE
		return tau_0;
#endif

		// Eq. 16 of the paper
		return 1.0f / (2.0f * M_PI * (1.0f - cosf(atanf(powf(sample_number + 1, -1.0f / 6.0f) * sqrt(M_FOUR_PI * tau_0 - 1.0f) / (M_TWO_PI * tau_0 - 1.0f)))));
	}
};

#endif
