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
	HIPRT_HOST_DEVICE static float regularize_reflection(const MicrofacetRegularizationSettings& regularization_settings, float initial_roughness, int sample_number)
	{
		if (!regularization_settings.DEBUG_DO_REGULARIZATION || PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_FALSE)
			return initial_roughness;

		float consistent_tau = MicrofacetRegularization::consistent_tau(regularization_settings.tau_0, sample_number + 1);
		float regularized_roughness = sqrtf(sqrtf(1.0f / (consistent_tau * M_PI)));

		return hippt::max(initial_roughness, regularized_roughness);
	}

	HIPRT_HOST_DEVICE static float regularize_refraction(const MicrofacetRegularizationSettings& regularization_settings, float initial_roughness, float eta_i, float eta_t, int sample_number)
	{
		if (!regularization_settings.DEBUG_DO_REGULARIZATION || PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_FALSE)
			return initial_roughness;

		// We're missing an inverse here
		float consistent_tau = MicrofacetRegularization::consistent_tau(regularization_settings.tau_0, sample_number + 1);
		float regularized_roughness = sqrtf(sqrtf(1.0f / (consistent_tau * M_PI * hippt::square(eta_i - eta_t) / (4.0f * hippt::square(hippt::max(eta_i, eta_t))))));

		return hippt::max(initial_roughness, regularized_roughness);
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
		return 1.0f / (2.0f * M_PI * (1.0f - cosf(atanf(powf(sample_number, -1.0f / 6.0f) * sqrt(M_FOUR_PI * tau_0 - 1.0f) / (M_TWO_PI * tau_0 - 1.0f)))));
	}
};

#endif
