/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDE_MICROFACET_REGULARIZATION_H
#define DEVICE_INCLUDE_MICROFACET_REGULARIZATION_H

#include "HostDeviceCommon/MicrofacetRegularizationSettings.h"

struct MicrofacetRegularization
{
	HIPRT_HOST_DEVICE static float regularize_reflection(const MicrofacetRegularizationSettings& regularization_settings, float initial_roughness)
	{
		if (!regularization_settings.DEBUG_DO_REGULARIZATION || PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_FALSE)
			return initial_roughness;

		float regularized_roughness = sqrtf(sqrtf(1.0f / (regularization_settings.tau * M_PI)));

		return hippt::max(initial_roughness, regularized_roughness);
	}

	HIPRT_HOST_DEVICE static float regularize_refraction(const MicrofacetRegularizationSettings& regularization_settings, float initial_roughness, float eta_i, float eta_t)
	{
		if (!regularization_settings.DEBUG_DO_REGULARIZATION || PrincipledBSDFDoMicrofacetRegularization == KERNEL_OPTION_FALSE)
			return initial_roughness;

		// We're missing an inverse here
		float regularized_roughness = sqrtf(sqrtf(1.0f / (regularization_settings.tau * M_PI * hippt::square(eta_i - eta_t) / (4.0f * hippt::square(hippt::max(eta_i, eta_t))))));

		return hippt::max(initial_roughness, regularized_roughness);
	}
};

#endif
