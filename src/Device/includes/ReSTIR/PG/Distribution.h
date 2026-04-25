/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_RESTIR_PG_DISTRIBUTION_H
#define DEVICE_INCLUDES_RESTIR_PG_DISTRIBUTION_H

#include "Device/includes/VMF.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPGOptions.h"

struct ReSTIRPGDistribution
{
	HIPRT_DEVICE float3_t sample(Xorshift32Generator& random_number_generator) const
	{
		// Use WRS for sampling a component according to the component weights, then sample the selected component distribution
		float cdf		   = 0.0f;
		float random_value = random_number_generator();

		for (int i = 0; i < ReSTIRPGDistributionComponentCount; ++i)
		{
			cdf += distribution_components[i].weight;
			if (random_value < cdf)
				return distribution_components[i].vmf.sample(random_number_generator);
		}

		// Should never happen, this would mean that the CDF does not sum to 1.0f
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	HIPRT_DEVICE float pdf(float3_t direction) const
	{
		float pdf = 0.0f;

		for (int i = 0; i < ReSTIRPGDistributionComponentCount; ++i)
			pdf += distribution_components[i].weight * distribution_components[i].vmf.density_evaluation(direction);

		return pdf;
	}

	HIPRT_DEVICE bool is_valid() const
	{
		float sum_of_weights = 0.0f;
		for (int i = 0; i < ReSTIRPGDistributionComponentCount; i++)
			sum_of_weights += distribution_components[i].weight;

		return hippt::abs(sum_of_weights - 1.0f) < 1.0e-2f;
	}

	HIPRT_DEVICE bool is_finite() const
	{
		for (int i = 0; i < ReSTIRPGDistributionComponentCount; i++)
		{
			if (!hippt::is_finite(distribution_components[i].weight) || !hippt::is_finite(distribution_components[i].vmf.axis.x) ||
				!hippt::is_finite(distribution_components[i].vmf.axis.y) || !hippt::is_finite(distribution_components[i].vmf.axis.z) ||
				!hippt::is_finite(distribution_components[i].vmf.sharpness))
				return false;
		}

		return true;
	}

	VMFMixtureComponent distribution_components[ReSTIRPGDistributionComponentCount];
};

#endif
