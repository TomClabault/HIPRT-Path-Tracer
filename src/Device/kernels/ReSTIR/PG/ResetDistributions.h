/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_KERNELS_RESTIR_PG_RESET_DISTRIBUTIONS_H
#define DEVICE_KERNELS_RESTIR_PG_RESET_DISTRIBUTIONS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRPGOptions.h"
#include "HostDeviceCommon/RenderData.h"

// Reference: https://codesandbox.io/p/sandbox/fibonacci-sphere-forked-yhvclc?file=%2Fsrc%2Findex.ts%3A19%2C1-36%2C1
HIPRT_DEVICE float3_t fibonacci_sphere(int i, int N)
{
	float golden_angle = hippt::M_Pi * (1.0f + hippt::sqrt(5.0f));
	// http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
	constexpr float epsilon = 0.36f;

	const float y	  = (1.0f - ((i + epsilon) / (N - 1.0f + 2.0f * epsilon)) * 2.0f);
	const float r	  = hippt::sqrt(1.0f - y * y);
	const float theta = golden_angle * i;
	const float x	  = hippt::intrin_cosf(theta) * r;
	const float z	  = hippt::intrin_sinf(theta) * r;

	return hippt::normalize(make_float3(x, y, z));
}

/**
 * Kernel dispatched in 1D with 1 thread per cell per VMF mixture component
 */
#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) ReSTIR_PG_ResetDistributions(HIPRTRenderData render_data)
#else
GLOBAL_KERNEL_SIGNATURE(void) inline ReSTIR_PG_ResetDistributions(HIPRTRenderData render_data, int x)
#endif
{
#ifdef __KERNELCC__
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
#endif

	unsigned int cell_index		 = x / ReSTIRPGDistributionComponentCount;
	unsigned int component_index = x % ReSTIRPGDistributionComponentCount;

	if (cell_index >= render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells)
		return;

	ReSTIRPGDistribution& distribution = render_data.render_settings.restir_pg_settings.hash_grid_distributions[cell_index];

	distribution.distribution_components[component_index].vmf.axis		= fibonacci_sphere(component_index, ReSTIRPGDistributionComponentCount);
	distribution.distribution_components[component_index].vmf.sharpness = 50.0f;
	distribution.distribution_components[component_index].weight		= 1.0f / (float)ReSTIRPGDistributionComponentCount;
}

#endif
