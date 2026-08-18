/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_NISML_MEGA_KERNEL_INFERENCE_H
#define KERNELS_NISML_MEGA_KERNEL_INFERENCE_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/LightSampling/NISML/NISML.h"
#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"
#include "HostDeviceCommon/RenderData.h"

#ifdef __KERNELCC__
// HIP does not support dynamic initialization of device pointers in constant memory, so keep the uploaded pointer as raw bytes.
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char NISML_MEGAKERNEL_INFERENCE_RENDER_DATA[sizeof(HIPRTRenderData*)];
}
GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(NeuralImportanceSamplingMLP::BLOCK_SIZE) NISMLMegaKernelInference()
#else
GLOBAL_KERNEL_SIGNATURE(void) inline NISMLMegaKernelInference(HIPRTRenderData render_data, int x, int y)
#endif
{
#ifdef __KERNELCC__
	HIPRTRenderData* render_data_pointer = *reinterpret_cast<HIPRTRenderData**>(NISML_MEGAKERNEL_INFERENCE_RENDER_DATA);
	HIPRTRenderData& render_data		 = *render_data_pointer;
	unsigned int query_index			 = blockIdx.x * blockDim.x + threadIdx.x;
#else
	unsigned int query_index = static_cast<unsigned int>(x + y * render_data.render_settings.render_resolution.x);
#endif

	unsigned int query_count = hippt::atomic_fetch_add(render_data.nisml_mega_kernel.query_count, 0u);
	bool valid_query		 = query_index < query_count;
	NISQuery query			 = {};
	if (valid_query)
		query = render_data.nisml_mega_kernel.queries[query_index];

	__shared__ fp16 activations_buffer[NeuralImportanceSamplingMLP::ACTIVATION_WIDTH * 2][NeuralImportanceSamplingMLP::BLOCK_SIZE];
	if (valid_query)
		load_nisml_input_wmma(render_data.nisml.position_learnable_dense_grid, render_data.world_settings.scene_min, render_data.world_settings.scene_max,
							  query.position, query.outgoing_direction, query.normal, &activations_buffer[0][0], threadIdx.x);
	else
		for (unsigned int input_index = 0; input_index < NeuralImportanceSamplingMLP::INPUT_SIZE_PADDED_WMMA; input_index++)
			activations_buffer[input_index][threadIdx.x] = static_cast<fp16>(0.0f);

	__syncthreads();
	render_data.nisml.mlp.inference_wmma(activations_buffer);

	if (!valid_query)
		return;

	unsigned int output_activation_offset = (NeuralImportanceSamplingMLP::LAYER_COUNT - 1) & 1 ? NeuralImportanceSamplingMLP::ACTIVATION_WIDTH : 0;
	// WMMA stores the output as [cluster][query lane], so every query needs its complete residual vector.
	for (unsigned int cluster_index = 0; cluster_index < NISML_MAX_CLUSTER_COUNT; cluster_index++)
		render_data.nisml_mega_kernel.residuals[query_index * render_data.nisml_mega_kernel.residual_stride + cluster_index] =
			static_cast<float>(activations_buffer[output_activation_offset + cluster_index][threadIdx.x]);
}

#endif
