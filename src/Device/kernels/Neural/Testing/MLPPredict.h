/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_MLP_PREDICT_H
#define KERNELS_MLP_PREDICT_H

#include "Device/includes/FixIntellisense.h"
#include "Device/includes/Neural/MLPDevice.h"

GLOBAL_KERNEL_SIGNATURE(void) MLPPredict(MLPDevice mlp, unsigned char* out_predicted_texture, unsigned int width, unsigned int height)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	MLPDevice::InputLayer input	  = { { static_cast<float>(x) / static_cast<float>(width - 1), static_cast<float>(y) / static_cast<float>(height - 1) } };
	MLPDevice::OutputLayer output = mlp.inference(input);

	float r = hippt::clamp(0.0f, 1.0f, output.output[0]);
	float g = hippt::clamp(0.0f, 1.0f, output.output[1]);
	float b = hippt::clamp(0.0f, 1.0f, output.output[2]);

	unsigned int pixel_index = x + y * width;

	out_predicted_texture[pixel_index * 3 + 0] = static_cast<unsigned char>(r * 255.0f);
	out_predicted_texture[pixel_index * 3 + 1] = static_cast<unsigned char>(g * 255.0f);
	out_predicted_texture[pixel_index * 3 + 2] = static_cast<unsigned char>(b * 255.0f);
}

#endif
