/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_NEURAL_INPUT_ENCODINGS_H
#define DEVICE_INCLUDES_NEURAL_INPUT_ENCODINGS_H

#include "Device/includes/FixIntellisense.h"
#include "HostDeviceCommon/Maths/Math.h"

/**
 * Encodes each raw input feature with frequency-major sine/cosine pairs.
 *
 * The encoded order is [sin(1*x*pi), cos(1*x*pi), sin(2*x*pi), cos(2*x*pi), ...]
 * for each raw feature before continuing with the next feature.
 */
template <unsigned int InputSize, unsigned int FrequencyCount>
HIPRT_DEVICE void encode_frequency_input(const float* input, float* encoded_input)
{
	for (unsigned int input_raw_index = 0; input_raw_index < InputSize; input_raw_index++)
	{
		float input_value = input[input_raw_index];

		for (unsigned int frequency_index = 0; frequency_index < FrequencyCount; frequency_index++)
		{
			float frequency			   = static_cast<float>(1 << frequency_index);
			unsigned int encoded_index = input_raw_index * FrequencyCount * 2 + frequency_index * 2;

			encoded_input[encoded_index + 0] = hippt::intrin_sinf(frequency * input_value * hippt::M_Pi);
			encoded_input[encoded_index + 1] = hippt::intrin_cosf(frequency * input_value * hippt::M_Pi);
		}
	}
}

enum class OneBlobKernel
{
	GAUSSIAN,
	QUARTIC
};

/**
 * Encodes a scalar input as the integral of a kernel over a uniform set of bins.
 *
 * The input is clamped to [0, 1]. The kernel is centered at the input and is not
 * wrapped or renormalized when its support crosses either boundary.
 */
template <unsigned int BinCount, OneBlobKernel Kernel>
HIPRT_DEVICE void encode_one_blob(float input, float* encoded_input)
{
	static_assert(BinCount > 0, "One-blob encoding requires at least one bin");

	input = hippt::clamp(0.0f, 1.0f, input);

	float bin_width = 1.0f / static_cast<float>(BinCount);

	for (unsigned int bin_index = 0; bin_index < BinCount; bin_index++)
	{
		float bin_lower = static_cast<float>(bin_index) * bin_width;
		float bin_upper = static_cast<float>(bin_index + 1) * bin_width;

		if constexpr (Kernel == OneBlobKernel::GAUSSIAN)
		{
			float sigma					 = bin_width;
			float inverse_sqrt_two_sigma = 1.0f / (hippt::sqrt(2.0f) * sigma);
			float lower_argument		 = (bin_lower - input) * inverse_sqrt_two_sigma;
			float upper_argument		 = (bin_upper - input) * inverse_sqrt_two_sigma;

			encoded_input[bin_index] = 0.5f * (erff(upper_argument) - erff(lower_argument));
		}
		else
		{
			float support_radius = bin_width;
			float support_lower	 = input - support_radius;
			float support_upper	 = input + support_radius;

			if (bin_upper <= support_lower || bin_lower >= support_upper)
			{
				encoded_input[bin_index] = 0.0f;
				continue;
			}

			float integration_lower = hippt::max(bin_lower, support_lower);
			float integration_upper = hippt::min(bin_upper, support_upper);
			float lower_normalized	= (integration_lower - input) / support_radius;
			float upper_normalized	= (integration_upper - input) / support_radius;

			float lower_cdf =
				0.5f + (15.0f / 16.0f) * (lower_normalized - (2.0f / 3.0f) * lower_normalized * lower_normalized * lower_normalized +
										  (1.0f / 5.0f) * lower_normalized * lower_normalized * lower_normalized * lower_normalized * lower_normalized);
			float upper_cdf =
				0.5f + (15.0f / 16.0f) * (upper_normalized - (2.0f / 3.0f) * upper_normalized * upper_normalized * upper_normalized +
										  (1.0f / 5.0f) * upper_normalized * upper_normalized * upper_normalized * upper_normalized * upper_normalized);

			encoded_input[bin_index] = upper_cdf - lower_cdf;
		}
	}
}

template <unsigned int BinCount, OneBlobKernel Kernel>
HIPRT_DEVICE void encode_one_blob_wmma(float input, fp16* encoded_input, unsigned int output_offset, unsigned int output_stride, unsigned int thread_index)
{
	static_assert(BinCount > 0, "One-blob encoding requires at least one bin");

	input = hippt::clamp(0.0f, 1.0f, input);

	float bin_width = 1.0f / static_cast<float>(BinCount);

	for (unsigned int bin_index = 0; bin_index < BinCount; bin_index++)
	{
		unsigned int output_index = (output_offset + bin_index) * output_stride + thread_index;

		if constexpr (Kernel == OneBlobKernel::GAUSSIAN)
		{
			float sigma					 = bin_width;
			float inverse_sqrt_two_sigma = 1.0f / (hippt::sqrt(2.0f) * sigma);
			float lower_argument		 = (bin_index * bin_width - input) * inverse_sqrt_two_sigma;
			float upper_argument		 = ((bin_index + 1) * bin_width - input) * inverse_sqrt_two_sigma;

			encoded_input[output_index] = static_cast<fp16>(0.5f * (erff(upper_argument) - erff(lower_argument)));
		}
		else
		{
			float support_radius = bin_width;
			float support_lower	 = input - support_radius;
			float support_upper	 = input + support_radius;

			float bin_lower = static_cast<float>(bin_index) * bin_width;
			float bin_upper = static_cast<float>(bin_index + 1) * bin_width;
			if (bin_upper <= support_lower || bin_lower >= support_upper)
			{
				encoded_input[output_index] = static_cast<fp16>(0.0f);

				continue;
			}

			float integration_lower = hippt::max(bin_lower, support_lower);
			float integration_upper = hippt::min(bin_upper, support_upper);
			float lower_normalized	= (integration_lower - input) / support_radius;
			float upper_normalized	= (integration_upper - input) / support_radius;

			float lower_cdf =
				0.5f + (15.0f / 16.0f) * (lower_normalized - (2.0f / 3.0f) * lower_normalized * lower_normalized * lower_normalized +
										  (1.0f / 5.0f) * lower_normalized * lower_normalized * lower_normalized * lower_normalized * lower_normalized);
			float upper_cdf =
				0.5f + (15.0f / 16.0f) * (upper_normalized - (2.0f / 3.0f) * upper_normalized * upper_normalized * upper_normalized +
										  (1.0f / 5.0f) * upper_normalized * upper_normalized * upper_normalized * upper_normalized * upper_normalized);

			encoded_input[output_index] = static_cast<fp16>(upper_cdf - lower_cdf);
		}
	}
}

/**
 * Real spherical-harmonics encoding, degree 4 = bands 0, 1, 2, 3
 *
 * Input:
 *   direction must be normalized and expressed directly in [-1, 1]^3.
 *
 * Output:
 *   16 SH coefficients, corresponding to bands l = 0..3.
 */
HIPRT_DEVICE static void encode_spherical_harmonics_degree_4(float3_t direction, float* encoded_input)
{
	float x = direction.x;
	float y = direction.y;
	float z = direction.z;

	float x2 = x * x;
	float y2 = y * y;
	float z2 = z * z;

	// l = 0
	encoded_input[0] = 0.28209479177387814f;

	// l = 1
	encoded_input[1] = -0.4886025119029199f * y;
	encoded_input[2] = 0.4886025119029199f * z;
	encoded_input[3] = -0.4886025119029199f * x;

	// l = 2
	encoded_input[4] = 1.0925484305920792f * x * y;
	encoded_input[5] = -1.0925484305920792f * y * z;
	encoded_input[6] = 0.3153915652525200f * (3.0f * z2 - 1.0f);
	encoded_input[7] = -1.0925484305920792f * x * z;
	encoded_input[8] = 0.5462742152960396f * (x2 - y2);

	// l = 3
	encoded_input[9]  = 0.5900435899266435f * y * (y2 - 3.0f * x2);
	encoded_input[10] = 2.890611442640554f * x * y * z;
	encoded_input[11] = 0.4570457994644658f * y * (1.0f - 5.0f * z2);
	encoded_input[12] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f);
	encoded_input[13] = 0.4570457994644658f * x * (1.0f - 5.0f * z2);
	encoded_input[14] = 1.445305721320277f * z * (x2 - y2);
	encoded_input[15] = 0.5900435899266435f * x * (3.0f * y2 - x2);
}

HIPRT_DEVICE static void encode_spherical_harmonics_degree_4_wmma(
	float3_t direction, fp16* encoded_input, unsigned int output_offset, unsigned int output_stride, unsigned int thread_index)
{
	float x = direction.x;
	float y = direction.y;
	float z = direction.z;

	float x2 = x * x;
	float y2 = y * y;
	float z2 = z * z;

	encoded_input[(output_offset + 0) * output_stride + thread_index] = static_cast<fp16>(0.28209479177387814f);

	encoded_input[(output_offset + 1) * output_stride + thread_index] = static_cast<fp16>(-0.4886025119029199f * y);
	encoded_input[(output_offset + 2) * output_stride + thread_index] = static_cast<fp16>(0.4886025119029199f * z);
	encoded_input[(output_offset + 3) * output_stride + thread_index] = static_cast<fp16>(-0.4886025119029199f * x);

	encoded_input[(output_offset + 4) * output_stride + thread_index] = static_cast<fp16>(1.0925484305920792f * x * y);
	encoded_input[(output_offset + 5) * output_stride + thread_index] = static_cast<fp16>(-1.0925484305920792f * y * z);
	encoded_input[(output_offset + 6) * output_stride + thread_index] = static_cast<fp16>(0.3153915652525200f * (3.0f * z2 - 1.0f));
	encoded_input[(output_offset + 7) * output_stride + thread_index] = static_cast<fp16>(-1.0925484305920792f * x * z);
	encoded_input[(output_offset + 8) * output_stride + thread_index] = static_cast<fp16>(0.5462742152960396f * (x2 - y2));

	encoded_input[(output_offset + 9) * output_stride + thread_index]  = static_cast<fp16>(0.5900435899266435f * y * (y2 - 3.0f * x2));
	encoded_input[(output_offset + 10) * output_stride + thread_index] = static_cast<fp16>(2.890611442640554f * x * y * z);
	encoded_input[(output_offset + 11) * output_stride + thread_index] = static_cast<fp16>(0.4570457994644658f * y * (1.0f - 5.0f * z2));
	encoded_input[(output_offset + 12) * output_stride + thread_index] = static_cast<fp16>(0.3731763325901154f * z * (5.0f * z2 - 3.0f));
	encoded_input[(output_offset + 13) * output_stride + thread_index] = static_cast<fp16>(0.4570457994644658f * x * (1.0f - 5.0f * z2));
	encoded_input[(output_offset + 14) * output_stride + thread_index] = static_cast<fp16>(1.445305721320277f * z * (x2 - y2));
	encoded_input[(output_offset + 15) * output_stride + thread_index] = static_cast<fp16>(0.5900435899266435f * x * (3.0f * y2 - x2));
}

#endif
