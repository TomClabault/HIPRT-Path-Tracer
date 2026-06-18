/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RGBE9995_ENVMAP_H
#define DEVICE_RGBE9995_ENVMAP_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/Packing.h"
#include "Image/Image.h"

/**
 * If GPU is true, then functions of this class will be templated such
 * that they compute / return data that can be used on the GPU
 */
template <bool GPU>
class RGBE9995Envmap
{
public:
	HIPRT_HOST void pack_from(const Image32Bit& image, float& out_scaling_factor)
	{
		bool warning_emitted = false;

		packed_data_CPU.resize(image.width * image.height);

		// Computing the maximum component of any texel of the envmap
		float max_component = 0.0f;
		for (int y = 0; y < image.height; y++)
		{
			for (int x = 0; x < image.width; x++)
			{
				int index = x + y * image.width;

				max_component = hippt::max(max_component, image.get_pixel_ColorRGB32F(index).max_component());
			}
		}

		// We're going to scale every texel of the envmap by a factor such that the maximum component of any texel is 65535.0f. This is to avoid clamping of the
		// envmap when the maximum component of any texel is greater than 65535.0f, which is the maximum value that can be represented by the RGBE9995 format.
		// Clamping would introduce a loss in dynamic range but scaling doesn't
		//
		// The scaling factor is also outputted so that the user can scale the envmap back to its original range when using it in the renderer
		out_scaling_factor = hippt::max(1.0f, max_component / 65535.0f);

#pragma omp parallel for
		for (int y = 0; y < image.height; y++)
		{
			for (int x = 0; x < image.width; x++)
			{
				int index = x + y * image.width;

				packed_data_CPU[index].pack(image.get_pixel_ColorRGB32F(index) / out_scaling_factor);
			}
		}

		if (GPU)
		{
			// If the data is for the GPU, upload the data to the GPU buffer and then discard the CPU data

			packed_data_GPU.resize(image.width * image.height);
			packed_data_GPU.upload_data(packed_data_CPU);

			// Clearing the CPU data
			packed_data_CPU = std::vector<RGBE9995Packed>();
		}
	}

	HIPRT_HOST RGBE9995Packed* get_data_pointer()
	{
		if (GPU)
			return packed_data_GPU.get_device_pointer();
		else
			return packed_data_CPU.data();
	}

private:
	// Linear array for the packed data of the envmap
	OrochiBuffer<RGBE9995Packed> packed_data_GPU;

	std::vector<RGBE9995Packed> packed_data_CPU;
};

#endif
