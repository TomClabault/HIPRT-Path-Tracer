/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RGBE9995_ENVMAP_H
#define DEVICE_RGBE9995_ENVMAP_H

#include "HostDeviceCommon/Packing.h"
#include "Image/Image.h"
#include "HIPRT-Orochi/OrochiBuffer.h"

 /**
  * If GPU is true, then functions of this class will be templated such
  * that they compute / return data that can be used on the GPU
  */
template <bool GPU>
class RGBE9995Envmap
{
public:
	HIPRT_HOST void pack_from(const Image32Bit& image)
	{
		bool warning_emitted = false;

		packed_data_CPU.resize(image.width * image.height);

#pragma omp parallel for
		for (int y = 0; y < image.height; y++)
		{
			for (int x = 0; x < image.width; x++)
			{
				int index = x + y * image.width;

				if (image.get_pixel_ColorRGB32F(index).max_component() > 65535.0f && !warning_emitted)
				{
					g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_WARNING, "Envmap maximum brightness reached (65535)! Clamping to 65535.");

					warning_emitted = true;
				}
				packed_data_CPU[index].pack(image.get_pixel_ColorRGB32F(index));
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
