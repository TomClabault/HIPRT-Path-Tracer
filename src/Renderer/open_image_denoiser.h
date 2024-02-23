#ifndef OPEN_IMAGE_DENOISER
#define OPEN_IMAGE_DENOISER

#include "Kernels/includes/hiprt_color.h"
#include <OpenImageDenoise/oidn.hpp>
#include <vector>

class OpenImageDenoiser
{
public:
	OpenImageDenoiser();

	void resize_buffers(int new_width, int new_height);
	std::vector<float> denoise(int width, int height, const std::vector<HIPRTColor>& to_denoise);

private:
	oidn::DeviceRef m_device;
	oidn::BufferRef m_color_buffer;
	oidn::FilterRef m_filter;
};

#endif