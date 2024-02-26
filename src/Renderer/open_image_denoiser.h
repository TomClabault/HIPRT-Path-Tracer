#ifndef OPEN_IMAGE_DENOISER
#define OPEN_IMAGE_DENOISER

#include "HostDeviceCommon/color.h"
#include <OpenImageDenoise/oidn.hpp>
#include <vector>

class OpenImageDenoiser
{
public:
	OpenImageDenoiser();

	void resize_buffers(int new_width, int new_height);
	std::vector<float> denoise(int width, int height, const std::vector<float>& to_denoise);
	std::vector<float> denoise(int width, int height, const std::vector<float>& to_denoise, const std::vector<hiprtFloat3>& world_space_normals_aov_buffer, const std::vector<Color>& albedo_aov_buffer);

private:
	oidn::DeviceRef m_device;
	oidn::BufferRef m_color_buffer;
	oidn::BufferRef m_normals_buffer;
	oidn::BufferRef m_albedo_buffer;
	oidn::FilterRef m_filter;
};

#endif