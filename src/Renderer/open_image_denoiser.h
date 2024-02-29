#ifndef OPEN_IMAGE_DENOISER
#define OPEN_IMAGE_DENOISER

#include "HostDeviceCommon/color.h"
#include <OpenImageDenoise/oidn.hpp>
#include <vector>

class OpenImageDenoiser
{
public:
	// No AOVs
	OpenImageDenoiser();
	OpenImageDenoiser(hiprtFloat3* world_space_normals_buffer);
	OpenImageDenoiser(Color* albedo_buffer);
	OpenImageDenoiser(hiprtFloat3* world_space_normals_buffer, Color* albedo_buffer);

	void resize_buffers(int new_width, int new_height);
	std::vector<float> denoise(int width, int height, const std::vector<float>& to_denoise);

private:
	bool m_use_normals = false;
	bool m_use_albedo = false;

	oidn::DeviceRef m_device;
	Color* m_color_buffer;
	hiprtFloat3* m_normals_buffer = nullptr;
	Color* m_albedo_buffer = nullptr;

	oidn::FilterRef m_beauty_filter;
	oidn::FilterRef m_albedo_filter;
	oidn::FilterRef m_normals_filter;
};

#endif