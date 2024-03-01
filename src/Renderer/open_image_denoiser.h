#ifndef OPEN_IMAGE_DENOISER
#define OPEN_IMAGE_DENOISER

#include "HostDeviceCommon/color.h"
#include <OpenImageDenoise/oidn.hpp>
#include <vector>

class OpenImageDenoiser
{
public:
	OpenImageDenoiser() : m_uninitialized(true) {}
	OpenImageDenoiser(Color* color_buffer);
	OpenImageDenoiser(Color* color_buffer, hiprtFloat3* world_space_normals_buffer);
	OpenImageDenoiser(Color* color_buffer, Color* albedo_buffer);
	OpenImageDenoiser(Color* color_buffer, hiprtFloat3* world_space_normals_buffer, Color* albedo_buffer);

	void resize(int new_width, int new_height, Color* color_buffer, hiprtFloat3* normals_buffer, Color* albedo_buffer);
	void create_filters(int width, int height);
	void denoise();
	std::vector<Color> get_denoised_data();
	void* get_denoised_data_pointer();

private:
	bool m_uninitialized = false;
	int m_width = 0, m_height = 0;

	bool m_use_normals = false;
	bool m_use_albedo = false;

	oidn::DeviceRef m_device;
	oidn::BufferRef m_denoised_buffer;

	Color* m_color_buffer = nullptr;
	hiprtFloat3* m_normals_buffer = nullptr;
	Color* m_albedo_buffer = nullptr;

	oidn::FilterRef m_beauty_filter;
	oidn::FilterRef m_albedo_filter;
	oidn::FilterRef m_normals_filter;
};

#endif