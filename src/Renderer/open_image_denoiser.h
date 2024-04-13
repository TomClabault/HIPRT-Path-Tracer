/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OPEN_IMAGE_DENOISER
#define OPEN_IMAGE_DENOISER

#include "HostDeviceCommon/color.h"
#include <OpenImageDenoise/oidn.hpp>
#include <vector>

class OpenImageDenoiser
{
public:
	OpenImageDenoiser();

	void set_buffers(Color* color_buffer, int width, int height);
	void set_buffers(Color* color_buffer, hiprtFloat3* normals_buffer, int width, int height);
	void set_buffers(Color* color_buffer, Color* albedo_buffer, int width, int height);
	void set_buffers(Color* color_buffer, hiprtFloat3* normals_buffer, Color* albedo_buffer, int width, int height);

	bool* get_denoise_albedo_var();
	bool* get_denoise_normals_var();

	std::vector<Color> get_denoised_data();
	void* get_denoised_data_pointer();
	void* get_denoised_normals_pointer();
	void* get_denoised_albedo_pointer();

	void denoise();
	void denoise_albedo();
	void denoise_normals();

private:
	void set_buffers(Color* color_buffer, int width, int height, bool override_use_normals, bool override_use_albedo);

	void create_beauty_filter();
	void create_AOV_filters();

	int m_width = 0, m_height = 0;

	bool m_use_albedo = false;
	bool m_use_normals = false;
	bool m_denoise_albedo = true;
	bool m_denoise_normals = true;

	oidn::DeviceRef m_device;
	oidn::BufferRef m_denoised_color_buffer;
	oidn::BufferRef m_denoised_albedo_buffer;
	oidn::BufferRef m_denoised_normals_buffer;

	Color* m_color_buffer = nullptr;
	hiprtFloat3* m_normals_buffer = nullptr;
	Color* m_albedo_buffer = nullptr;

	oidn::FilterRef m_beauty_filter;
	oidn::FilterRef m_albedo_filter;
	oidn::FilterRef m_normals_filter;
};

#endif
