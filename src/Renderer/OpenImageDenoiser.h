/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OPEN_IMAGE_DENOISER
#define OPEN_IMAGE_DENOISER

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/Color.h"
#include "OpenGL/OpenGLInteropBuffer.h"

#include <OpenImageDenoise/oidn.hpp>
#include <vector>

class OpenImageDenoiser
{
public:
	OpenImageDenoiser();

	void set_use_albedo(bool use_albedo);
	void set_denoise_albedo(bool denoise_normals_or_not);
	void set_use_normals(bool use_normal);
	void set_denoise_normals(bool denoise_normals_or_not);

	void initialize();

	/**
	 * Resizes the buffers of this denoiser. Don't forget to call finalize() after calling resize()!
	 */
	void resize(int new_width, int new_height);

	/**
	 * Function that finalizes the creation of the internal denoising
	 * filters etc... once everything is setup (set_use_albedo / set_use_normals
	 * have been called if necessary, subsequent buffers have been provided, ...)
	*/
	void finalize();

	/**
	 * Denoises 'data_to_denoise' and uses the AOVs to improve denoising quality if provided
	 * and if normals/albedo denoising is enabled on the denoiser.
	 * 
	 * See set_use_albedo(bool use_albedo), set_denoise_albedo(bool denoise_normals_or_not), set_use_normals(bool use_normal), ...
	 */
	void denoise(std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> data_to_denoise, 
				 std::shared_ptr<OpenGLInteropBuffer<float3>> normals_aov = nullptr, 
				 std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> albedo_aov = nullptr);
	/**
	 * Overload to denoise from non OpenGL Interop AOV buffers
	 */
	void denoise(std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> data_to_denoise,
				 std::shared_ptr<OrochiBuffer<float3>> normals_aov,
				 std::shared_ptr<OrochiBuffer<ColorRGB32F>> albedo_aov);
	/**
	 * Function used to copy the denoiser result after a call to denoise() to a given buffer
	 */
	void copy_denoised_data_to_buffer(std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> out_buffer);
	/**
	 * Overload for non-interop buffers
	 */
	void copy_denoised_data_to_buffer(std::shared_ptr<OrochiBuffer<ColorRGB32F>> out_buffer);


private:
	void create_device();

	bool check_valid_state();
	bool check_device();
	bool check_buffer_sizes();

	// Internal denoise function that takes raw pointers
	void denoise(ColorRGB32F* data_to_denoise, float3* normals_aov, ColorRGB32F*);

	bool m_use_albedo = false;
	bool m_denoise_albedo = true;
	bool m_use_normals = false;
	bool m_denoise_normals = true;

	int m_width, m_height;

	// If true, this means that we couldn't get a device to denoise with
	bool m_denoiser_invalid = false;
	// If true, we're using a CPU device and we're going to have to adapt
	// the way we copy the GPU framebuffer to the OIDN buffers i.e. we're
	// going to have to use memcpyDeviceToHost instead of memcpyDeviceToDevice
	bool m_cpu_device = false;
	oidn::DeviceRef m_device;

	oidn::FilterRef m_beauty_filter = nullptr;
	oidn::FilterRef m_albedo_filter = nullptr;
	oidn::FilterRef m_normals_filter = nullptr;

	oidn::BufferRef m_input_color_buffer_oidn;
	oidn::BufferRef m_normals_buffer_denoised_oidn = nullptr;
	oidn::BufferRef m_albedo_buffer_denoised_oidn = nullptr;
	oidn::BufferRef m_denoised_buffer;
};

#endif
