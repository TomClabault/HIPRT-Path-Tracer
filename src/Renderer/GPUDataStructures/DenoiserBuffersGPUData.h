/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_RENDERER_DENOISER_BUFFERS_H
#define GPU_RENDERER_DENOISER_BUFFERS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "HostDeviceCommon/Color.h"
#include "OpenGL/OpenGLInteropBuffer.h"
#include "UI/ApplicationSettings.h"

class GPURenderer;

struct DenoiserBuffersGPUData
{
	float3* map_normals_buffer();
	void resize_normals_buffer(size_t new_element_count);
	void unmap_normals_buffer();

	ColorRGB32F* map_albedo_buffer();
	void resize_albedo_buffer(size_t new_element_count);
	void unmap_albedo_buffer();

	void set_use_interop_AOV_buffers(GPURenderer* renderer, bool use_interop);

	// Buffer for holding the denoised frame (the denoiser data will be copied
		// to this buffer and then displayed to the viewport)
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> m_denoised_framebuffer;
	// Normals G-buffer
	std::shared_ptr<OpenGLInteropBuffer<float3>> m_normals_AOV_interop_buffer;
	std::shared_ptr<OrochiBuffer<float3>> m_normals_AOV_no_interop_buffer;
	// Albedo G-buffer
	std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>>m_albedo_AOV_interop_buffer;
	std::shared_ptr<OrochiBuffer<ColorRGB32F>>m_albedo_AOV_no_interop_buffer;

	bool use_interop_AOVs = ApplicationSettings::DENOISER_USE_INTEROP_BUFFERS_DEFAULT;
};

#endif
