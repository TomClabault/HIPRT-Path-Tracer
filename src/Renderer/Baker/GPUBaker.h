/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_BAKER_H
#define GPU_BAKER_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Image/Image.h"
#include "Renderer/Baker/GGXHemisphericalAlbedoSettings.h"
#include "Renderer/Baker/GGXGlassHemisphericalAlbedoSettings.h"
#include "Renderer/GPURenderer.h"

#include <mutex>

class GPUBaker
{
public:
	GPUBaker(std::shared_ptr<GPURenderer> renderer);

	void bake_ggx_hemispherical_albedo(const GGXHemisphericalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_hemispherical_albedo_bake_complete() const;

	void bake_ggx_glass_hemispherical_albedo(const GGXGlassHemisphericalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_glass_hemispherical_albedo_bake_complete() const;

private:
	std::shared_ptr<GPURenderer> m_renderer = nullptr;

	oroStream_t m_bake_stream;
	// Mutex so that if we're baking multiple textures at the same time,
	// we don't into issue with the compilers wanting to take the priority
	// (over background compiling kerneks) at the same time
	std::mutex m_compiler_priority_mutex;

	GPUKernel m_ggx_hemispherical_albedo_bake_kernel;
	OrochiBuffer<float> m_ggx_hemispherical_albedo_bake_buffer;
	bool m_ggx_hemispherical_albedo_bake_complete = true;

	// State for baking hemispherical albedo for glass materials
	GPUKernel m_ggx_glass_hemispherical_albedo_bake_kernel;
	OrochiBuffer<float> m_ggx_glass_hemispherical_albedo_bake_buffer;
	OrochiBuffer<float> m_ggx_glass_hemispherical_albedo_bake_buffer_inverse;
	bool m_ggx_glass_hemispherical_albedo_bake_complete = true;
};

#endif
