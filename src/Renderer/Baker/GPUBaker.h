/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef GPU_BAKER_H
#define GPU_BAKER_H

#include "Compiler/GPUKernel.h"
#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Image/Image.h"
#include "Renderer/Baker/GlossyDielectricDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GGXDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GGXGlassDirectionalAlbedoSettings.h"
#include "Renderer/GPURenderer.h"

#include <mutex>

class GPUBaker
{
public:
	GPUBaker(std::shared_ptr<GPURenderer> renderer);

	void bake_ggx_directional_albedo(const GGXDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_directional_albedo_bake_complete() const;

	void bake_glossy_dielectric_directional_albedo(const GlossyDielectricDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_glossy_dielectric_directional_albedo_bake_complete() const;

	void bake_ggx_glass_directional_albedo(const GGXGlassDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_glass_directional_albedo_bake_complete() const;

private:
	std::shared_ptr<GPURenderer> m_renderer = nullptr;

	oroStream_t m_bake_stream;
	// Mutex so that if we're baking multiple textures at the same time,
	// we don't into issue with the compilers wanting to take the priority
	// (over background compiling kerneks) at the same time
	std::mutex m_compiler_priority_mutex;

	// State for baking GGX conductors directional albedo
	GPUKernel m_ggx_directional_albedo_bake_kernel;
	OrochiBuffer<float> m_ggx_directional_albedo_bake_buffer;
	bool m_ggx_directional_albedo_bake_complete = true;

	// State for baking GGX specular + diffuse (i.e. glossy material)
	// directional albedo
	GPUKernel m_glossy_dielectric_directional_albedo_bake_kernel;
	OrochiBuffer<float> m_glossy_dielectric_directional_albedo_bake_buffer;
	bool m_glossy_dielectric_directional_albedo_bake_complete = true;

	// State for baking directional albedo for glass materials
	GPUKernel m_ggx_glass_directional_albedo_bake_kernel;
	OrochiBuffer<float> m_ggx_glass_directional_albedo_bake_buffer;
	OrochiBuffer<float> m_ggx_glass_directional_albedo_bake_buffer_inverse;
	bool m_ggx_glass_directional_albedo_bake_complete = true;
};

#endif
