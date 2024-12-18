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
#include "Renderer/Baker/GGXConductorDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GGXFresnelDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GGXGlassDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GGXThinGlassDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GPUBakerKernel.h"
#include "Renderer/GPURenderer.h"

#include <mutex>

class GPUBaker
{
public:
	GPUBaker(std::shared_ptr<GPURenderer> renderer);

	void bake_ggx_conductor_directional_albedo(const GGXConductorDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_conductor_directional_albedo_bake_complete() const;

	void bake_ggx_fresnel_directional_albedo(const GGXFresnelDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_fresnel_directional_albedo_bake_complete() const;

	void bake_glossy_dielectric_directional_albedo(const GlossyDielectricDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_glossy_dielectric_directional_albedo_bake_complete() const;

	void bake_ggx_glass_directional_albedo(const GGXGlassDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_glass_directional_albedo_bake_complete() const;
	
	void bake_ggx_thin_glass_directional_albedo(const GGXThinGlassDirectionalAlbedoSettings& bake_settings, const std::string& output_filename);
	bool is_ggx_thin_glass_directional_albedo_bake_complete() const;

private:
	std::shared_ptr<GPURenderer> m_renderer = nullptr;

	oroStream_t m_bake_stream;
	// Mutex so that if we're baking multiple textures at the same time,
	// we don't run into issue with the compilers wanting to take the priority
	// (over background compiling kerneks) at the same time
	std::shared_ptr<std::mutex> m_compiler_priority_mutex;

	GPUBakerKernel m_ggx_conductor_directional_albedo_bake_kernel;
	GPUBakerKernel m_ggx_fresnel_directional_albedo_bake_kernel;
	GPUBakerKernel m_glossy_dielectric_directional_albedo_bake_kernel;
	GPUBakerKernel m_ggx_glass_entering_directional_albedo_bake_kernel;
	GPUBakerKernel m_ggx_glass_exiting_directional_albedo_bake_kernel;
	GPUBakerKernel m_ggx_thin_glass_directional_albedo_bake_kernel;
};

#endif
