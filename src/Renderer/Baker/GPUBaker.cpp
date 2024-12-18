/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/Baker/GPUBaker.h"
#include "Threads/ThreadManager.h"

extern ImGuiLogger g_imgui_logger;

GPUBaker::GPUBaker(std::shared_ptr<GPURenderer> renderer) : m_renderer(renderer) 
{
	OROCHI_CHECK_ERROR(oroStreamCreate(&m_bake_stream));
	m_compiler_priority_mutex = std::make_shared<std::mutex>();

	m_ggx_conductor_directional_albedo_bake_kernel = GPUBakerKernel(m_renderer, m_bake_stream, m_compiler_priority_mutex,
		DEVICE_KERNELS_DIRECTORY "/Baking/GGXConductorDirectionalAlbedo.h", "GGXConductorDirectionalAlbedoBake", "GGX conductor directional albedo");

	m_ggx_fresnel_directional_albedo_bake_kernel = GPUBakerKernel(m_renderer, m_bake_stream, m_compiler_priority_mutex,
		DEVICE_KERNELS_DIRECTORY "/Baking/GGXFresnelDirectionalAlbedo.h", "GGXFresnelDirectionalAlbedoBake", "GGX fresnel directional albedo");

	m_glossy_dielectric_directional_albedo_bake_kernel = GPUBakerKernel(m_renderer, m_bake_stream, m_compiler_priority_mutex,
		DEVICE_KERNELS_DIRECTORY "/Baking/GlossyDielectricDirectionalAlbedo.h", "GlossyDielectricDirectionalAlbedoBake", "dielectric directional albedo");

	m_ggx_glass_entering_directional_albedo_bake_kernel = GPUBakerKernel(m_renderer, m_bake_stream, m_compiler_priority_mutex,
		DEVICE_KERNELS_DIRECTORY "/Baking/GGXGlassDirectionalAlbedo.h", "GGXGlassDirectionalAlbedoBakeEntering", "GGX glass directional albedo 1/2");
	m_ggx_glass_exiting_directional_albedo_bake_kernel = GPUBakerKernel(m_renderer, m_bake_stream, m_compiler_priority_mutex,
		DEVICE_KERNELS_DIRECTORY "/Baking/GGXGlassDirectionalAlbedo.h", "GGXGlassDirectionalAlbedoBakeExiting", "GGX glass directional albedo 2/2");

	m_ggx_thin_glass_directional_albedo_bake_kernel = GPUBakerKernel(m_renderer, m_bake_stream, m_compiler_priority_mutex,
		DEVICE_KERNELS_DIRECTORY "/Baking/GGXThinGlassDirectionalAlbedo.h", "GGXThinGlassDirectionalAlbedoBake", "GGX thin glass directional albedo");
}

void GPUBaker::bake_ggx_conductor_directional_albedo(const GGXConductorDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_ggx_conductor_directional_albedo_bake_kernel.bake_internal(
		make_int3(bake_settings.texture_size_cos_theta, bake_settings.texture_size_roughness, 1),
		&bake_settings, bake_settings.integration_sample_count, output_filename);
}

bool GPUBaker::is_ggx_conductor_directional_albedo_bake_complete() const
{
	return m_ggx_conductor_directional_albedo_bake_kernel.is_complete();
}

void GPUBaker::bake_ggx_fresnel_directional_albedo(const GGXFresnelDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_ggx_fresnel_directional_albedo_bake_kernel.bake_internal(
		make_int3(bake_settings.texture_size_cos_theta, bake_settings.texture_size_roughness, bake_settings.texture_size_ior),
		&bake_settings, bake_settings.integration_sample_count, output_filename);
}

bool GPUBaker::is_ggx_fresnel_directional_albedo_bake_complete() const
{
	return m_ggx_fresnel_directional_albedo_bake_kernel.is_complete();
}

void GPUBaker::bake_glossy_dielectric_directional_albedo(const GlossyDielectricDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_glossy_dielectric_directional_albedo_bake_kernel.bake_internal(
		make_int3(bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, bake_settings.texture_size_ior),
		&bake_settings, bake_settings.integration_sample_count, output_filename);
}

bool GPUBaker::is_glossy_dielectric_directional_albedo_bake_complete() const
{
	return m_glossy_dielectric_directional_albedo_bake_kernel.is_complete();
}

void GPUBaker::bake_ggx_glass_directional_albedo(const GGXGlassDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_ggx_glass_entering_directional_albedo_bake_kernel.bake_internal(
		make_int3(bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, bake_settings.texture_size_ior),
		&bake_settings, bake_settings.integration_sample_count, output_filename);

	m_ggx_glass_exiting_directional_albedo_bake_kernel.bake_internal(
		make_int3(bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, bake_settings.texture_size_ior),
		&bake_settings, bake_settings.integration_sample_count, "inv_" + output_filename);
}

bool GPUBaker::is_ggx_glass_directional_albedo_bake_complete() const
{
	return m_ggx_glass_entering_directional_albedo_bake_kernel.is_complete() && m_ggx_glass_exiting_directional_albedo_bake_kernel.is_complete();
}

void GPUBaker::bake_ggx_thin_glass_directional_albedo(const GGXThinGlassDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_ggx_thin_glass_directional_albedo_bake_kernel.bake_internal(
		make_int3(bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, bake_settings.texture_size_ior),
		&bake_settings, bake_settings.integration_sample_count, output_filename);
}

bool GPUBaker::is_ggx_thin_glass_directional_albedo_bake_complete() const
{
	return m_ggx_thin_glass_directional_albedo_bake_kernel.is_complete();
}
