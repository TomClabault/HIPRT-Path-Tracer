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
}

void GPUBaker::bake_ggx_hemispherical_albedo(const GGXHemisphericalAlbedoSettings& bake_settings)
{
	// Starting everything on a thread to avoid blocking to UI (during the compilation
	// of the kernel mainly)
	ThreadManager::start_thread("bake_ggx_hemispherical_albedo_bake_start", [this, &bake_settings] {
		std::string kernel_filepath = DEVICE_KERNELS_DIRECTORY "/Baking/GGXHemisphericalAlbedo.h";
		std::string kernel_function = "GGXHemisphericalAlbedoBake";

		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

		if (!m_ggx_hemispherical_albedo_bake_kernel.has_been_compiled())
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Compiling GGX hemispherical directional albedo kernel...");

			// Taking the priority for the compilation as otherwise, the kernels
			// precompiling in the background are going to have the hand and we'll
			// never be able to compile our bake kernel and we'll never start baking
			// (until all kernels are precompiled in background of course but that's
			// going to take a long time)
			m_renderer->take_kernel_compilation_priority();
			m_ggx_hemispherical_albedo_bake_kernel = GPUKernel(kernel_filepath, kernel_function);
			m_ggx_hemispherical_albedo_bake_kernel.compile(m_renderer->get_hiprt_orochi_ctx());
			m_renderer->release_kernel_compilation_priority();
		}

		m_ggx_hemispherical_albedo_bake_buffer.resize(bake_settings.texture_size * bake_settings.texture_size);

		// Forcefullt dropping the const because launch_args wants non-const arguments even though
		// they are never going to be modified
		GGXHemisphericalAlbedoSettings& bake_settings_non_const = const_cast<GGXHemisphericalAlbedoSettings&>(bake_settings);
		HIPRTRenderData& render_data = m_renderer->get_render_data();
		float* device_buffer = m_ggx_hemispherical_albedo_bake_buffer.get_device_pointer();

		void* launch_args[] = { &render_data, &bake_settings_non_const, &device_buffer };

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Launching GGX hemispherical directional albedo baking...");
		m_ggx_hemispherical_albedo_bake_kernel.launch_asynchronous(8, 8, bake_settings.texture_size, bake_settings.texture_size, launch_args, m_bake_stream);
		m_last_ggx_hemi_albedo_bake_settings = bake_settings;

		// Callback for setting the baker ready when the baking is done

		oroLaunchHostFunc(m_bake_stream, [](void* userdata) {
			bool* bake_complete = reinterpret_cast<bool*>(userdata);
			*bake_complete = true;

			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "GGX hemispherical directional albedo bake completed!");
		}, &m_ggx_hemispherical_albedo_bake_complete);
	});

	ThreadManager::detach_threads("bake_ggx_hemispherical_albedo_bake_start");
}

bool GPUBaker::is_ggx_hemispherical_albedo_bake_complete() const
{
	return m_ggx_hemispherical_albedo_bake_complete;
}

Image32Bit GPUBaker::get_bake_ggx_hemispherical_albedo_result()
{
	m_ggx_hemispherical_albedo_bake_complete = false;

	std::vector<float> baked_data = m_ggx_hemispherical_albedo_bake_buffer.download_data();

	return Image32Bit(baked_data, m_last_ggx_hemi_albedo_bake_settings.texture_size, m_last_ggx_hemi_albedo_bake_settings.texture_size, 1);
}
