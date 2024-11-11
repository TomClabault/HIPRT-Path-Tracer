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

void GPUBaker::bake_ggx_directional_albedo(const GGXDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_ggx_directional_albedo_bake_complete = false;

	// Starting everything on a thread to avoid blocking to UI (during the compilation
	// of the kernel mainly)
	ThreadManager::start_thread("bake_ggx_directional_albedo_bake_start", [this, &bake_settings, &output_filename] {
		std::string kernel_filepath = DEVICE_KERNELS_DIRECTORY "/Baking/GGXDirectionalAlbedo.h";
		std::string kernel_function = "GGXDirectionalAlbedoBake";

		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

		if (!m_ggx_directional_albedo_bake_kernel.has_been_compiled())
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Compiling GGX directional albedo kernel...");

			// Taking the priority for the compilation as otherwise, the kernels
			// precompiling in the background are going to have the hand and we'll
			// never be able to compile our bake kernel and we'll never start baking
			// (until all kernels are precompiled in background of course but that's
			// going to take a long time)
			std::lock_guard<std::mutex> lock(m_compiler_priority_mutex);
			m_renderer->take_kernel_compilation_priority();
			m_ggx_directional_albedo_bake_kernel = GPUKernel(kernel_filepath, kernel_function);
			m_ggx_directional_albedo_bake_kernel.compile(m_renderer->get_hiprt_orochi_ctx());
			m_renderer->release_kernel_compilation_priority();
		}

		m_ggx_directional_albedo_bake_buffer.resize(bake_settings.texture_size_cos_theta * bake_settings.texture_size_roughness);

		// Forcefully dropping the const because launch_args wants non-const arguments even though
		// they are never going to be modified
		GGXDirectionalAlbedoSettings& bake_settings_non_const = const_cast<GGXDirectionalAlbedoSettings&>(bake_settings);
		HIPRTRenderData& render_data = m_renderer->get_render_data();
		float* device_buffer = m_ggx_directional_albedo_bake_buffer.get_device_pointer();

		void* launch_args[] = { &render_data, &bake_settings_non_const, &device_buffer };

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Launching GGX directional albedo baking...");
		m_ggx_directional_albedo_bake_kernel.launch_asynchronous(8, 8, bake_settings.texture_size_cos_theta, bake_settings.texture_size_roughness, launch_args, m_bake_stream);

		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_bake_stream));

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "GGX directional albedo bake completed!");

		std::vector<float> baked_data = m_ggx_directional_albedo_bake_buffer.download_data();
		Image32Bit image = Image32Bit(baked_data, bake_settings.texture_size_cos_theta, bake_settings.texture_size_roughness, 1);
		image.write_image_hdr(output_filename.c_str(), false);

		m_ggx_directional_albedo_bake_buffer.free();
		m_ggx_directional_albedo_bake_complete = true;
	});

	ThreadManager::detach_threads("bake_ggx_directional_albedo_bake_start");
}

bool GPUBaker::is_ggx_directional_albedo_bake_complete() const
{
	return m_ggx_directional_albedo_bake_complete;
}

void GPUBaker::bake_glossy_dielectric_directional_albedo(const GlossyDielectricDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_glossy_dielectric_directional_albedo_bake_complete = false;

	// Starting everything on a thread to avoid blocking to UI (during the compilation
	// of the kernel mainly)
	ThreadManager::start_thread("bake_glossy_dielectric_directional_albedo_bake_start", [this, &bake_settings, &output_filename] {
		std::string kernel_filepath = DEVICE_KERNELS_DIRECTORY "/Baking/GlossyDielectricDirectionalAlbedo.h";
		std::string kernel_function = "GlossyDielectricDirectionalAlbedoBake";

		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

		if (!m_glossy_dielectric_directional_albedo_bake_kernel.has_been_compiled())
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Compiling dielectric directional albedo kernel...");

			// Taking the priority for the compilation as otherwise, the kernels
			// precompiling in the background are going to have the hand and we'll
			// never be able to compile our bake kernel and we'll never start baking
			// (until all kernels are precompiled in background of course but that's
			// going to take a long time)
			std::lock_guard<std::mutex> lock(m_compiler_priority_mutex);
			m_renderer->take_kernel_compilation_priority();
			m_glossy_dielectric_directional_albedo_bake_kernel = GPUKernel(kernel_filepath, kernel_function);
			m_glossy_dielectric_directional_albedo_bake_kernel.compile(m_renderer->get_hiprt_orochi_ctx());
			m_renderer->release_kernel_compilation_priority();
		}

		m_glossy_dielectric_directional_albedo_bake_buffer.resize(bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness * bake_settings.texture_size_ior);

		// Forcefully dropping the const because launch_args wants non-const arguments even though
		// they are never going to be modified
		GlossyDielectricDirectionalAlbedoSettings& bake_settings_non_const = const_cast<GlossyDielectricDirectionalAlbedoSettings&>(bake_settings);
		HIPRTRenderData& render_data = m_renderer->get_render_data();
		float* device_buffer = m_glossy_dielectric_directional_albedo_bake_buffer.get_device_pointer();

		void* launch_args[] = { &render_data, &bake_settings_non_const, &device_buffer };

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Launching glossy dielectric directional albedo baking...");
		m_glossy_dielectric_directional_albedo_bake_kernel.launch_asynchronous_3D(4, 4, 4, bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, bake_settings.texture_size_ior, launch_args, m_bake_stream);

		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_bake_stream));

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Glossy dielectric directional albedo bake completed!");

		// Writing results to disk
		std::vector<float> baked_data = m_glossy_dielectric_directional_albedo_bake_buffer.download_data();
		for (int i = 0; i < bake_settings.texture_size_ior; i++)
		{
			Image32Bit image = Image32Bit(baked_data.data() + i * bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness, bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, /* nb channels */ 1);

			std::string final_filename = std::to_string(i) + output_filename;
			image.write_image_hdr(final_filename.c_str(), false);
		}

		m_glossy_dielectric_directional_albedo_bake_buffer.free();
		m_glossy_dielectric_directional_albedo_bake_complete = true;
	});

	ThreadManager::detach_threads("bake_glossy_dielectric_directional_albedo_bake_start");
}

bool GPUBaker::is_glossy_dielectric_directional_albedo_bake_complete() const
{
	return m_glossy_dielectric_directional_albedo_bake_complete;
}

void GPUBaker::bake_ggx_glass_directional_albedo(const GGXGlassDirectionalAlbedoSettings& bake_settings, const std::string& output_filename)
{
	m_ggx_glass_directional_albedo_bake_complete = false;

	// Starting everything on a thread to avoid blocking to UI (during the compilation
	// of the kernel mainly)
	ThreadManager::start_thread("bake_ggx_glass_directional_albedo_bake_start", [this, &bake_settings, &output_filename] {
		std::string kernel_filepath = DEVICE_KERNELS_DIRECTORY "/Baking/GGXGlassDirectionalAlbedo.h";
		std::string kernel_function = "GGXGlassDirectionalAlbedoBake";

		OROCHI_CHECK_ERROR(oroCtxSetCurrent(m_renderer->get_hiprt_orochi_ctx()->orochi_ctx));

		if (!m_ggx_glass_directional_albedo_bake_kernel.has_been_compiled())
		{
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Compiling GGX glass directional albedo kernel...");

			// Taking the priority for the compilation as otherwise, the kernels
			// precompiling in the background are going to have the hand and we'll
			// never be able to compile our bake kernel and we'll never start baking
			// (until all kernels are precompiled in background of course but that's
			// going to take a long time)
			std::lock_guard<std::mutex> lock(m_compiler_priority_mutex);
			m_renderer->take_kernel_compilation_priority();
			m_ggx_glass_directional_albedo_bake_kernel = GPUKernel(kernel_filepath, kernel_function);
			m_ggx_glass_directional_albedo_bake_kernel.compile(m_renderer->get_hiprt_orochi_ctx());
			m_renderer->release_kernel_compilation_priority();
		}

		m_ggx_glass_directional_albedo_bake_buffer.resize(bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness * bake_settings.texture_size_ior);
		m_ggx_glass_directional_albedo_bake_buffer_inverse.resize(bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness * bake_settings.texture_size_ior);

		// Forcefully dropping the const because launch_args wants non-const arguments even though
		// they are never going to be modified
		GGXGlassDirectionalAlbedoSettings& bake_settings_non_const = const_cast<GGXGlassDirectionalAlbedoSettings&>(bake_settings);
		HIPRTRenderData& render_data = m_renderer->get_render_data();
		float* device_buffer = m_ggx_glass_directional_albedo_bake_buffer.get_device_pointer();
		float* device_buffer_inverse = m_ggx_glass_directional_albedo_bake_buffer_inverse.get_device_pointer();

		void* launch_args[] = { &render_data, &bake_settings_non_const, &device_buffer, &device_buffer_inverse };

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Launching GGX glass directional albedo baking...");
		m_ggx_glass_directional_albedo_bake_kernel.launch_asynchronous_3D(4, 4, 4, bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, bake_settings.texture_size_ior, launch_args, m_bake_stream);

		OROCHI_CHECK_ERROR(oroStreamSynchronize(m_bake_stream));
			
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "GGX glass directional albedo bake completed!");

		// Writing results to disk
		std::vector<float> baked_data = m_ggx_glass_directional_albedo_bake_buffer.download_data();
		for (int i = 0; i < bake_settings.texture_size_ior; i++)
		{
			Image32Bit image = Image32Bit(baked_data.data() + i * bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness, bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, /* nb channels */ 1);
			
			std::string final_filename = std::to_string(i) + output_filename;
			image.write_image_hdr(final_filename.c_str(), false);
		}

		std::vector<float> baked_data_inverse = m_ggx_glass_directional_albedo_bake_buffer_inverse.download_data();
		for (int i = 0; i < bake_settings.texture_size_ior; i++)
		{
			Image32Bit image = Image32Bit(baked_data_inverse.data() + i * bake_settings.texture_size_cos_theta_o * bake_settings.texture_size_roughness, bake_settings.texture_size_cos_theta_o, bake_settings.texture_size_roughness, /* nb channels */ 1);

			std::string final_filename = std::to_string(i) + "inv_" + output_filename;
			image.write_image_hdr(final_filename.c_str(), false);
		}

		m_ggx_glass_directional_albedo_bake_buffer.free();
		m_ggx_glass_directional_albedo_bake_buffer_inverse.free();
		m_ggx_glass_directional_albedo_bake_complete = true;
	});

	ThreadManager::detach_threads("bake_ggx_glass_directional_albedo_bake_start");
}

bool GPUBaker::is_ggx_glass_directional_albedo_bake_complete() const
{
	return m_ggx_glass_directional_albedo_bake_complete;
}
