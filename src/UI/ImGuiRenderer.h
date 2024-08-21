/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_RENDERER_H
#define IMGUI_RENDERER_H

#include "Renderer/OpenImageDenoiser.h"
#include "UI/ApplicationSettings.h"
#include "UI/PerformanceMetricsComputer.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <memory>

class GPURenderer;
class RenderWindow;

class ImGuiRenderer
{
public:
	ImGuiRenderer();

	/**
  	 * Adds a tooltip to the last widget that auto wraps after 80 characters
	 */
	static void WrappingTooltip(const std::string& text);

	void set_render_window(RenderWindow* renderer);

	void draw_imgui_interface();

	void draw_render_settings_panel();
	void draw_camera_panel();
	void draw_environment_panel();

	void draw_sampling_panel();
	void display_ReSTIR_DI_bias_status(std::shared_ptr<GPUKernelCompilerOptions> kernel_options);

	void draw_objects_panel();
	void draw_denoiser_panel();
	void draw_post_process_panel();
	void draw_performance_settings_panel();
	void draw_performance_metrics_panel();
	void draw_debug_panel();

	void rescale_ui();

private:
	RenderWindow* m_render_window = nullptr;

	std::shared_ptr<ApplicationSettings> m_application_settings = nullptr;
	std::shared_ptr<GPURenderer> m_renderer = nullptr;
	std::shared_ptr<OpenImageDenoiser> m_render_window_denoiser = nullptr;
	std::shared_ptr<PerformanceMetricsComputer> m_render_window_perf_metrics = nullptr;
};

#endif
