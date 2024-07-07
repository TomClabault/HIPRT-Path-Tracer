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

	void set_render_window(RenderWindow* renderer);

	void draw_imgui_interface();
	void draw_render_settings_panel();
	void draw_environment_panel();
	void draw_sampling_panel();
	void draw_objects_panel();
	void draw_denoiser_panel();
	void draw_post_process_panel();
	void draw_performance_panel();
	void draw_debug_panel();

	void rescale_ui();

private:
	RenderWindow* m_render_window;

	std::shared_ptr<ApplicationSettings> m_application_settings;
	std::shared_ptr<GPURenderer> m_renderer;
	std::shared_ptr<OpenImageDenoiser> m_denoiser;
	std::shared_ptr<PerformanceMetricsComputer> m_perf_metrics;
};

#endif
