/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_SETTINGS_WINDOW_H
#define IMGUI_SETTINGS_WINDOW_H

#include "Compiler/GPUKernelCompilerOptions.h"
#include "UI/ImGui/ImGuiRendererPerformancePreset.h"
#include "UI/PerformanceMetricsComputer.h"

#include "imgui.h"

class RenderWindow;
class GPURenderer;

class ImGuiSettingsWindow
{
public:
	static const char* TITLE;
	static const float BASE_SIZE;

	void set_render_window(RenderWindow* render_window);

	void draw();
	static void draw_camera_panel_static(const std::string& panel_title, RenderWindow* render_window, std::shared_ptr<GPURenderer> renderer);

private:
	void draw_header();
	void draw_render_settings_panel();
	void draw_russian_roulette_options();
	void display_view_selector();
	void apply_performance_preset(ImGuiRendererPerformancePreset performance_preset);
	void draw_camera_panel();
	// Static because we call this method from other ImGui classes to be able
	// to render the same panel
	void draw_environment_panel();

	void draw_sampling_panel();
	void draw_principled_bsdf_energy_conservation();
	void display_ReSTIR_DI_bias_status(std::shared_ptr<GPUKernelCompilerOptions> kernel_options);

	void draw_denoiser_panel();
	void draw_post_process_panel();

	void draw_performance_settings_panel();
	void draw_perf_metric_specific_panel(std::shared_ptr<PerformanceMetricsComputer> perf_metrics, const std::string& perf_metrics_key, const std::string& label);
	template <class... Args>
	std::string format_perf_metrics_tooltip_line(const std::string& label, const std::string& suffix, const std::string& longest_header_for_padding, const std::string& formatter_after_header, const Args& ...args);

	void draw_performance_metrics_panel();
	void draw_shader_kernels_panel();
	void draw_debug_panel();

	RenderWindow* m_render_window = nullptr;

	std::shared_ptr<ApplicationSettings> m_application_settings = nullptr;
	std::shared_ptr<GPURenderer> m_renderer = nullptr;
	std::shared_ptr<OpenImageDenoiser> m_render_window_denoiser = nullptr;
	std::shared_ptr<PerformanceMetricsComputer> m_render_window_perf_metrics = nullptr;

	ImVec2 m_current_size;
};

#endif
