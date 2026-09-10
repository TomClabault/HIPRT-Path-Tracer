/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_H
#define UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_H

#include "UI/ImGui/ImGuiConvergenceGraphWidgetScreenshotter.h"

#include "imgui.h"
#include "implot.h"

#include <string>
#include <vector>

class RenderWindow;

class ImGuiConvergenceGraphWidget
{
public:
	ImGuiConvergenceGraphWidget();

	void draw(ImVec2 plot_size = ImVec2(-1, -1));

	std::vector<unsigned char> screenshot_graph_to_memory(int& out_width, int& out_height, bool flip_y);

	int& get_plot_width();
	int& get_plot_height();
	float& get_line_weight();
	std::string& get_plot_title();
	bool& get_log_x_axis();
	bool& get_log_y_axis();
	std::string& get_x_axis_name();
	std::string& get_y_axis_name();

	std::vector<std::string>& get_recorded_legends();
	std::vector<std::vector<float>>& get_recorded_xs_list();
	std::vector<std::vector<float>>& get_recorded_ys_list();
	std::vector<int>& get_recorded_line_styles();
	std::vector<int>& get_recorded_color_indices();

	bool screenshot_graph_to_file(const std::string_view filename);
	void screenshot_graph_to_clipboard();

	void set_x_axis_name(const std::string& name);
	void set_y_axis_name(const std::string& name);
	void set_render_window(RenderWindow* render_window);

private:
	void prepare_screenshot_fbo();
	void draw_plot_only(ImVec2 plotSize);

	void restore_viewport_texture_after_screenshotting();

private:
	RenderWindow* m_render_window = nullptr;

	int m_plot_width		  = 575;
	int m_plot_height		  = 400;
	float m_line_weight		  = 3.0f;
	std::string m_plot_title  = "Convergence graph";
	bool m_log_x_axis		  = false;
	bool m_log_y_axis		  = false;
	std::string m_x_axis_name = "Samples";
	std::string m_y_axis_name = "Error (RMSE)";

	std::vector<std::string> m_recorded_legends;
	// The final list of points that will be used for graphing
	std::vector<std::vector<float>> m_recorded_xs_list;
	std::vector<std::vector<float>> m_recorded_ys_list;
	std::vector<int> m_recorded_line_styles;
	std::vector<int> m_recorded_color_indices;

	ImGuiConvergenceGraphWidgetScreenshotter m_screenshoter;
};

#endif // #ifndef UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_H
