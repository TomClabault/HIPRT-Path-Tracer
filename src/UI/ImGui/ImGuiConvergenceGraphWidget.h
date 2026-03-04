/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_H
#define UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_H

#include "imgui.h"

#include <string>
#include <vector>

class ImGuiConvergenceGraphWidget
{
public:
	void draw();

	void request_screenshot(bool request, bool to_file);
	void process_screenshots();

	std::vector<unsigned char> screenshot_graph_to_memory(int& out_width, int& out_height);

	int& get_plot_width();
	int& get_plot_height();
	float& get_line_weight();
	std::string& get_plot_title();
	std::string& get_x_axis_name();
	std::string& get_y_axis_name();

	std::vector<std::string>& get_recorded_legends();
	std::vector<std::vector<float>>& get_recorded_xs_list();
	std::vector<std::vector<float>>& get_recorded_ys_list();

	void set_x_axis_name(const std::string& name);
	void set_y_axis_name(const std::string& name);

private:
	bool screenshot_graph_to_file(const char* filename);
	void screenshot_graph_to_clipboard();

private:
	int m_plot_width		  = 575;
	int m_plot_height		  = 400;
	float m_line_weight		  = 3.0f;
	std::string m_plot_title  = "Convergence graph";
	std::string m_x_axis_name = "Samples";
	std::string m_y_axis_name = "Error (RMSE)";

	std::vector<std::string> m_recorded_legends;
	// The final list of points that will be used for graphing
	std::vector<std::vector<float>> m_recorded_xs_list;
	std::vector<std::vector<float>> m_recorded_ys_list;

	// Private data for handling screenshots
	bool m_screenshot_to_file_requested		 = false;
	bool m_screenshot_to_clipboard_requested = false;

	ImVec2 m_last_plot_pos	= ImVec2(0, 0);
	ImVec2 m_last_plot_size = ImVec2(0, 0);
};

#endif
