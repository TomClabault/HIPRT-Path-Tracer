/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GLFW/glfw3.h"
#include "UI/ImGui/ImGuiConvergenceGraphWidget.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

#include "implot.h"
#include "stb_image_write.h"

extern ImGuiLogger g_imgui_logger;

void ImGuiConvergenceGraphWidget::draw()
{
	// TODO save the data to a file such that we can keep plotting accross sessions?
	if (ImPlot::BeginPlot(m_plot_title.c_str(), ImVec2(m_plot_width, m_plot_height)))
	{
		ImPlot::SetupLegend(ImPlotLocation_East | ImPlotLocation_North, 0);

		ImPlot::SetupAxes(m_x_axis_name.c_str(), m_y_axis_name.c_str(), ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

		for (size_t i = 0; i < m_recorded_legends.size(); i++)
		{
			ImVec4 colors[] = { ImVec4(78 / 255.0f, 121 / 255.0f, 167 / 255.0f, 1.0f),	ImVec4(242 / 255.0f, 142 / 255.0f, 43 / 255.0f, 1.0f),
								ImVec4(225 / 255.0f, 87 / 255.0f, 89 / 255.0f, 1.0f),	ImVec4(118 / 255.0f, 183 / 255.0f, 178 / 255.0f, 1.0f),
								ImVec4(89 / 255.0f, 161 / 255.0f, 79 / 255.0f, 1.0f),	ImVec4(237 / 255.0f, 201 / 255.0f, 72 / 255.0f, 1.0f),
								ImVec4(176 / 255.0f, 122 / 255.0f, 161 / 255.0f, 1.0f), ImVec4(255 / 255.0f, 157 / 255.0f, 167 / 255.0f, 1.0f),
								ImVec4(156 / 255.0f, 117 / 255.0f, 95 / 255.0f, 1.0f),	ImVec4(186 / 255.0f, 176 / 255.0f, 172 / 255.0f, 1.0f) };

			if (i > 10)
				ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, m_line_weight);
			else
				ImPlot::SetNextLineStyle(colors[i], m_line_weight);
			ImPlot::PlotLine(m_recorded_legends.at(i).c_str(), m_recorded_xs_list.at(i).data(), m_recorded_ys_list.at(i).data(),
							 m_recorded_xs_list.at(0).size());
		}

		ImPlot::EndPlot();

		// Store the last plot position and size for screenshot purposes
		ImVec2 min = ImGui::GetItemRectMin();
		ImVec2 max = ImGui::GetItemRectMax();

		m_last_plot_pos	   = min;
		m_last_plot_size.x = max.x - min.x;
		m_last_plot_size.y = max.y - min.y;
	}
}

void ImGuiConvergenceGraphWidget::request_screenshot(bool request, bool to_file)
{
	if (to_file)
		m_screenshot_to_file_requested = request;
	else
		m_screenshot_to_clipboard_requested = request;
}

void ImGuiConvergenceGraphWidget::process_screenshots()
{
	if (m_screenshot_to_file_requested)
	{
		if (!screenshot_graph_to_file("convergence_graph.png"))
			g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Failed to save convergence graph screenshot to file.");

		m_screenshot_to_file_requested = false;
	}
	else if (m_screenshot_to_clipboard_requested)
	{
		screenshot_graph_to_clipboard();

		m_screenshot_to_clipboard_requested = false;
	}
}

std::vector<unsigned char> ImGuiConvergenceGraphWidget::screenshot_graph_to_memory(int& out_width, int& out_height)
{
	ImGuiIO& io		= ImGui::GetIO();
	ImVec2 fb_scale = io.DisplayFramebufferScale; // Handle DPI

	int px = (int)(m_last_plot_pos.x * fb_scale.x + 0.5f);
	// OpenGL origin is bottom-left, ImGui uses top-left, so compute y accordingly:
	int py = (int)((io.DisplaySize.y - (m_last_plot_pos.y + m_last_plot_size.y)) * fb_scale.y + 0.5f);

	out_width  = (int)(m_last_plot_size.x * fb_scale.x + 0.5f);
	out_height = (int)(m_last_plot_size.y * fb_scale.y + 0.5f);

	if (out_width <= 0 || out_height <= 0)
		return std::vector<unsigned char>();

	std::vector<unsigned char> pixels(out_width * out_height * 4);

	glFinish();
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(px, py, out_width, out_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

	// Flip rows because glReadPixels returns bottom->top but PNG expects top->down
	std::vector<unsigned char> flipped(out_width * out_height * 4);
	for (int row = 0; row < out_height; ++row)
	{
		memcpy(&flipped[row * out_width * 4], &pixels[(out_height - 1 - row) * out_width * 4], (size_t)out_width * 4);
	}

	return flipped;
}

bool ImGuiConvergenceGraphWidget::screenshot_graph_to_file(const char* filename)
{
	int width, height;
	std::vector<unsigned char> pixels = screenshot_graph_to_memory(width, height);

	unsigned int stride_bytes = width * 4;
	return stbi_write_png(filename, width, height, 4, pixels.data(), stride_bytes) != 0;
}

void ImGuiConvergenceGraphWidget::screenshot_graph_to_clipboard()
{
	int width, height;
	std::vector<unsigned char> pixels = screenshot_graph_to_memory(width, height);

	Utils::copy_image_to_clipboard(Image8Bit(pixels, width, height, 4), false);
}

int& ImGuiConvergenceGraphWidget::get_plot_width()
{
	return m_plot_width;
}

int& ImGuiConvergenceGraphWidget::get_plot_height()
{
	return m_plot_height;
}

float& ImGuiConvergenceGraphWidget::get_line_weight()
{
	return m_line_weight;
}

std::string& ImGuiConvergenceGraphWidget::get_plot_title()
{
	return m_plot_title;
}

std::string& ImGuiConvergenceGraphWidget::get_x_axis_name()
{
	return m_x_axis_name;
}

std::string& ImGuiConvergenceGraphWidget::get_y_axis_name()
{
	return m_y_axis_name;
}

std::vector<std::string>& ImGuiConvergenceGraphWidget::get_recorded_legends()
{
	return m_recorded_legends;
}

std::vector<std::vector<float>>& ImGuiConvergenceGraphWidget::get_recorded_xs_list()
{
	return m_recorded_xs_list;
}

std::vector<std::vector<float>>& ImGuiConvergenceGraphWidget::get_recorded_ys_list()
{
	return m_recorded_ys_list;
}

void ImGuiConvergenceGraphWidget::set_x_axis_name(const std::string& name)
{
	m_x_axis_name = name;
}

void ImGuiConvergenceGraphWidget::set_y_axis_name(const std::string& name)
{
	m_y_axis_name = name;
}
