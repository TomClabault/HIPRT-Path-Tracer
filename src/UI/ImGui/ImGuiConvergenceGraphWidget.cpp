/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "UI/ImGui/ImGuiConvergenceGraphWidget.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "UI/RenderWindow.h"
#include "Utils/Utils.h"

#include "implot.h"
#include "stb_image_write.h"

extern ImGuiLogger g_imgui_logger;

ImGuiConvergenceGraphWidget::ImGuiConvergenceGraphWidget() : m_screenshoter(this) {}

void ImGuiConvergenceGraphWidget::draw(ImVec2 plotSize)
{
	if (plotSize.x < 0)
		plotSize.x = (float)m_plot_width;
	if (plotSize.y < 0)
		plotSize.y = (float)m_plot_height;

	// TODO save the data to a file such that we can keep plotting accross sessions?
	if (ImPlot::BeginPlot(m_plot_title.c_str(), ImVec2(plotSize.x, plotSize.y)))
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

			if (i >= 10)
				ImPlot::SetNextLineStyle(IMPLOT_AUTO_COL, m_line_weight);
			else
				ImPlot::SetNextLineStyle(colors[i], m_line_weight);
			ImPlot::PlotLine(m_recorded_legends.at(i).c_str(), m_recorded_xs_list.at(i).data(), m_recorded_ys_list.at(i).data(),
							 m_recorded_xs_list.at(0).size());
		}

		ImPlot::EndPlot();
	}
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

bool ImGuiConvergenceGraphWidget::screenshot_graph_to_file(const std::string_view filename)
{
	return m_screenshoter.screenshot_graph_to_file(m_plot_width, m_plot_height, filename.data());
}

void ImGuiConvergenceGraphWidget::screenshot_graph_to_clipboard()
{
	m_screenshoter.screenshot_graph_to_clipboard(m_plot_width, m_plot_height);
}

void ImGuiConvergenceGraphWidget::set_x_axis_name(const std::string& name)
{
	m_x_axis_name = name;
}

void ImGuiConvergenceGraphWidget::set_y_axis_name(const std::string& name)
{
	m_y_axis_name = name;
}

void ImGuiConvergenceGraphWidget::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
	m_screenshoter.set_render_window(render_window);
}
