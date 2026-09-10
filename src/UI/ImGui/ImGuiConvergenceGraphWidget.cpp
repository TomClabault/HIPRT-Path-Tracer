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
#include "implot_internal.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstring>

extern ImGuiLogger g_imgui_logger;

namespace
{
	struct DashedLegendSwatch
	{
		ImGuiID item_id;
		ImU32 color;
	};

	struct LegendLabelComparator
	{
		ImPlotItemGroup* items;

		bool operator()(int left_index, int right_index) const
		{
			return std::strcmp(items->GetLegendLabel(left_index), items->GetLegendLabel(right_index)) < 0;
		}
	};

	// Keep dash lengths visible and consistent when the plot data or axis scale changes.
	void draw_dashed_line(const std::vector<float>& x_values, const std::vector<float>& y_values, ImU32 color, float line_weight)
	{
		ImDrawList* plot_draw_list	   = ImPlot::GetPlotDrawList();
		float dash_length			   = 8.0f;
		float gap_length			   = 5.0f;
		float remaining_pattern_length = dash_length;
		bool draw_dash				   = true;
		size_t point_count			   = std::min(x_values.size(), y_values.size());

		ImPlot::PushPlotClipRect();
		for (size_t point_index = 1; point_index < point_count; point_index++)
		{
			ImVec2 segment_start = ImPlot::PlotToPixels((double)x_values.at(point_index - 1), (double)y_values.at(point_index - 1));
			ImVec2 segment_end	 = ImPlot::PlotToPixels((double)x_values.at(point_index), (double)y_values.at(point_index));
			if (!std::isfinite(segment_start.x) || !std::isfinite(segment_start.y) || !std::isfinite(segment_end.x) || !std::isfinite(segment_end.y))
			{
				remaining_pattern_length = dash_length;
				draw_dash				 = true;
				continue;
			}

			float direction_x	 = segment_end.x - segment_start.x;
			float direction_y	 = segment_end.y - segment_start.y;
			float segment_length = std::sqrt(direction_x * direction_x + direction_y * direction_y);
			if (segment_length <= 0.0f)
				continue;

			float segment_offset = 0.0f;
			while (segment_offset < segment_length)
			{
				float piece_length = std::min(remaining_pattern_length, segment_length - segment_offset);
				float start_factor = segment_offset / segment_length;
				float end_factor   = (segment_offset + piece_length) / segment_length;
				ImVec2 piece_start = ImVec2(segment_start.x + direction_x * start_factor, segment_start.y + direction_y * start_factor);
				ImVec2 piece_end   = ImVec2(segment_start.x + direction_x * end_factor, segment_start.y + direction_y * end_factor);

				if (draw_dash)
					plot_draw_list->AddLine(piece_start, piece_end, color, line_weight);

				segment_offset += piece_length;
				remaining_pattern_length -= piece_length;
				if (remaining_pattern_length <= 0.0001f)
				{
					draw_dash				 = !draw_dash;
					remaining_pattern_length = draw_dash ? dash_length : gap_length;
				}
			}
		}
		ImPlot::PopPlotClipRect();
	}

	void draw_dashed_legend_swatches(ImPlotPlot& plot, const std::vector<DashedLegendSwatch>& dashed_legend_swatches, float line_weight)
	{
		if (dashed_legend_swatches.empty() || plot.Items.GetLegendCount() == 0)
			return;

		float text_height	  = ImGui::GetTextLineHeight();
		float icon_size		  = text_height;
		float icon_shrink	  = 2.0f;
		float sum_label_width = 0.0f;
		bool vertical		  = !ImHasFlag(plot.Items.Legend.Flags, ImPlotLegendFlags_Horizontal);
		std::vector<int> legend_indices;

		for (int legend_index = 0; legend_index < plot.Items.GetLegendCount(); legend_index++)
			legend_indices.push_back(legend_index);

		if (ImHasFlag(plot.Items.Legend.Flags, ImPlotLegendFlags_Sort))
			std::sort(legend_indices.begin(), legend_indices.end(), LegendLabelComparator{ &plot.Items });

		ImDrawList* draw_list = ImGui::GetWindowDrawList();
		ImGui::PushClipRect(plot.Items.Legend.RectClamped.Min, plot.Items.Legend.RectClamped.Max, true);

		for (int display_index = 0; display_index < (int)legend_indices.size(); display_index++)
		{
			int legend_index  = ImHasFlag(plot.Items.Legend.Flags, ImPlotLegendFlags_Reverse) ? (int)legend_indices.size() - 1 - display_index : display_index;
			ImPlotItem* item  = plot.Items.GetLegendItem(legend_indices.at(legend_index));
			const char* label = plot.Items.GetLegendLabel(legend_indices.at(legend_index));
			float label_width = ImGui::CalcTextSize(label, nullptr, true).x;
			float top_left_x  = plot.Items.Legend.Rect.Min.x + ImPlot::GetStyle().LegendInnerPadding.x;
			float top_left_y  = plot.Items.Legend.Rect.Min.y + ImPlot::GetStyle().LegendInnerPadding.y;
			if (vertical)
				top_left_y += display_index * (text_height + ImPlot::GetStyle().LegendSpacing.y);
			else
				top_left_x += display_index * (icon_size + ImPlot::GetStyle().LegendSpacing.x) + sum_label_width;
			ImVec2 top_left = ImVec2(top_left_x, top_left_y);
			sum_label_width += label_width;

			ImRect icon_bounds;
			icon_bounds.Min = ImVec2(top_left.x + icon_shrink, top_left.y + icon_shrink);
			icon_bounds.Max = ImVec2(top_left.x + icon_size - icon_shrink, top_left.y + icon_size - icon_shrink);

			for (size_t swatch_index = 0; swatch_index < dashed_legend_swatches.size(); swatch_index++)
			{
				if (dashed_legend_swatches.at(swatch_index).item_id != item->ID)
					continue;

				float swatch_line_weight = std::max(1.0f, std::min(line_weight, icon_bounds.GetHeight()));
				float line_y			 = (icon_bounds.Min.y + icon_bounds.Max.y) * 0.5f;
				float dash_length		 = 3.0f;
				float gap_length		 = 2.0f;
				float line_start		 = icon_bounds.Min.x;
				while (line_start < icon_bounds.Max.x)
				{
					float line_end = std::min(line_start + dash_length, icon_bounds.Max.x);
					draw_list->AddLine(ImVec2(line_start, line_y), ImVec2(line_end, line_y), dashed_legend_swatches.at(swatch_index).color, swatch_line_weight);
					line_start += dash_length + gap_length;
				}
				break;
			}
		}

		ImGui::PopClipRect();
	}
} // namespace

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
		ImPlotPlot* plot = GImPlot->CurrentPlot;
		std::vector<DashedLegendSwatch> dashed_legend_swatches;

		ImPlot::SetupLegend(ImPlotLocation_East | ImPlotLocation_North, 0);
		ImPlot::SetupAxes(m_x_axis_name.c_str(), m_y_axis_name.c_str(), ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
		ImPlot::SetupAxisScale(ImAxis_X1, m_log_x_axis ? ImPlotScale_Log10 : ImPlotScale_Linear);
		ImPlot::SetupAxisScale(ImAxis_Y1, m_log_y_axis ? ImPlotScale_Log10 : ImPlotScale_Linear);

		for (size_t i = 0; i < m_recorded_legends.size(); i++)
		{
			ImVec4 colors[] = { ImVec4(78 / 255.0f, 121 / 255.0f, 167 / 255.0f, 1.0f),	ImVec4(242 / 255.0f, 142 / 255.0f, 43 / 255.0f, 1.0f),
								ImVec4(225 / 255.0f, 87 / 255.0f, 89 / 255.0f, 1.0f),	ImVec4(118 / 255.0f, 183 / 255.0f, 178 / 255.0f, 1.0f),
								ImVec4(89 / 255.0f, 161 / 255.0f, 79 / 255.0f, 1.0f),	ImVec4(237 / 255.0f, 201 / 255.0f, 72 / 255.0f, 1.0f),
								ImVec4(176 / 255.0f, 122 / 255.0f, 161 / 255.0f, 1.0f), ImVec4(255 / 255.0f, 157 / 255.0f, 167 / 255.0f, 1.0f),
								ImVec4(156 / 255.0f, 117 / 255.0f, 95 / 255.0f, 1.0f),	ImVec4(186 / 255.0f, 176 / 255.0f, 172 / 255.0f, 1.0f) };

			int color_index = m_recorded_color_indices.at(i);
			ImVec4 line_color;
			if (color_index >= 0)
				line_color = color_index < 10 ? colors[color_index] : ImPlot::GetColormapColor(color_index);
			else if (i >= 10)
				line_color = ImPlot::GetColormapColor((int)i);
			else
				line_color = colors[i];

			if (m_recorded_line_styles.at(i) == 1)
			{
				ImVec4 transparent_line_color = line_color;
				transparent_line_color.w	  = 0.0f;
				ImPlot::SetNextLineStyle(transparent_line_color, m_line_weight);
				std::string dashed_fit_label = "##dashed_fit_" + std::to_string(i);
				ImPlot::PlotLine(dashed_fit_label.c_str(), m_recorded_xs_list.at(i).data(), m_recorded_ys_list.at(i).data(), m_recorded_xs_list.at(0).size());

				ImVec4 transparent_legend_color = line_color;
				transparent_legend_color.w		= 0.0f;
				ImPlot::SetNextLineStyle(transparent_legend_color, m_line_weight);
				ImPlot::PlotDummy(m_recorded_legends.at(i).c_str());
				ImPlotItem* dashed_legend_item = plot->Items.GetItem(m_recorded_legends.at(i).c_str());
				if (dashed_legend_item != nullptr)
				{
					DashedLegendSwatch dashed_legend_swatch;
					dashed_legend_swatch.item_id = dashed_legend_item->ID;
					dashed_legend_swatch.color	 = ImGui::ColorConvertFloat4ToU32(line_color);
					dashed_legend_swatches.push_back(dashed_legend_swatch);
				}
				draw_dashed_line(m_recorded_xs_list.at(i), m_recorded_ys_list.at(i), ImGui::ColorConvertFloat4ToU32(line_color), m_line_weight);
			}
			else
			{
				ImPlot::SetNextLineStyle(line_color, m_line_weight);
				ImPlot::PlotLine(m_recorded_legends.at(i).c_str(), m_recorded_xs_list.at(i).data(), m_recorded_ys_list.at(i).data(),
								 m_recorded_xs_list.at(0).size());
			}
		}

		ImPlot::EndPlot();
		draw_dashed_legend_swatches(*plot, dashed_legend_swatches, m_line_weight);
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

bool& ImGuiConvergenceGraphWidget::get_log_x_axis()
{
	return m_log_x_axis;
}

bool& ImGuiConvergenceGraphWidget::get_log_y_axis()
{
	return m_log_y_axis;
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

std::vector<int>& ImGuiConvergenceGraphWidget::get_recorded_line_styles()
{
	return m_recorded_line_styles;
}

std::vector<int>& ImGuiConvergenceGraphWidget::get_recorded_color_indices()
{
	return m_recorded_color_indices;
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
