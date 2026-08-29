/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_SCREENSHOTTER_H
#define UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_SCREENSHOTTER_H

#include "GL/glew.h"
#include "imgui.h"
#include "implot.h"

#include <string_view>
#include <vector>

class ImGuiConvergenceGraphWidget;
class RenderWindow;

class ImGuiConvergenceGraphWidgetScreenshotter
{
public:
	static constexpr int SCREENSHOT_PADDING_WIDTH  = 15;
	static constexpr int SCREENSHOT_PADDING_HEIGHT = 15;

	ImGuiConvergenceGraphWidgetScreenshotter(ImGuiConvergenceGraphWidget* graph_widget);

	bool screenshot_graph_to_file(int screenshot_width, int screenshot_height, const std::string_view filename);
	void screenshot_graph_to_clipboard(int screenshot_width, int screenshot_height);

	void set_render_window(RenderWindow* render_window);

private:
	void init();

	std::vector<unsigned char> screenshot_graph_to_memory(int screenshot_width, int screenshot_height, bool flip_y);

	void regenerate_capture_fbo(int width, int height);
	void destroy_capture_fbo();

	void restore_viewport_texture_after_screenshotting();

private:
	ImGuiConvergenceGraphWidget* m_graph_widget = nullptr;
	RenderWindow* m_render_window				= nullptr;

	ImGuiContext* m_captureImguiCtx	  = nullptr;
	ImPlotContext* m_captureImPlotCtx = nullptr;

	GLuint m_capture_fbo				= 0;
	GLuint m_capture_color_tex			= 0;
	GLuint m_capture_depth_renderbuffer = 0;
	int m_capture_width					= 0;
	int m_capture_height				= 0;

	bool m_init_done = false;
};

#endif // #ifndef UI_IMGUI_IMGUI_CONVERGENCE_GRAPH_WIDGET_SCREENSHOTTER_H
