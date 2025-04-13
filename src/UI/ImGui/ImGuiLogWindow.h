/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_LOG_WINDOW_H
#define IMGUI_LOG_WINDOW_H

#include "UI/ImGui/ImGuiLogger.h"

#include "imgui.h"

class RenderWindow;

class ImGuiLogWindow
{
public:
	static const char* TITLE;
	static const float BASE_SIZE;

	void set_render_window(RenderWindow* render_window);

	void draw();

private:
	RenderWindow* m_render_window;

	ImVec2 m_current_size;
};

#endif
