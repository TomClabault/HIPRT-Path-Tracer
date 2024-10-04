/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_RENDER_WINDOW_H
#define IMGUI_RENDER_WINDOW_H

#include "imgui.h"

class RenderWindow;

class ImGuiRenderWindow
{
public:
	static const char* TITLE;

	void set_render_window(RenderWindow* render_window);

	void draw();

	bool is_hovered() const;

private:
	RenderWindow* m_render_window;

	ImVec2 m_current_size;
	bool m_is_hovered = false;
};

#endif
