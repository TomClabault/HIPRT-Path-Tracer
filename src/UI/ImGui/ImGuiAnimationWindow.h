/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_ANIMATION_WINDOW_H
#define IMGUI_ANIMATION_WINDOW_H

#include "Renderer/GPURenderer.h"

class RenderWindow;

class ImGuiAnimationWindow
{
public:
	static const char* TITLE;

	void set_render_window(RenderWindow* render_window);

	void draw();
	void draw_header();
	void draw_camera_panel();
	void draw_envmap_panel();
	void draw_frame_sequence_rendering_panel();

private:
	RenderWindow* m_render_window = nullptr;

	std::shared_ptr<GPURenderer> m_renderer;
};

#endif
