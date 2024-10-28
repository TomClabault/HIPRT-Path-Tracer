/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_BAKING_WINDOW_H
#define IMGUI_BAKING_WINDOW_H

#include "Renderer/GPURenderer.h"

class RenderWindow;

class ImGuiBakingWindow
{
public:
	static const char* TITLE;

	void set_render_window(RenderWindow* render_window);

	void draw();
	void draw_ggx_energy_conservation_panel();
	void draw_GGX_E();
	void draw_GGX_glass_E();

private:
	RenderWindow* m_render_window = nullptr;

	std::shared_ptr<GPURenderer> m_renderer;
};

#endif
