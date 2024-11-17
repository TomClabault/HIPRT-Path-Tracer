/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMGUI_OBJECTS_WINDOW_H
#define IMGUI_OBJECTS_WINDOW_H

#include "Renderer/GPURenderer.h"

class RenderWindow;

class ImGuiObjectsWindow
{
public:
	static const char* TITLE;

	void set_render_window(RenderWindow* render_window);

	void draw();
	void draw_global_objects_panel();
	void draw_objects_panel();
	bool draw_material_presets(RendererMaterial& material);

private:
	RenderWindow* m_render_window = nullptr;

	std::shared_ptr<GPURenderer> m_renderer;
};

#endif
