/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
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

	static bool draw_material_editor(CPUMaterial& material,
									 const std::string& material_name,
									 std::shared_ptr<GPUKernelCompilerOptions> kernel_options,
									 bool& out_emission_changed);
	static bool draw_material_presets(CPUMaterial& material);

	std::unordered_set<int> filter_displayed_materials(int material_count,
													   const std::vector<std::string>& material_names,
													   const std::vector<std::string>& mesh_names,
													   const std::vector<int>& mesh_material_indices,
													   const std::string& filter_string) const;

private:
	RenderWindow* m_render_window = nullptr;

	std::shared_ptr<GPURenderer> m_renderer;
};

#endif
