/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ImGui/ImGuiRenderWindow.h"
#include "UI/RenderWindow.h"

const char* ImGuiRenderWindow::TITLE = "RenderWindow";

void ImGuiRenderWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void ImGuiRenderWindow::draw()
{
	ImGuiWindowFlags window_flags = 0;
	window_flags |= ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_::ImGuiWindowFlags_NoInputs;

	ImGui::Begin(ImGuiRenderWindow::TITLE, nullptr, window_flags);

	ImGui::Image((void*)(intptr_t)m_render_window->get_display_view_system()->m_fbo_texture, ImGui::GetWindowSize(), ImVec2(0, 1), ImVec2(1, 0));

	ImVec2 current_size = ImGui::GetWindowSize();
	if (current_size.x != m_current_size.x || current_size.y != m_current_size.y)
		m_render_window->resize(current_size.x, current_size.y);

	m_current_size = current_size;

	ImGui::End();
}

int ImGuiRenderWindow::get_width()
{
	return static_cast<int>(m_current_size.x);
}

int ImGuiRenderWindow::get_height()
{
	return static_cast<int>(m_current_size.y);
}
