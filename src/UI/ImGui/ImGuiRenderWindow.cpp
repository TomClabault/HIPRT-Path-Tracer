/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ImGui/ImGuiRenderWindow.h"
#include "UI/RenderWindow.h"

const char* ImGuiRenderWindow::TITLE = "Viewport";

void ImGuiRenderWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void ImGuiRenderWindow::draw()
{
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

	ImGui::Begin(ImGuiRenderWindow::TITLE, nullptr);

	// GetWindowContentRegion() to get the size without the title bar and other decorations.
	ImVec2 content_min = ImGui::GetWindowContentRegionMin();
	ImVec2 content_max = ImGui::GetWindowContentRegionMax();
	ImVec2 current_size = ImVec2(content_max.x - content_min.x, content_max.y - content_min.y);

	ImGui::Image((void*)(intptr_t)m_render_window->get_display_view_system()->m_fbo_texture, current_size, ImVec2(0, 1), ImVec2(1, 0));

	if (current_size.x != m_current_size.x || current_size.y != m_current_size.y)
		m_render_window->resize(current_size.x, current_size.y);

	m_current_size = current_size;
	m_is_hovered = ImGui::IsWindowHovered();

	ImGui::PopStyleVar(3);
	ImGui::End();
}

bool ImGuiRenderWindow::is_hovered() const
{
	return m_is_hovered;
}

ImVec2 ImGuiRenderWindow::get_size() const
{
	return m_current_size;
}
