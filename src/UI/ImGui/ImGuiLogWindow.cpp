/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ImGui/ImGuiLogWindow.h"
#include "UI/RenderWindow.h"

const char* ImGuiLogWindow::TITLE = "Logs";
const float ImGuiLogWindow::BASE_SIZE = 250.0f;

extern ImGuiLogger g_imgui_logger;

void ImGuiLogWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
}

void ImGuiLogWindow::draw()
{
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 10.0f));

	ImGui::Begin(ImGuiLogWindow::TITLE, nullptr);

	g_imgui_logger.draw(ImGuiLogWindow::TITLE);

	ImGui::PopStyleVar(3);
	ImGui::End();
}
