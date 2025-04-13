/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "Threads/ThreadManager.h"
#include "UI/ImGui/ImGuiRenderer.h"
#include "UI/RenderWindow.h"

#include "imgui_internal.h"

#include <chrono>
#include <unordered_map>

ImGuiRenderer::ImGuiRenderer()
{
	ImGuiViewport* viewport = ImGui::GetMainViewport();
	float windowDpiScale = viewport->DpiScale;
	if (windowDpiScale > 1.0f)
		ImGui::GetStyle().ScaleAllSizes(windowDpiScale);
}

void ImGuiRenderer::init_imgui(GLFWwindow* glfw_window)
{
	// Setting ImGui up
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	ImGui_ImplGlfw_InitForOpenGL(glfw_window, true);
	ImGui_ImplOpenGL3_Init();
}

void ImGuiRenderer::add_tooltip(const std::string& tooltip_text, ImGuiHoveredFlags flags)
{
	if (ImGui::IsItemHovered(flags))
		ImGuiRenderer::wrapping_tooltip(tooltip_text);
}

void ImGuiRenderer::add_warning(const std::string& warning_text)
{
	ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Warning: ");
	ImGuiRenderer::show_help_marker(warning_text.c_str());
}

bool ImGuiRenderer::ComboWithTooltips(const std::string& combo_text, int* combo_value, const char** items, size_t items_count, const char** tooltips)
{
	if (ImGui::BeginCombo(combo_text.c_str(), items[*combo_value]))
	{
		for (int i = 0; i < items_count; i++)
		{
			const bool is_selected = (*combo_value == i);

			if (ImGui::Selectable(items[i], is_selected))
			{
				*combo_value = i;

				ImGui::EndCombo();
				return true;
			}
			ImGuiRenderer::add_tooltip(tooltips[i]);

			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	return false;
}

void ImGuiRenderer::wrapping_tooltip(const std::string& text)
{
	ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 32.0f, 0.0f));
	ImGui::BeginTooltip();
	ImGui::PushTextWrapPos(0.0f);
	ImGui::Text("%s", text.c_str());
	ImGui::PopTextWrapPos();
	ImGui::EndTooltip();
}

void ImGuiRenderer::show_help_marker(const std::string& text, ImVec4 color)
{
	ImGui::SameLine();
	if (color.x == -1.0f && color.y == -1.0f && color.z == -1.0 && color.w == -1.0f)
		// Default "disabled" color
		ImGui::TextDisabled("(?)");
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Text, color);
		ImGui::Text("(?)");
		ImGui::PopStyleColor();
	}
	add_tooltip(text);
}

void ImGuiRenderer::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
	m_imgui_settings_window.set_render_window(render_window);
	m_imgui_animation_window.set_render_window(render_window);
	m_imgui_baking_window.set_render_window(render_window);
	m_imgui_objects_window.set_render_window(render_window);
	m_imgui_render_window.set_render_window(render_window);
	m_imgui_log_window.set_render_window(render_window);
}

void ImGuiRenderer::draw_interface()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	rescale_ui();
	draw_dockspace();
	draw_settings_window();
	draw_animation_window();
	draw_baking_window();
	draw_objects_window();
	draw_log_window();
	draw_render_window();

	// ImGui::ShowDemoWindow();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiRenderer::rescale_ui()
{
	ImGuiIO& io = ImGui::GetIO();
	ImGuiViewport* viewport = ImGui::GetMainViewport();

	io.FontGlobalScale = viewport->DpiScale;
}

void ImGuiRenderer::draw_dockspace()
{
	// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
	// because it would be confusing to have two docking targets within each others.
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;

	ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->Pos);
	ImGui::SetNextWindowSize(viewport->Size);
	ImGui::SetNextWindowViewport(viewport->ID);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration;

	ImGui::Begin("HIPRT-Path-Tracer", nullptr, window_flags);

	// DockSpace
	ImGuiIO& io = ImGui::GetIO();
	if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
	{
		static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;

		ImGuiID dockspace_id = ImGui::GetID("DockSpace");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

		static auto first_time = true;
		if (first_time)
		{
			first_time = false;

			ImGui::DockBuilderRemoveNode(dockspace_id); // clear any previous layout
			ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
			ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);

			int renderer_width = m_render_window->get_renderer()->m_render_resolution.x;
			int renderer_height = m_render_window->get_renderer()->m_render_resolution.y;
			m_dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, ImGuiSettingsWindow::BASE_SIZE / (renderer_width + ImGuiSettingsWindow::BASE_SIZE), nullptr, &dockspace_id);
			m_dock_id_bottom = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, ImGuiLogWindow::BASE_SIZE / (renderer_height + ImGuiLogWindow::BASE_SIZE), nullptr, &dockspace_id);

			// we now dock our windows into the docking node we made above
			ImGui::DockBuilderDockWindow(ImGuiLogWindow::TITLE, m_dock_id_bottom);
			ImGui::DockBuilderDockWindow(ImGuiSettingsWindow::TITLE, m_dock_id_left);
			ImGui::DockBuilderDockWindow(ImGuiRenderWindow::TITLE, dockspace_id);
			ImGui::DockBuilderFinish(dockspace_id);
		}
	}

	ImGui::PopStyleVar(3);
	ImGui::End();
}

void ImGuiRenderer::draw_settings_window()
{
	m_imgui_settings_window.draw();
}

void ImGuiRenderer::draw_animation_window()
{
	// "Tabbing" / "docking" / "putting" the window into the left part of the dock
	// (basically, this window will act as a tab of the "Settings" window
	ImGui::SetNextWindowDockID(m_dock_id_left, ImGuiCond_Always);

	m_imgui_animation_window.draw();
}

void ImGuiRenderer::draw_baking_window()
{
	// "Tabbing" / "docking" / "putting" the window into the left part of the dock
	// (basically, this window will act as a tab of the "Settings" window
	ImGui::SetNextWindowDockID(m_dock_id_left, ImGuiCond_Always);

	m_imgui_baking_window.draw();
}

void ImGuiRenderer::draw_objects_window()
{
	// "Tabbing" / "docking" / "putting" the window into the left part of the dock
	// (basically, this window will act as a tab of the "Settings" window
	ImGui::SetNextWindowDockID(m_dock_id_left, ImGuiCond_Always);

	m_imgui_objects_window.draw();
}

void ImGuiRenderer::draw_render_window()
{
	m_imgui_render_window.draw();
}

void ImGuiRenderer::draw_log_window()
{
	m_imgui_log_window.draw();
}

ImGuiRenderWindow& ImGuiRenderer::get_imgui_render_window()
{
	return m_imgui_render_window;
}
