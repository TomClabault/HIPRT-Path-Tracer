/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
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

void ImGuiRenderer::wrapping_tooltip(const std::string& text)
{
	ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 32.0f, 0.0f));
	ImGui::BeginTooltip();
	ImGui::PushTextWrapPos(0.0f);
	ImGui::Text("%s", text.c_str());
	ImGui::PopTextWrapPos();
	ImGui::EndTooltip();
}

void ImGuiRenderer::show_help_marker(const std::string& text)
{
	ImGui::SameLine();
	ImGui::TextDisabled("(?)");
	add_tooltip(text);
}

void ImGuiRenderer::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;
	m_imgui_settings_window.set_render_window(render_window);
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
	draw_log_window();
	draw_render_window();

	ImGui::ShowDemoWindow();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiRenderer::rescale_ui()
{
	ImGuiIO& io = ImGui::GetIO();
	ImGuiViewport* viewport = ImGui::GetMainViewport();
	float windowDpiScale = viewport->DpiScale;

	// Scaling by the DPI -10% as judged more pleasing
	io.FontGlobalScale = windowDpiScale;
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
			auto dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, ImGuiSettingsWindow::BASE_SIZE / (renderer_width + ImGuiSettingsWindow::BASE_SIZE), nullptr, &dockspace_id);
			auto dock_id_bottom = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, ImGuiLogWindow::BASE_SIZE / (renderer_height + ImGuiLogWindow::BASE_SIZE), nullptr, &dockspace_id);

			// we now dock our windows into the docking node we made above
			ImGui::DockBuilderDockWindow(ImGuiLogWindow::TITLE, dock_id_bottom);
			ImGui::DockBuilderDockWindow(ImGuiSettingsWindow::TITLE, dock_id_left);
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
