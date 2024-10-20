/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ImGui/ImGuiAnimationWindow.h"
#include "UI/RenderWindow.h"

#include "imgui.h"

const char* ImGuiAnimationWindow::TITLE = "Animation";

void ImGuiAnimationWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;

	m_renderer = m_render_window->get_renderer();
}

void ImGuiAnimationWindow::draw()
{
	ImGui::Begin(ImGuiAnimationWindow::TITLE);

	ImGui::PushItemWidth(16 * ImGui::GetFontSize());

	draw_general_settings();
	draw_camera_panel();
	draw_envmap_panel();

	ImGui::PopItemWidth();

	ImGui::End();
}

void ImGuiAnimationWindow::draw_general_settings()
{
	if (ImGui::CollapsingHeader("General Settings"))
	{
		RendererAnimationState& animation_state = m_renderer->get_animation_state();
		if (ImGui::InputInt("Number of animation frames", &animation_state.number_of_animation_frames))
			animation_state.reset();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		std::string animation_button_text = animation_state.do_animations ? "Stop animations" : "Start animations";
		if (ImGui::Button(animation_button_text.c_str()))
			animation_state.do_animations = !animation_state.do_animations;

		ImGui::BeginDisabled(!m_renderer->get_render_settings().accumulate);
		std::string start_rendering_animation_text = animation_state.is_rendering_frame_sequence ? "Stop rendering frame sequence" : "Start rendering frame sequence";
		if (ImGui::Button(start_rendering_animation_text.c_str()))
		{
			m_render_window->set_render_dirty(true);

			animation_state.is_rendering_frame_sequence = !animation_state.is_rendering_frame_sequence;
			animation_state.reset();
		}
		if (!m_renderer->get_render_settings().accumulate)
			ImGuiRenderer::show_help_marker("Feature disabled because accumulation is not enabled.");
		else
			ImGuiRenderer::show_help_marker("Starts rendering a sequence of frame. After each frame has "
											"converged (according to the various stopping conditions set in "
											"\"Settings -> Render Settings\"), a screenshot is dumped to "
											"the disk, the animations are step and the next frame starts "
											"rendering.");
		ImGui::EndDisabled();
		if (animation_state.is_rendering_frame_sequence)
		{
			ImGui::SameLine();
			ImGui::Text("Rendering... %d / %d frames", animation_state.frames_rendered_so_far, animation_state.number_of_animation_frames);
		}

		if (!m_renderer->get_render_settings().accumulate)
		{
			ImGui::TreePush("Enable accumulation for start rendering frame sequence tree");
			if (ImGui::Button("Enable accumulation"))
			{
				m_renderer->get_render_settings().accumulate = true;
				m_render_window->set_render_dirty(true);
			}
			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

void ImGuiAnimationWindow::draw_camera_panel()
{
	if (ImGui::CollapsingHeader("Camera"))
	{
		ImGui::TreePush("Camera animation tree");

		Camera& camera = m_renderer->get_camera();
		CameraAnimation& camera_animation = m_renderer->get_camera_animation();

		ImGui::Checkbox("Animate", &camera_animation.animate);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::SeparatorText("Transformation");
		if (ImGui::DragFloat3("Position", reinterpret_cast<float*>(&camera.m_translation)))
			m_render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::CollapsingHeader("Rotate around object"))
		{
			ImGui::TreePush("Rotate around object camera animation tree");

			static bool default_rotation_set = false;
			static int selected_object = 0;

			ImGui::Checkbox("Rotate around object during animation", &camera_animation.m_do_rotation_animation);
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			ImGui::Text("Rotate around object");
			if (ImGui::BeginListBox("##rotate_around_objects", ImVec2(-FLT_MIN, 7 * ImGui::GetTextLineHeightWithSpacing())))
			{
				const std::vector<std::string>& mesh_names = m_renderer->get_mesh_names();
				const std::vector<std::string>& material_names = m_renderer->get_material_names();
				for (int n = 0; n < mesh_names.size(); n++)
				{
					const bool is_selected = (selected_object == n);

					const std::string& mesh_name = mesh_names[n];
					const std::string& material_name = material_names[m_renderer->get_mesh_material_indices()[n]];
					std::string object_text = mesh_name + " (" + material_name + ")";
					if (ImGui::Selectable(object_text.c_str(), is_selected))
					{
						selected_object = n;

						float3 object_center = m_renderer->get_mesh_bounding_boxes()[n].get_center();
						camera_animation.m_rotate_around_point = glm::vec3(object_center.x, object_center.y, object_center.z);
					}

					// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndListBox();
			}

			if (!default_rotation_set)
			{
				if (m_renderer->get_mesh_bounding_boxes().size() > 0)
				{
					float3 default_rotate_around_point = m_renderer->get_mesh_bounding_boxes()[0].get_center();
					camera_animation.m_rotate_around_point = glm::vec3(default_rotate_around_point.x, default_rotate_around_point.y, default_rotate_around_point.z);

					default_rotation_set = true;
				}
			}

			if (ImGui::Button("Center camera on object"))
			{
				camera.look_at_object(m_renderer->get_mesh_bounding_boxes()[selected_object]);

				m_render_window->set_render_dirty(true);
			}

			ImGui::SeparatorText("Rotation options");

			static float& rotation_value = camera_animation.m_rotation_value;
			static CameraRotationType& rotation_type = camera_animation.m_rotation_type;
			if (ImGui::RadioButton("##second_per_rotation", (int*)&rotation_type, 0))
				rotation_value = 8.0f;
			ImGui::SameLine();
			ImGui::BeginDisabled(rotation_type != 0);
			if (ImGui::SliderFloat("Rotation duration (seconds per rotation)", &rotation_value, 2.0f, 10.0f))
				rotation_value = std::max(0.001f, rotation_value);
			ImGui::EndDisabled();

			if (ImGui::RadioButton("##degrees_per_frame", (int*)&rotation_type, 1))
				rotation_value = 1.0f;
			ImGui::SameLine();
			ImGui::BeginDisabled(rotation_type != 1);
			ImGui::SliderFloat("Rotation speed (degrees per frame)", &rotation_value, 0.0f, 90.0f);
			ImGui::EndDisabled();

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiAnimationWindow::draw_envmap_panel()
{
	if (ImGui::CollapsingHeader("Environment Map"))
	{
		ImGui::TreePush("Envmap animation window tree");

		bool& animate_envmap = m_renderer->get_envmap().animate;
		float& animation_speed_X = m_renderer->get_envmap().animation_speed_X;
		float& animation_speed_Y = m_renderer->get_envmap().animation_speed_Y;
		float& animation_speed_Z = m_renderer->get_envmap().animation_speed_Z;

		ImGui::Checkbox("Animate", &animate_envmap);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::Text("Speeds are in degrees per second");
		ImGui::SliderFloat("Animation Speed X", &animation_speed_X, 0.0f, 360.0f);
		ImGui::SliderFloat("Animation Speed Y", &animation_speed_Y, 0.0f, 360.0f);
		ImGui::SliderFloat("Animation Speed Z", &animation_speed_Z, 0.0f, 360.0f);

		ImGui::TreePop();
	}
}
