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

	draw_header();
	draw_frame_sequence_rendering_panel();
	draw_camera_panel();
	draw_envmap_panel();

	ImGui::PopItemWidth();

	ImGui::End();
}

void ImGuiAnimationWindow::draw_header()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	RendererAnimationState& animation_state = m_renderer->get_animation_state();

	ImGui::SeparatorText("General settings");
	if (ImGui::Checkbox("Accumulate", &render_settings.accumulate))
	{
		m_render_window->set_render_dirty(true);

		if (!render_settings.accumulate)
		{
			m_render_window->get_application_settings()->auto_sample_per_frame = false;
			render_settings.samples_per_frame = 1;
		}
	}

	std::string animation_button_text = animation_state.do_animations ? "Disable animations" : "Enable animations";
	if (ImGui::Button(animation_button_text.c_str()))
		animation_state.do_animations = !animation_state.do_animations;
	if (m_renderer->get_render_settings().accumulate && !animation_state.is_rendering_frame_sequence)
	{
		ImGui::TreePush("Animations info tree");
		ImGui::Text("Warning: ");
		ImGuiRenderer::show_help_marker("Animations are not playing right now because\n"
					"accumulation is on. Nothing can move while accumulation\n"
					"is on unless you're rendering a frame sequence, in\n"
					"which case animations will step forward after a frame\n"
					"is rendered (converged according to the renderer settings).");
		ImGui::TreePop();
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiAnimationWindow::draw_frame_sequence_rendering_panel()
{
	if (ImGui::CollapsingHeader("Frame Sequence Rendering"))
	{
		RendererAnimationState& animation_state = m_renderer->get_animation_state();

		ImGui::TreePush("Frame Sequence Rendering Tree");

		ImGui::Text("Currently at frame %d / %d", animation_state.frames_rendered_so_far, animation_state.number_of_animation_frames);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		static int move_n_frames_forward = 0;
		ImGui::InputInt("Move N frames forward", &move_n_frames_forward); 
		ImGuiRenderer::show_help_marker("Advances all the animations N frames forward.");
		ImGui::SameLine();
		ImGui::BeginDisabled(animation_state.do_animations == false);
		if (ImGui::Button("Go!"))
		{
			bool can_step_backup = animation_state.can_step_animation;

			animation_state.can_step_animation = true;
			for (int i = 0; i < move_n_frames_forward; i++)
				m_renderer->step_animations(16.67f);

			animation_state.can_step_animation = can_step_backup;
			animation_state.frames_rendered_so_far += move_n_frames_forward;
			move_n_frames_forward = 0;

			m_render_window->set_render_dirty(true);
		}
		ImGui::EndDisabled();
		if (animation_state.do_animations == false)
			ImGuiRenderer::show_help_marker("Disabled because animations are not enabled right now.");

		if (ImGui::InputInt("Number of frames to render", &animation_state.number_of_animation_frames))
			animation_state.reset();

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

		ImGui::TreePop();
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

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Rotation options");

			static float& rotation_value = camera_animation.m_rotation_value;
			static CameraRotationType& rotation_type = camera_animation.m_rotation_type;
			if (ImGui::RadioButton("##second_per_rotation", (int*)&rotation_type, 0))
				rotation_value = 8.0f;
			ImGui::SameLine();
			ImGui::BeginDisabled(rotation_type != 0);
			if (ImGui::SliderFloat("Rotation duration (seconds per 360 degrees)", &rotation_value, 2.0f, 10.0f))
				rotation_value = std::max(0.001f, rotation_value);
			ImGuiRenderer::show_help_marker("The camera will take that much time to rotate "
											"by 360 degrees. This is probably what you want "
											"for real time (no accumulation) camera animation.");
			ImGui::EndDisabled();

			if (ImGui::RadioButton("##degrees_per_frame", (int*)&rotation_type, 1))
				rotation_value = 1.0f;
			ImGui::SameLine();
			ImGui::BeginDisabled(rotation_type != 1);
			ImGui::SliderFloat("Rotation speed (degrees per frame)", &rotation_value, 0.0f, 90.0f);
			ImGuiRenderer::show_help_marker("The camera will rotate by the given degrees "
											"at each frame. This is probably what you want "
											"for frame sequence rendering.");
			ImGui::EndDisabled();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		ImGuiSettingsWindow::draw_camera_panel_static("Camera Settings", m_render_window, m_renderer);

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

		if (animate_envmap)
		{
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			ImGui::Text("Speeds are in degrees per second");
			ImGui::SliderFloat("Animation Speed X", &animation_speed_X, 0.0f, 360.0f);
			ImGui::SliderFloat("Animation Speed Y", &animation_speed_Y, 0.0f, 360.0f);
			ImGui::SliderFloat("Animation Speed Z", &animation_speed_Z, 0.0f, 360.0f);
		}

		ImGui::TreePop();
	}
}
