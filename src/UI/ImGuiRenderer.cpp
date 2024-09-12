/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "UI/ImGuiRenderer.h"
#include "UI/RenderWindow.h"

#include <chrono>
#include <unordered_map>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

ImGuiRenderer::ImGuiRenderer()
{
	ImGuiViewport* viewport = ImGui::GetMainViewport();
	float windowDpiScale = viewport->DpiScale;
	if (windowDpiScale > 1.0f)
		ImGui::GetStyle().ScaleAllSizes(windowDpiScale);
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

	m_application_settings = render_window->get_application_settings();
	m_renderer = render_window->get_renderer();
	m_render_window_denoiser = render_window->get_denoiser();
	m_render_window_perf_metrics = m_render_window->get_performance_metrics();

}

void ImGuiRenderer::draw_imgui_interface()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	ImGuiIO& io = ImGui::GetIO();
	// ImGui::ShowDemoWindow();

	ImGui::Begin("Settings");

	ImGui::Text("Render time: %.3fs", m_render_window->get_current_render_time() / 1000.0f);
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", render_settings.sample_number, m_render_window->get_samples_per_second(), m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);

	if (render_settings.has_access_to_adaptive_sampling_buffers())
	{
		unsigned int converged_count = m_renderer->get_status_buffer_values().pixel_converged_count;
		unsigned int total_pixel_count = m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y;

		bool can_print_convergence = false;
		can_print_convergence |= render_settings.sample_number > render_settings.adaptive_sampling_min_samples;
		can_print_convergence |= render_settings.stop_pixel_noise_threshold > 0.0f;

		if (can_print_convergence)
		{
			ImGui::Text("Pixels converged: %d / %d - %.4f%%", converged_count, total_pixel_count, static_cast<float>(converged_count) / total_pixel_count * 100.0f);

			// Adding some information on what noise threshold is being used
			std::string text = "Current noise threshold is: ";
			if (render_settings.enable_adaptive_sampling && render_settings.sample_number > render_settings.adaptive_sampling_min_samples)
			{
				if (render_settings.stop_pixel_noise_threshold > render_settings.adaptive_sampling_noise_threshold)
					// If the pixel noise threshold is stronger, then the displayed convergence counter
					// is going to be according to the stop noise threshold so that's what we're adding in the tooltip
					// there
					text += std::to_string(render_settings.stop_pixel_noise_threshold) + " (pixel noise threshold)";
				else
					text += std::to_string(render_settings.adaptive_sampling_noise_threshold) + " (adaptive sampling)";
			}
			else if (render_settings.stop_pixel_noise_threshold > 0.0f)
				text += std::to_string(render_settings.stop_pixel_noise_threshold) + " (pixel noise threshold)";
			ImGuiRenderer::show_help_marker(text);
		}
		else
		{
			ImGui::Text("Pixels converged: N/A");
			ImGuiRenderer::show_help_marker("Adaptive sampling hasn't kicked in yet... Convergence computation hasn't started.");
		}
	}
	else
	{
		ImGui::Text("Pixels converged: N/A");
		ImGuiRenderer::show_help_marker("Convergence is only computed when either adaptive sampling or the \"Pixel noise threshold\" render stopping condition is used.");
	}

	ImGui::Separator();

	if (ImGui::Button("Save viewport to PNG"))
		m_render_window->get_screenshoter()->write_to_png();

	ImGui::Separator();

	ImGui::PushItemWidth(16 * ImGui::GetFontSize());

	draw_render_settings_panel();
	draw_camera_panel();
	draw_environment_panel();
	draw_sampling_panel();
	draw_objects_panel();
	draw_denoiser_panel();
	draw_post_process_panel();
	draw_performance_settings_panel();
	draw_performance_metrics_panel();
	draw_debug_panel();

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiRenderer::draw_render_settings_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (!ImGui::CollapsingHeader("Render Settings"))
		return;
	ImGui::TreePush("Render settings tree");

	ImGui::SeparatorText("Viewport Settings");
	std::vector<const char*> items = { "- Default", "- Denoiser blend", "- Denoiser - Normals", "- Denoiser - Denoised normals", "- Denoiser - Albedo", "- Denoiser - Denoised albedo" };
	if (render_settings.has_access_to_adaptive_sampling_buffers())
		items.push_back("- Pixel convergence heatmap");

	int display_view_selected = m_render_window->get_display_view_system()->get_current_display_view_type();
	if (ImGui::Combo("Display View", &display_view_selected, items.data(), items.size()))
		m_render_window->get_display_view_system()->queue_display_view_change(static_cast<DisplayViewType>(display_view_selected));

	ImGui::BeginDisabled(m_application_settings->keep_same_resolution);
	float resolution_scaling_backup = m_application_settings->render_resolution_scale;
	if (ImGui::InputFloat("Resolution scale", &m_application_settings->render_resolution_scale))
	{
		float& resolution_scale = m_application_settings->render_resolution_scale;
		if (resolution_scale <= 0)
			resolution_scale = resolution_scaling_backup;

		m_render_window->change_resolution_scaling(resolution_scale);
		m_render_window->set_render_dirty(true);
	}
	ImGui::EndDisabled();

	if (ImGui::Checkbox("Keep same render resolution", &m_application_settings->keep_same_resolution))
	{
		if (m_application_settings->keep_same_resolution)
		{
			// Remembering the width and height we need to target
			m_application_settings->target_width = m_renderer->m_render_resolution.x;
			m_application_settings->target_height = m_renderer->m_render_resolution.y;
		}
	}

	ImGui::Separator();

	if (ImGui::Checkbox("Accumulate", &render_settings.accumulate))
	{
		m_render_window->set_render_dirty(true);

		if (!render_settings.accumulate)
		{
			m_render_window->get_application_settings()->auto_sample_per_frame = false;
			render_settings.samples_per_frame = 1;
		}
	}

	if (ImGui::InputInt("Samples per frame", &render_settings.samples_per_frame))
		// Clamping to 1
		render_settings.samples_per_frame = std::max(1, render_settings.samples_per_frame);
	ImGui::SameLine();
	ImGui::Checkbox("Auto", &m_application_settings->auto_sample_per_frame);
	if (m_application_settings->auto_sample_per_frame)
	{
		ImGui::TreePush("Target GPU framerate tree");
		if (ImGui::InputFloat("Target GPU framerate", &m_application_settings->target_GPU_framerate))
			// Clamping to 1 FPS because going below that is dangerous in terms of driver timeouts
			m_application_settings->target_GPU_framerate = std::max(1.0f, m_application_settings->target_GPU_framerate);
		ImGuiRenderer::show_help_marker("The samples per frame will be automatically adjusted such that the GPU"
				   " takes approximately 1000.0f / TargetFramerate milliseconds to complete"
				   " a frame. Useful to keep the GPU busy after almost all pixels have converged."
				   " Lowering this settings increases rendering efficiency but can cause camera"
				   " movements to be stuttery.");

		ImGui::TreePop();
	}
	if (ImGui::InputInt("Max bounces", &render_settings.nb_bounces))
	{
		// Clamping to 0 in case the user input a negative number of bounces	
		render_settings.nb_bounces = std::max(render_settings.nb_bounces, 0);
		m_render_window->set_render_dirty(true);
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (ImGui::CollapsingHeader("Render stopping condition"))
	{
		ImGui::TreePush("Stopping condition tree");
		{

			if (ImGui::InputInt("Max Sample Count", &m_application_settings->max_sample_count))
				m_application_settings->max_sample_count = std::max(m_application_settings->max_sample_count, 0);

			if (ImGui::InputFloat("Max Render Time (s)", &m_application_settings->max_render_time))
				m_application_settings->max_render_time = std::max(m_application_settings->max_render_time, 0.0f);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Pixel Stop Noise Threshold");
			ImGui::Checkbox("Use pixel stop noise threshold stopping condition", &render_settings.enable_pixel_stop_noise_threshold);
			ImGuiRenderer::show_help_marker("If enabled, stops the renderer after a certain proportion "
				"of pixels of the image have converged. \"converged\" is evaluated according to the "
				"threshold of the adaptive sampling if it is enabled. If adaptive sampling is not "
				"enabled, \"converged\" is defined by the \"Pixel noise threshold\" variance "
				"threshold below.");

			ImGui::BeginDisabled(!render_settings.enable_pixel_stop_noise_threshold);
			{
				if (ImGui::InputFloat("Pixel proportion", &render_settings.stop_pixel_percentage_converged))
					render_settings.stop_pixel_percentage_converged = std::max(0.0f, std::min(render_settings.stop_pixel_percentage_converged, 100.0f));
				ImGuiRenderer::show_help_marker("The proportion of pixels that need to have converge "
					"to the noise threshold for the rendering to stop. In percentage [0, 100].");
			}
			ImGui::EndDisabled();

			ImGui::BeginDisabled(render_settings.enable_adaptive_sampling || !render_settings.enable_pixel_stop_noise_threshold);
			{
				// Only letting the user manipulate the stop pixel noise threshold if adaptive sampling is not enabled
				// because if adaptive sampling is enabled, then the stop pixel noise threshold feature can only
				// be used to give a render stopping condition (after a certain proportion of pixels have converged).
				//
				// Said otherwise, if adaptive sampling is enabled, then we're not using the stop pixel noise threshold
				// at all so it doesn't need to be exposed to the user
				if (ImGui::InputFloat("Pixel noise threshold", &render_settings.stop_pixel_noise_threshold))
				{
					render_settings.stop_pixel_noise_threshold = std::max(0.0f, render_settings.stop_pixel_noise_threshold);

					m_render_window->set_render_dirty(true);
				}
				std::string pixel_noise_threshold_help_string = "Cannot be set lower than the adaptive sampling threshold. 0.0 to disable.";
				if (render_settings.enable_adaptive_sampling)
					pixel_noise_threshold_help_string += "\n\nDisabled because adaptive sampling is enabled. Both cannot be used at the same time.";
				ImGuiRenderer::show_help_marker(pixel_noise_threshold_help_string);

				if (ImGui::Button("Copy adaptive sampling's threshold"))
				{
					render_settings.stop_pixel_noise_threshold = render_settings.adaptive_sampling_noise_threshold;

					m_render_window->set_render_dirty(true);
				}
				std::string copy_button_help_string = "Copies the adaptive sampling variance threshold for the stop pixel noise threshold.";
				if (render_settings.enable_adaptive_sampling)
					copy_button_help_string += "\n\nDisabled because adaptive sampling is enabled. Both cannot be used at the same time.";
				ImGuiRenderer::show_help_marker(copy_button_help_string);
				ImGui::Dummy(ImVec2(0.0f, 20.0f));
			}
			ImGui::EndDisabled(); // render_settings.enable_adaptive_sampling

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

		}
		// Stopping condition tree
		ImGui::TreePop();
	}
	
	if (ImGui::CollapsingHeader("Light clamping"))
	{
		ImGui::TreePush("Light clamping tree");

		if (ImGui::SliderFloat("Direct ligthing contribution clamp", &render_settings.direct_contribution_clamp, 0.0f, 10.0f))
		{
			render_settings.direct_contribution_clamp = std::max(0.0f, render_settings.direct_contribution_clamp);
			m_render_window->set_render_dirty(true);
		}
		if (ImGui::SliderFloat("Envmap ligthing contribution clamp", &render_settings.envmap_contribution_clamp, 0.0f, 10.0f))
		{
			render_settings.envmap_contribution_clamp = std::max(0.0f, render_settings.envmap_contribution_clamp);
			m_render_window->set_render_dirty(true);
		}
		if (ImGui::SliderFloat("Indirect ligthing contribution clamp", &render_settings.indirect_contribution_clamp, 0.0f, 10.0f))
		{
			render_settings.indirect_contribution_clamp = std::max(0.0f, render_settings.indirect_contribution_clamp);
			m_render_window->set_render_dirty(true);
		}
		
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		// Light clamping tree
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Nested dielectrics"))
	{
		ImGui::TreePush("Nested dielectrics tree");

	 	std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_path_tracer_options();
		const char* items[] = { "- Automatic", "- With priorities" };
		if (ImGui::Combo("Nested dielectrics strategy", kernel_options->get_pointer_to_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY), items, IM_ARRAYSIZE(items)))
		{
			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		ImGui::TreePop();
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiRenderer::draw_camera_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (ImGui::CollapsingHeader("Camera"))
	{
		ImGui::TreePush("Camera tree");

		if (ImGui::Checkbox("Do ray jittering", &m_renderer->get_camera().do_jittering))
			m_render_window->set_render_dirty(true);

		ImGui::BeginDisabled(!render_settings.accumulate);
		ImGui::Checkbox("Render low resolution when interacting", &render_settings.allow_render_low_resolution);
		if (!render_settings.accumulate)
			add_tooltip("Cannot render at low resolution when not accumulating. If you want to render at "
				"a lower resolution, you can use the resolution scale in \"Render Settings\"for that.");
		ImGui::SliderInt("Render low resolution downscale", &render_settings.render_low_resolution_scaling, 1, 8);
		if (!render_settings.accumulate)
			add_tooltip("Cannot render at low resolution when not accumulating. If you want to render at "
				"a lower resolution, you can use the resolution scale in \"Render Settings\"for that.");
		ImGui::EndDisabled();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiRenderer::draw_environment_panel()
{
	bool render_made_piggy = false;

	if (ImGui::CollapsingHeader("Environment"))
	{
		ImGui::TreePush("Environment tree");

		bool has_envmap = m_renderer->has_envmap();
		render_made_piggy |= ImGui::RadioButton("None", ((int*)&m_renderer->get_world_settings().ambient_light_type), 0); ImGui::SameLine();
		render_made_piggy |= ImGui::RadioButton("Use uniform lighting", ((int*)&m_renderer->get_world_settings().ambient_light_type), 1); ImGui::SameLine();
		ImGui::BeginDisabled(!has_envmap);
		render_made_piggy |= ImGui::RadioButton("Use envmap lighting", ((int*)&m_renderer->get_world_settings().ambient_light_type), 2);
		if (!has_envmap)
			// Showing a tooltip for why the envmap button is disabled
			ImGuiRenderer::show_help_marker("No envmap loaded.");
		ImGui::EndDisabled();

		if (m_renderer->get_world_settings().ambient_light_type == AmbientLightType::UNIFORM)
		{
			render_made_piggy |= ImGui::ColorEdit3("Uniform light color", (float*)&m_renderer->get_world_settings().uniform_light_color, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
		}
		else if (m_renderer->get_world_settings().ambient_light_type == AmbientLightType::ENVMAP)
		{
			static float rota_X = 0.0f, rota_Y = 0.0f, rota_Z = 0.0f;
			bool rotation_changed;

			rotation_changed = false;
			rotation_changed |= ImGui::SliderFloat("Envmap rotation X", &rota_X, 0.0f, 1.0f);
			rotation_changed |= ImGui::SliderFloat("Envmap rotation Y", &rota_Y, 0.0f, 1.0f);
			rotation_changed |= ImGui::SliderFloat("Envmap rotation Z", &rota_Z, 0.0f, 1.0f);

			if (rotation_changed)
			{
				glm::mat4x4 rotation_matrix;

				// glm::orientate3 interprets the X, Y and Z angles we give it as a yaw/pitch/roll semantic.
				// 
				// The standard yaw/pitch/roll interpretation is:
				//	- Yaw for rotation around Z
				//	- Pitch for rotation around Y
				//	- Roll for rotation around X
				// 
				// but with a Z-up coordinate system. We want a Y-up coordinate system so
				// we want our Yaw to rotate around Y instead of Z (and our Pitch to rotate around Z).
				// 
				// This means that we need to reverse Y and Z.
				// 
				// See this picture for a visual aid on what we **don't** want (the z-up):
				// https://www.researchgate.net/figure/xyz-and-pitch-roll-and-yaw-systems_fig4_253569466
				rotation_matrix = glm::orientate3(glm::vec3(rota_X * 2.0f * M_PI, rota_Z * 2.0f * M_PI, rota_Y * 2.0f * M_PI));
				m_renderer->get_world_settings().envmap_rotation_matrix = *reinterpret_cast<float4x4*>(&rotation_matrix);
			}

			render_made_piggy |= rotation_changed;
			render_made_piggy |= ImGui::SliderFloat("Envmap intensity", (float*)&m_renderer->get_world_settings().envmap_intensity, 0.0f, 10.0f);
			ImGui::TreePush("Envmap intensity tree");
			render_made_piggy |= ImGui::Checkbox("Scale background intensity", (bool*)&m_renderer->get_world_settings().envmap_scale_background_intensity);
			ImGui::TreePop();
		}

		// Ensuring no negative light color
		m_renderer->get_world_settings().uniform_light_color.clamp(0.0f, 1.0e38f);

		ImGui::TreePop();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}

	if (render_made_piggy)
		m_render_window->set_render_dirty(true);
}

void ImGuiRenderer::draw_sampling_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_path_tracer_options();

	if (ImGui::CollapsingHeader("Sampling"))
	{
		ImGui::TreePush("Sampling tree");

		if (ImGui::CollapsingHeader("Adaptive sampling"))
		{

			ImGui::TreePush("Adaptive sampling tree");

			// Cannot use adaptive sampling without accumulation
			ImGui::BeginDisabled(!render_settings.accumulate);

			if (ImGui::Checkbox("Enable adaptive sampling", (bool*)&render_settings.enable_adaptive_sampling))
				m_render_window->set_render_dirty(true);
			if (!render_settings.accumulate)
				add_tooltip("Cannot use adaptive sampling when accumulation is not on.");

			float adaptive_sampling_noise_threshold_before = render_settings.adaptive_sampling_noise_threshold;
			ImGui::BeginDisabled(!render_settings.enable_adaptive_sampling);
			if (ImGui::InputInt("Adaptive sampling minimum samples", &render_settings.adaptive_sampling_min_samples))
				m_render_window->set_render_dirty(true);
			if (ImGui::InputFloat("Adaptive sampling noise threshold", &render_settings.adaptive_sampling_noise_threshold))
			{
				render_settings.adaptive_sampling_noise_threshold = std::max(0.0f, render_settings.adaptive_sampling_noise_threshold);
			
				m_render_window->set_render_dirty(true);
			}

			// !Cannot use adaptive sampling without accumulation
			ImGui::EndDisabled();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();

			// !render_settings.accumulate
			ImGui::EndDisabled();
		}

		if (ImGui::CollapsingHeader("Direct lighting"))
		{
			ImGui::TreePush("Direct lighting sampling tree");

			const char* items[] = { "- No direct light sampling", "- Uniform one light", "- BSDF Sampling", "- MIS (1 Light + 1 BSDF)", "- RIS BDSF + Light candidates", "- ReSTIR DI (Primary Hit Only) + RIS"};
			if (ImGui::Combo("Direct light sampling strategy", kernel_options->get_pointer_to_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			// Display additional widgets to control the parameters of the direct light
			// sampling strategy chosen (the number of candidates for RIS for example)
			switch (kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY))
			{
			case LSS_NO_DIRECT_LIGHT_SAMPLING:
				break;

			case LSS_UNIFORM_ONE_LIGHT:
				break;

			case LSS_MIS_LIGHT_BSDF:
				break;

			case LSS_RIS_BSDF_AND_LIGHT:
			{
				static bool use_visibility_ris_target_function = RISUseVisiblityTargetFunction;
				if (ImGui::Checkbox("Use visibility in RIS target function", &use_visibility_ris_target_function))
				{
					kernel_options->set_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, use_visibility_ris_target_function ? 1 : 0);
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::Checkbox("Use geometry term in RIS target function", &render_settings.ris_settings.geometry_term_in_target_function))
				{
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("RIS # of BSDF candidates", &render_settings.ris_settings.number_of_bsdf_candidates, 0, 32))
				{
					// Clamping to 0
					render_settings.ris_settings.number_of_bsdf_candidates = std::max(0, render_settings.ris_settings.number_of_bsdf_candidates);

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("RIS # of light candidates", &render_settings.ris_settings.number_of_light_candidates, 0, 128))
				{
					// Clamping to 0
					render_settings.ris_settings.number_of_light_candidates = std::max(0, render_settings.ris_settings.number_of_light_candidates);

					m_render_window->set_render_dirty(true);
				}

				break;
			}

			case LSS_RESTIR_DI:
			{
				display_ReSTIR_DI_bias_status(kernel_options);

				ImGui::SeparatorText("General Settings");
				ImGui::TreePush("ReSTIR DI - General Settings Tree");
				{
					// Whether or not to use the visibility term in the target function used for
					// resampling in ReSTIR DI (applies to all passes: initial candidates, temporal reuse, spatial reuse)
					static bool use_target_function_visibility = ReSTIR_DI_TargetFunctionVisibility;
					if (ImGui::Checkbox("Use visibility in target function (in all passes)", &use_target_function_visibility))
					{
						kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_TARGET_FUNCTION_VISIBILITY, use_target_function_visibility ? 1 : 0);
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}

					if (ImGui::SliderInt("M-cap", &render_settings.restir_di_settings.m_cap, 0, 96))
					{
						render_settings.restir_di_settings.m_cap = std::max(0, render_settings.restir_di_settings.m_cap);
						m_render_window->set_render_dirty(true);
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));

					// Backup values to remember what the heuristic threshold was before it was disabled
					// The normal similarity backup is the angle in degrees
					static float normal_similarity_heuristic_backup = render_settings.restir_di_settings.normal_similarity_angle_degrees;
					static float plane_distane_heuristic_backup = render_settings.restir_di_settings.plane_distance_threshold;
					static float roughness_similarity_heuristic_backup = render_settings.restir_di_settings.roughness_similarity_threshold;

					static bool use_heuristics_at_all = true;
					if (ImGui::Checkbox("Use Heuristics for neighbor rejection", &use_heuristics_at_all))
					{
						if (!use_heuristics_at_all)
						{
							// "Disabling" the heuristics by setting "impossible" values so that the heuristics
							// tests in the shader always pass
							render_settings.restir_di_settings.normal_similarity_angle_precomp = -1.0f;
							render_settings.restir_di_settings.plane_distance_threshold = 1.0e35f;
							render_settings.restir_di_settings.roughness_similarity_threshold = 1000.0f;
						}
						else
						{
							// Restoring heuristics to their backup values
							render_settings.restir_di_settings.normal_similarity_angle_precomp = std::cos(normal_similarity_heuristic_backup * M_PI / 180.0f);
							render_settings.restir_di_settings.plane_distance_threshold = plane_distane_heuristic_backup;
							render_settings.restir_di_settings.roughness_similarity_threshold = roughness_similarity_heuristic_backup;
						}

						m_render_window->set_render_dirty(true);
					}
					ImGuiRenderer::show_help_marker("Using heuristics to reject neighbor that are too dissimilar (in "
						"terms of normal orientation/roughnes/... to the pixel doing the resampling "
						"can help reduce variance. It also reduces bias but never removes it "
						"completely, it just makes it less obvious.");
					
					if (use_heuristics_at_all)
					{
						ImGui::TreePush("ReSTIR DI Heursitics Tree");




						static bool do_normal_similarity_test = true;
						if (ImGui::Checkbox("Use Normal Similarity Heuristic", &do_normal_similarity_test))
						{
							if (do_normal_similarity_test)
							{
								// The user re-enabled the normal heuristic, restoring the threshold to its backup value
								render_settings.restir_di_settings.normal_similarity_angle_degrees = normal_similarity_heuristic_backup;
								render_settings.restir_di_settings.normal_similarity_angle_precomp = std::cos(normal_similarity_heuristic_backup * M_PI / 180.0f);
							}
							else
								// If not using the heuristic, setting the threshold to 0.0f so that the normal heuristic always passes
								render_settings.restir_di_settings.normal_similarity_angle_precomp = -1.0f;

							m_render_window->set_render_dirty(true);
						}

						if (do_normal_similarity_test)
						{
							if (ImGui::SliderFloat("Normal Similarity Threshold", &render_settings.restir_di_settings.normal_similarity_angle_degrees, 0.0f, 360.0f, "%.3f deg", ImGuiSliderFlags_AlwaysClamp))
							{
								render_settings.restir_di_settings.normal_similarity_angle_precomp = std::cos(render_settings.restir_di_settings.normal_similarity_angle_degrees * M_PI / 180.0f);

								// Backuping the value for later if user disables the normal similarity
								normal_similarity_heuristic_backup = render_settings.restir_di_settings.normal_similarity_angle_degrees;

								m_render_window->set_render_dirty(true);
							}
						}




						static bool do_plane_distance_test = true;
						if (ImGui::Checkbox("Use Plane Distance Heuristic", &do_plane_distance_test))
						{
							if (do_plane_distance_test)
								// Restoring the value of the plane distance heuristic
								render_settings.restir_di_settings.plane_distance_threshold = plane_distane_heuristic_backup;
							else
								// Setting a super high value so that the plane distance heuristic always passes if we're not using it
								render_settings.restir_di_settings.plane_distance_threshold = 1.0e35f;

							m_render_window->set_render_dirty(true);
						}

						if (do_plane_distance_test)
						{
							if (ImGui::SliderFloat("Plane Distance Threshold", &render_settings.restir_di_settings.plane_distance_threshold, 0.0f, 1.0f))
							{
								plane_distane_heuristic_backup = render_settings.restir_di_settings.plane_distance_threshold;

								m_render_window->set_render_dirty(true);
							}
						}




						static bool do_roughness_heuristic = true;
						if (ImGui::Checkbox("Use Roughness Heuristic", &do_roughness_heuristic))
						{
							if (do_roughness_heuristic)
								// The user re-enabled the heuristic so we're restoring its value
								render_settings.restir_di_settings.roughness_similarity_threshold = roughness_similarity_heuristic_backup;
							else
								// Basically disabling the heuristic by setting a very high tolerance for the difference in roughness
								// so that the heuristic always passes
								render_settings.restir_di_settings.roughness_similarity_threshold = 1000.0f;

							m_render_window->set_render_dirty(true);
						}
						if (do_roughness_heuristic)
						{
							if (ImGui::SliderFloat("Roughness Threshold", &render_settings.restir_di_settings.roughness_similarity_threshold, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
							{
								roughness_similarity_heuristic_backup = render_settings.restir_di_settings.roughness_similarity_threshold;

								m_render_window->set_render_dirty(true);
							}
						}



						// ReSTIR DI Heursitics Tree
						ImGui::TreePop();
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop();
				}

				ImGui::SeparatorText("Initial Candidates Pass");
				ImGui::TreePush("ReSTIR DI - Initial Candidate Pass Tree");
				{
					if (ImGui::SliderInt("# of BSDF initial candidates", &render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates, 0, 32))
					{
						// Clamping to 0
						render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates= std::max(0, render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates);

						m_render_window->set_render_dirty(true);
					}

					if (ImGui::SliderInt("# of initial light candidates", &render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates, 0, 128))
					{
						// Clamping to 0
						render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates = std::max(0, render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates);

						m_render_window->set_render_dirty(true);
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop();
				}

				ImGui::SeparatorText("Visibility Reuse Pass");
				ImGui::TreePush("ReSTIR DI - Visibility Reuse Pass Tree");
				{
					static bool do_visibility_reuse = ReSTIR_DI_DoVisibilityReuse;
					if (ImGui::Checkbox("Do visibility reuse", &do_visibility_reuse))
					{
						kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE, do_visibility_reuse ? 1 : 0);
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop();
				}				

				ImGui::SeparatorText("Temporal Reuse Pass");
				ImGui::TreePush("ReSTIR DI - Temporal Reuse Pass Tree");
				{
					if (ImGui::Checkbox("Do Temporal Reuse", &render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass))
						m_render_window->set_render_dirty(true);

					if (render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
					{
						// Same line as "Do Temporal Reuse"
						ImGui::SameLine();
						if (ImGui::Button("Reset Temporal Reservoirs"))
						{
							render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested = true;
							m_render_window->set_render_dirty(true);
						}

						bool last_frame_g_buffer_needed = true;
						last_frame_g_buffer_needed &= !render_settings.accumulate;
						last_frame_g_buffer_needed &= render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass;

						if (ImGui::Checkbox("Use Last Frame G-Buffer", &render_settings.restir_di_settings.temporal_pass.use_last_frame_g_buffer))
							m_render_window->set_render_dirty(true);
						ImGuiRenderer::show_help_marker("For complete unbiasedness with camera motion, the G-buffer of the previous "
							"frame is required. This however comes at a VRAM cost which we may not want to pay. "
							"This is especially true when accumulating frames with a still camera in which case "
							"there is no motion meaning that the G-buffer of the previous frame isn't needed "
							"and can be freed from VRAM.");

						if (ImGui::SliderInt("Max temporal neighbor search count", &render_settings.restir_di_settings.temporal_pass.max_neighbor_search_count, 0, 16))
						{
							// Clamping
							render_settings.restir_di_settings.temporal_pass.max_neighbor_search_count = std::max(0, render_settings.restir_di_settings.temporal_pass.max_neighbor_search_count);

							m_render_window->set_render_dirty(true);
						}

						if (ImGui::SliderInt("Temporal neighbor search radius", &render_settings.restir_di_settings.temporal_pass.neighbor_search_radius, 0, 16))
						{
							// Clamping
							render_settings.restir_di_settings.temporal_pass.neighbor_search_radius = std::max(0, render_settings.restir_di_settings.temporal_pass.neighbor_search_radius);

							m_render_window->set_render_dirty(true);
						}
					}
						
					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop();
				}

				ImGui::SeparatorText("Spatial Reuse Pass");
				ImGui::TreePush("ReSTIR DI - Spatial Reuse Pass Tree");
				{
					if (ImGui::Checkbox("Do Spatial Reuse", &render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass))
						m_render_window->set_render_dirty(true);

					if (render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
					{
						if (ImGui::SliderInt("Spatial Reuse Pass Count", &render_settings.restir_di_settings.spatial_pass.number_of_passes, 1, 8))
						{
							// Clamping
							render_settings.restir_di_settings.spatial_pass.number_of_passes = std::max(1, render_settings.restir_di_settings.spatial_pass.number_of_passes);

							m_render_window->set_render_dirty(true);
						}

						if (ImGui::SliderInt("Spatial Reuse Radius (px)", &render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius, 1, 64))
						{
							// Clamping
							render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius = std::max(1, render_settings.restir_di_settings.spatial_pass.spatial_reuse_radius);

							m_render_window->set_render_dirty(true);
						}

						if (ImGui::SliderInt("Neighbor Reuse Count", &render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count, 1, 64))
						{
							// Clamping
							render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count = std::max(1, render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count);

							m_render_window->set_render_dirty(true);
						}

						if (ImGui::Checkbox("Neighbor Samples Random Rotation", &render_settings.restir_di_settings.spatial_pass.do_neighbor_rotation))
							m_render_window->set_render_dirty(true);
						ImGuiRenderer::show_help_marker("If checked, spatial neighbors sampled (using the Hammersley point set) will be randomly rotated. Because neighbor locations are generated with a Hammersley point set (deterministic), not rotating them results in every pixel of every rendered image reusing the same neighbor locations which can decrease reuse efficiency.");

						ImGui::BeginDisabled(!render_settings.enable_adaptive_sampling);
						if (ImGui::Checkbox("Allow Reuse of Converged Neighbors", &render_settings.restir_di_settings.spatial_pass.allow_converged_neighbors_reuse))
							m_render_window->set_render_dirty(true);
						std::string reuse_of_converged_neighbors_help = "If checked, then the spatial reuse passes are allowed "
							"to reuse from neighboring pixels which have converged (and thus neighbors that "
							"are not being sampled anymore = neighbors whose reservoirs do not evolve anymore). "
							"This improves performance but at the cost of bias when non-converged "
							"pixels try to reuse from converged pixels. The bias will thus typically manifest "
							"on the parts of the image that are the hardest to render.";
						if (!render_settings.enable_adaptive_sampling)
							reuse_of_converged_neighbors_help += "\n\nDisabled because adaptive sampling isn't enabled.";
						ImGuiRenderer::show_help_marker(reuse_of_converged_neighbors_help);
						if (render_settings.restir_di_settings.spatial_pass.allow_converged_neighbors_reuse)
						{
							if (ImGui::SliderFloat("Converged Neighbor Reuse Probability", &render_settings.restir_di_settings.spatial_pass.converged_neighbor_reuse_probability, 0.0f, 1.0f))
								m_render_window->set_render_dirty(true);
							ImGuiRenderer::show_help_marker("Allows trading bias for rendering performance by spatially reusing converged neighbors only with a certain probability instead of never / always."
								"\n\n 0.0 nevers reuses converged neighbors. No bias but performance impact."
								"\n\n 1.0 always reuses converged neighbors. Biased but no performance impact.");
						}
						ImGui::EndDisabled();
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::SeparatorText("Bias correction");

					const char* bias_correction_mode_items[] = {
						"- 1/M Weights (Biased)",
						"- 1/Z Weights (Unbiased)",
						"- MIS-like Weights (Unbiased)",
						"- MIS Weights GBH (Unbiased)",
						"- Pairwise MIS Weights (Unbiased)",
						"- Pairwise MIS Weights Defensive (Unbiased)",
					};
					if (ImGui::Combo("Bias Correction Weights", kernel_options->get_pointer_to_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS), bias_correction_mode_items, IM_ARRAYSIZE(bias_correction_mode_items)))
					{
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}
					ImGuiRenderer::show_help_marker("What weights to use to resample reservoirs");

					bool disable_confidence_weights = kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
						|| kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z;

					ImGui::BeginDisabled(disable_confidence_weights);
					if (ImGui::Checkbox("Use Confidence Weights", &render_settings.restir_di_settings.use_confidence_weights))
						m_render_window->set_render_dirty(true);
					std::string confidence_weight_help_string = "Whether or not to use confidence weights when resampling the samples. Confidence weights allow proper temporal reuse.";
					if (disable_confidence_weights)
						confidence_weight_help_string += "\n\nDisabled because 1/M or 1/Z weights use confidence weights by design.";
					ImGuiRenderer::show_help_marker(confidence_weight_help_string);
					ImGui::EndDisabled();

					// No visibility bias correction for 1/M weights
					bool bias_correction_visibility_disabled = kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_M;
					static bool bias_correction_use_visibility = ReSTIR_DI_BiasCorrectionUseVisiblity;
					ImGui::BeginDisabled(bias_correction_visibility_disabled);
					if (ImGui::Checkbox("Use visibility in bias correction", &bias_correction_use_visibility))
					{
						kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY, bias_correction_use_visibility ? 1 : 0);
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}
					if (bias_correction_visibility_disabled)
						ImGuiRenderer::show_help_marker("Visibility bias correction cannot be used with 1/M weights.");
					ImGui::EndDisabled();

					// Only 1/Z or pairwise MIS weights need that option to be fully unbiased
					// Also, we only need this with one more than 1 spatial reuse pass
					bool disable_raytrace_spatial_reuse_reservoirs = !(
						kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
						|| kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
						|| kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE)
						|| render_settings.restir_di_settings.spatial_pass.number_of_passes == 1 || !render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass;
					static bool raytrace_spatial_reuse_reservoirs = ReSTIR_DI_RaytraceSpatialReuseReservoirs;
					ImGui::BeginDisabled(disable_raytrace_spatial_reuse_reservoirs);
					if (ImGui::Checkbox("Ray-trace spatial reuse reservoirs", &raytrace_spatial_reuse_reservoirs))
					{
						kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_RAYTRACE_SPATIAL_REUSE_RESERVOIR, raytrace_spatial_reuse_reservoirs ? 1 : 0);
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}
					ImGui::EndDisabled();
					std::string ray_trace_spatial_reuse_reservoirs_string = "Because of some optimizations done in the spatial reuse pass, "
						"multiple spatial reuse passes may be biased with some weighting schemes. Ray-tracing a visibility "
						"ray at the end of each spatial reuse pass (if more than 1 pass) is then necessary (even with that additional ray, "
						"the performance boost is net positive) to ensure unbiasedness. The introduced bias is usually pretty small so disabling "
						"this option for a small performance boost may be worth it";
					if (disable_raytrace_spatial_reuse_reservoirs)
					{
						if (!render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
							ray_trace_spatial_reuse_reservoirs_string += "\n\nDisabled because spatial reuse is not enabled";
						else if (render_settings.restir_di_settings.spatial_pass.number_of_passes == 1)
							ray_trace_spatial_reuse_reservoirs_string += "\n\nDisabled because there is only 1 spatial reuse pass. Bias arises with more than 1 spatial reuse pass.";
						else
							ray_trace_spatial_reuse_reservoirs_string += "\n\nDisabled because not using either 1/Z or pairwise bias correction weights";
					}
					ImGuiRenderer::show_help_marker(ray_trace_spatial_reuse_reservoirs_string);

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop();
				}

				break;
			}

			default:
				break;
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Envmap lighting"))
		{
			ImGui::TreePush("Envmap sampling tree");

			const char* items[] = { "- No envmap importance sampling", "- Importance Sampling - Binary Search" };
			if (ImGui::Combo("Envmap sampling strategy", kernel_options->get_pointer_to_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiRenderer::display_ReSTIR_DI_bias_status(std::shared_ptr<GPUKernelCompilerOptions> kernel_options)
{
	ImGui::Text("Status: "); ImGui::SameLine();

	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	std::vector<std::string> bias_reasons;
	std::vector<std::string> hover_explanations;
	if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_M)
	{
		bias_reasons.push_back("- 1/M biased weights");
		hover_explanations.push_back("1/M weights do not take the number of neighbors that "
			"could have produced the resampled sample into account.This leads to darkening "
			"bias because we're not weighting our picked sample as if it could have been "
			"produced by M neighbors whereas less neighbors than that could have actually produced it.");
	}

	if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE) == KERNEL_OPTION_TRUE
		&& kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY) == KERNEL_OPTION_FALSE)
	{
		bias_reasons.push_back("- Visibility reuse without visibility in bias correction");
		hover_explanations.push_back("When using the visibility reuse pass at the end of the "
			"initial candidates sampling pass, light samples that are occluded are discarded.\n"
			"Temporal & spatial reuse pass will then only resample on unoccluded samples.\n"
			"If not accounting for visibility when counting valid neighbors, we may determine "
			"that a neighbor could have produced the picked sample when actually, it couldn't "
			"because from the neighbor's point of view, the sample could have been occluded "
			"(visibility reuse pass).\n"
			"This overestimates the number of valid neighbors and results in darkening.\n\n");
	}

	if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE) == KERNEL_OPTION_FALSE 
		&& kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_FALSE
	    && kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY) == KERNEL_OPTION_TRUE
		&& kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z)
	{
		bias_reasons.push_back("- Visibility in bias correction without visibility reuse");
		hover_explanations.push_back("When taking visibility into account in the counting of "
			"valid neighbors, we're going to assume that if the picked sample (from resampling "
			"the neighbors) is occluded from the neighbor's point of view, then that neighbor "
			"couldn't have produced that sample.\n\n"
			"However, that's incorrect.\n\n"
			"The initial candidate sampling pass doesn't take visibility into account and can "
			"thus produce occluded samples. Without the visibility reuse pass (or visibility used "
			"directly in the target function), this statement stays true.\n"
			"This means that \"a sample that is occluded from the neighbor's point of view\" could actually "
			"have been produced.\n"
			"We are then underestimating the number of valid neighbors that could have produced "
			"our sample and we end up with brightening bias.\n"
			"This is only an issue with 1/Z weights because MIS-like and proper MIS weights do "
			"not blindly overweight a sample as 1/Z does (and then hopes that we divide by Z accordingly).");
	}

	if (render_settings.enable_adaptive_sampling 
		&& render_settings.restir_di_settings.spatial_pass.allow_converged_neighbors_reuse 
		&& render_settings.restir_di_settings.spatial_pass.converged_neighbor_reuse_probability > 0.0f)
	{
		bias_reasons.push_back("- Adaptive Sampling + \"Allow Reuse of Converged Neighbors\"");
		hover_explanations.push_back("Adaptive sampling disables the sampling of some pixels. The "
			"spatial reuse pass then reuses from neighbors that do not evolve anymore (if they've "
			"been disabled by adaptive sampling) and that causes some slight convergence issues, "
			"especially on parts of the image where adaptive sampling does the more work. This "
			"manifest as bias on the hardest-to-render parts of the scene.");
	}

	if (!bias_reasons.empty())
	{
		ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Biased");
		ImGui::TreePush("Bias reasons");

		for (int i = 0; i < bias_reasons.size(); i++)
		{
			ImGui::Text("%s", bias_reasons[i].c_str());
			ImGuiRenderer::show_help_marker(hover_explanations[i].c_str());

		}
		ImGui::TreePop();

	}
	else
		ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Unbiased");
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiRenderer::draw_objects_panel()
{
	if (!ImGui::CollapsingHeader("Objects"))
		return;
	ImGui::TreePush("Objects tree");

	std::vector<RendererMaterial> materials = m_renderer->get_materials();
	std::vector<std::string> material_names = m_renderer->get_material_names();

	bool material_changed = false;
	static int currently_selected_material = 0;

	std::vector<const char*> items = { "- None", "- Lambertian BRDF", "- Oren Nayar BRDF", "- Disney BSDF" };
	if (ImGui::Combo("All Objects BSDF Override", m_renderer->get_path_tracer_options()->get_pointer_to_macro_value(GPUKernelCompilerOptions::BSDF_OVERRIDE), items.data(), items.size()))
	{
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);
	}
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	if (ImGui::CollapsingHeader("All objects"))
	{
		ImGui::TreePush("All objects tree");

		if (ImGui::BeginListBox("All objects", ImVec2(-FLT_MIN, 7 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (int n = 0; n < materials.size(); n++)
			{
				const bool is_selected = (currently_selected_material == n);
				if (ImGui::Selectable(material_names[n].c_str(), is_selected))
					currently_selected_material = n;

				// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndListBox();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Emissive objects"))
	{
		ImGui::TreePush("Emissive objects tree");

		if (ImGui::BeginListBox("Emissive objects", ImVec2(-FLT_MIN, 7 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (int n = 0; n < materials.size(); n++)
			{
				if (!materials[n].is_emissive())
					continue;

				const bool is_selected = (currently_selected_material == n);
				if (ImGui::Selectable(material_names[n].c_str(), is_selected))
					currently_selected_material = n;

				// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndListBox();
		}

		ImGui::TreePop();
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (materials.size() > 0)
	{
		std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_path_tracer_options();
		RendererMaterial& material = materials[currently_selected_material];

		ImGui::PushItemWidth(28 * ImGui::GetFontSize());

		ImGui::Text("%s", material_names[currently_selected_material].c_str());
		material_changed |= ImGui::ColorEdit3("Base color", (float*)&material.base_color);
		material_changed |= ImGui::SliderFloat("Subsurface", &material.subsurface, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Specular", &material.specular, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Specular tint strength", &material.specular_tint, 0.0f, 1.0f);
		material_changed |= ImGui::ColorEdit3("Specular color", (float*)&material.specular_color);
		material_changed |= ImGui::SliderFloat("Roughness", &material.roughness, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Anisotropic", &material.anisotropic, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Anisotropic rotation", &material.anisotropic_rotation, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Sheen", &material.sheen, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Sheen tint strength", &material.sheen_tint, 0.0f, 1.0f);
		material_changed |= ImGui::ColorEdit3("Sheen color", (float*)&material.sheen_color);
		material_changed |= ImGui::SliderFloat("Clearcoat", &material.clearcoat, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Clearcoat roughness", &material.clearcoat_roughness, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Clearcoat IOR", &material.clearcoat_ior, 0.0f, 5.0f);
		material_changed |= ImGui::SliderFloat("IOR", &material.ior, 0.0f, 5.0f);
		ImGui::Separator();
		material_changed |= ImGui::SliderFloat("Transmission", &material.specular_transmission, 0.0f, 1.0f);
		material_changed |= ImGui::SliderFloat("Absorption distance", &material.absorption_at_distance, 0.0f, 20.0f);
		material_changed |= ImGui::ColorEdit3("Absorption color", (float*)&material.absorption_color);
		unsigned short int zero = 0, eight = 8;
		ImGui::BeginDisabled(material.specular_transmission == 0.0f || kernel_options->get_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITIES);
		material_changed |= ImGui::SliderScalar("Dielectric priority", ImGuiDataType_U16, &material.dielectric_priority, &zero, &eight);
		ImGui::EndDisabled();
		material_changed |= ImGui::ColorEdit3("Emission", (float*)&material.emission, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);

		ImGui::PopItemWidth();

		ImGui::Separator();

		if (material_changed)
		{
			material.make_safe();
			material.precompute_properties();

			m_renderer->update_materials(materials);
			m_render_window->set_render_dirty(true);
		}
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiRenderer::draw_denoiser_panel()
{
	if (!ImGui::CollapsingHeader("Denoiser"))
		return;
	ImGui::TreePush("Denoiser tree");

	if (ImGui::Checkbox("Enable denoiser", &m_application_settings->enable_denoising))
	{
		m_render_window->get_display_view_system()->queue_display_view_change(m_application_settings->enable_denoising ? DisplayViewType::DENOISED_BLEND : DisplayViewType::DEFAULT);
		m_application_settings->denoiser_settings_changed = true;
	}
	ImGui::BeginDisabled(!m_application_settings->enable_denoising);
	if (ImGui::CollapsingHeader("AOVs"))
	{
		ImGui::TreePush("Denoiser AOVs Tree");
		if (ImGui::Checkbox("Use albedo AOV", &m_application_settings->denoiser_use_albedo))
		{
			m_application_settings->denoiser_settings_changed = true;

			m_render_window_denoiser->set_use_albedo(m_application_settings->denoiser_use_albedo);
			if (!m_application_settings->denoiser_use_albedo)
			{
				// We're forcing the use of normals AOV off here because it seems like OIDN doesn't support normal
				// AOV without also using albedo AOV (at least I got some oidn::Exception when I tried
				// using the normals without the albedo).
				// TODO this may have to do with wrong HIP buffers being used. Try this out again after we're using proper HIP buffers
				m_application_settings->denoiser_use_normals = false;
				m_render_window_denoiser->set_use_normals(false);
			}

			m_render_window_denoiser->finalize();
		}
		ImGui::SameLine();
		if (ImGui::Checkbox("Denoise albedo", &m_application_settings->denoiser_denoise_albedo))
		{
			m_application_settings->denoiser_settings_changed = true;

			m_render_window_denoiser->set_denoise_albedo(m_application_settings->denoiser_denoise_albedo);
			m_render_window_denoiser->finalize();
		}
		ImGui::BeginDisabled(!m_application_settings->denoiser_use_albedo);
		if (ImGui::Checkbox("Use normals AOV", &m_application_settings->denoiser_use_normals))
		{
			m_application_settings->denoiser_settings_changed = true;

			m_render_window_denoiser->set_use_normals(m_application_settings->denoiser_use_normals);
			m_render_window_denoiser->finalize();
		}
		ImGui::SameLine();
		if (ImGui::Checkbox("Denoise normals", &m_application_settings->denoiser_denoise_normals))
		{
			m_application_settings->denoiser_settings_changed = true;

			m_render_window_denoiser->set_denoise_normals(m_application_settings->denoiser_denoise_normals);
			m_render_window_denoiser->finalize();
		}
		ImGui::EndDisabled();
		ImGui::TreePop();
	}
	ImGui::Checkbox("Only denoise when rendering is done", &m_application_settings->denoise_when_rendering_done);
	ImGui::SliderInt("Denoise Sample Skip", &m_application_settings->denoiser_sample_skip, 1, 128);
	ImGui::SliderFloat("Denoiser blend", &m_application_settings->denoiser_blend, 0.0f, 1.0f);
	ImGui::EndDisabled();

	ImGui::Text("Denoising time: %.3fms", m_application_settings->last_denoised_duration / 1000.0f);

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiRenderer::draw_post_process_panel()
{
	if (!ImGui::CollapsingHeader("Post-processing"))
		return;
	ImGui::TreePush("Post-processing tree");

	ImGui::Checkbox("Do tonemapping", &m_application_settings->do_tonemapping);
	ImGui::InputFloat("Gamma", &m_application_settings->tone_mapping_gamma);
	ImGui::InputFloat("Exposure", &m_application_settings->tone_mapping_exposure);

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiRenderer::draw_performance_settings_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (!ImGui::CollapsingHeader("Performance Settings"))
		return;

	ImGui::TreePush("Performance settings tree");

	ImGui::Text("Device: %s", m_renderer->get_device_properties().name);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_path_tracer_options();
	HardwareAccelerationSupport hwi_supported = m_renderer->device_supports_hardware_acceleration();

	static bool use_hardware_acceleration = kernel_options->has_macro("__USE_HWI__");
	ImGui::BeginDisabled(hwi_supported != HardwareAccelerationSupport::SUPPORTED);
	if (ImGui::Checkbox("Use ray tracing hardware acceleration", &use_hardware_acceleration))
	{
		if (use_hardware_acceleration)
			kernel_options->set_macro_value("__USE_HWI__", 1);
		else
			kernel_options->remove_macro("__USE_HWI__");

		m_renderer->recompile_kernels();
	}
	ImGui::EndDisabled();

	// Printing a custom tooltip depending on whether or not we support hardware acceleration
	// and, if not supported, why we don't support it 
	switch (hwi_supported)
	{
	case SUPPORTED:
		ImGuiRenderer::show_help_marker("Whether or not to enable hardware accelerated ray tracing (bbox & triangle intersections)");
		break;

	case AMD_UNSUPPORTED:
		ImGuiRenderer::show_help_marker("Hardware accelerated ray tracing is only supported on RDNA2+ AMD GPUs.");
		break;

	case NVIDIA_UNSUPPORTED:
		ImGuiRenderer::show_help_marker("HIPRT cannot access NVIDIA's proprietary hardware accelerated ray-tracing. Hardware ray-tracing unavailable.");
		break;
	}

	if (ImGui::InputFloat("GPU Stall Percentage", &m_application_settings->GPU_stall_percentage))
		m_application_settings->GPU_stall_percentage = std::max(0.0f, std::min(m_application_settings->GPU_stall_percentage, 99.9f));
	ImGuiRenderer::show_help_marker("How much percent of the time the GPU will be forced to be idle (not rendering anything)."
		" This feature is basically only meant for GPUs that get too hot to avoid burning your GPUs during long renders if you have"
		" time to spare.");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::TreePop();
}

void ImGuiRenderer::draw_performance_metrics_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (!ImGui::CollapsingHeader("Performance Metrics"))
		return;

	ImGui::TreePush("Performance metrics tree");

	ImGui::Text("Device: %s", m_renderer->get_device_properties().name);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (ImGui::Button("Apply benchmark settings"))
	{
		render_settings.freeze_random = true;
		render_settings.enable_adaptive_sampling = false;
		m_application_settings->auto_sample_per_frame = false;
		render_settings.samples_per_frame = 1;

		m_render_window->set_render_dirty(true);
	}
	if (ImGui::Checkbox("Freeze random", (bool*)&render_settings.freeze_random))
		m_render_window->set_render_dirty(true);

	bool rolling_window_size_changed = false;
	int rolling_window_size = m_render_window_perf_metrics->get_window_size();
	ImGui::Text("Measures Window Size"); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("25", &rolling_window_size, 25); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("100", &rolling_window_size, 100); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("250", &rolling_window_size, 250); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("1000", &rolling_window_size, 1000);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	if (rolling_window_size_changed)
		m_render_window_perf_metrics->resize_window(rolling_window_size);

	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::CAMERA_RAYS_FUNC_NAME, "Camera rays pass");
	if (m_renderer->get_path_tracer_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::RESTIR_DI_INITIAL_CANDIDATES_FUNC_NAME, "ReSTIR Initial Candidates");

		if (render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
			draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::RESTIR_DI_TEMPORAL_REUSE_FUNC_NAME, "ReSTIR Temporal Reuse");
		if (render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
			draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::RESTIR_DI_SPATIAL_REUSE_FUNC_NAME, "ReSTIR Spatial Reuse");
	}
	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::PATH_TRACING_KERNEL, "Path Tracing Pass");
	ImGui::Separator();
	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::FULL_FRAME_TIME_KEY, "Total Sample Time");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	ImGui::TreePop();
}

void ImGuiRenderer::draw_perf_metric_specific_panel(std::shared_ptr<PerformanceMetricsComputer> perf_metrics, const std::string& perf_metrics_key, const std::string& label)
{
	float variance, min, max;
	variance = perf_metrics->get_variance(perf_metrics_key);
	min = perf_metrics->get_min(perf_metrics_key);
	max = perf_metrics->get_max(perf_metrics_key);

	static std::unordered_map<std::string, bool> key_to_display_graph;
	if (key_to_display_graph.find(perf_metrics_key) == key_to_display_graph.end())
		key_to_display_graph[perf_metrics_key] = false;

	// Pusing the ID for that perf key metrics so that no ImGui widgets collide
	ImGui::PushID(perf_metrics_key.c_str());

	ImGui::Text("%s: %.3fms (%.1f FPS)", label.c_str(), perf_metrics->get_current_value(perf_metrics_key), 1000.0f / perf_metrics->get_current_value(perf_metrics_key)); 
	if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
	{
		std::string line_1 = format_perf_metrics_tooltip_line(label, " (avg):", " (min / max):", " %.3fms (%.1f FPS)", perf_metrics->get_average(perf_metrics_key), 1000.0f / perf_metrics->get_average(perf_metrics_key));
		std::string line_2 = format_perf_metrics_tooltip_line(label, " (var):", " (min / max):", " %.3fms", variance);
		std::string line_3 = format_perf_metrics_tooltip_line(label, " (std dev):", " (min / max):", " %.3fms", std::sqrt(variance));
		std::string line_4 = format_perf_metrics_tooltip_line(label, " (min / max):", " (min / max):", " %.3fms / %.3fms", min, max);

		std::string tooltip = line_1 + "\n" + line_2 + "\n" + line_3 + "\n" + line_4;
		ImGuiRenderer::add_tooltip(tooltip);
	}
	
	ImGui::SameLine();
	ImGui::Checkbox("Show graph", &key_to_display_graph[perf_metrics_key]);
	if (key_to_display_graph[perf_metrics_key])
	{
		static std::unordered_map<std::string, std::pair<float, float>> key_to_min_max;
		if (key_to_min_max.find(perf_metrics_key) == key_to_min_max.end())
			key_to_min_max[perf_metrics_key] = std::make_pair(min, max);

		float& scale_min = key_to_min_max[perf_metrics_key].first;
		float& scale_max = key_to_min_max[perf_metrics_key].second;
		scale_min = perf_metrics->get_data_index(perf_metrics_key) == 0 ? min : scale_min;
		scale_max = perf_metrics->get_data_index(perf_metrics_key) == 0 ? max : scale_max;

		ImGui::PlotHistogram("",
			PerformanceMetricsComputer::data_getter,
			perf_metrics->get_data(perf_metrics_key).data(),
			perf_metrics->get_value_count(perf_metrics_key),
			/* value offset */0,
			label.c_str(),
			scale_min, scale_max,
			/* size */ ImVec2(0, 80));

		static std::unordered_map<std::string, bool> key_to_auto_rescale;
		if (key_to_auto_rescale.find(perf_metrics_key) == key_to_auto_rescale.end())
			key_to_auto_rescale[perf_metrics_key] = true;

		bool& auto_rescale = key_to_auto_rescale[perf_metrics_key];
		ImGui::SameLine();
		if (ImGui::Button("Rescale") || auto_rescale)
		{
			scale_min = min;
			scale_max = max;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Auto-rescale", &auto_rescale);
	}

	// Popping the ID for that perf key metrics
	ImGui::PopID();
}

template <class... Args>
std::string ImGuiRenderer::format_perf_metrics_tooltip_line(const std::string& label, const std::string& suffix, const std::string& longest_header_for_padding, const std::string& formatter_after_header, const Args& ...args)
{
	// Creating the formatter for automatically left-padding the header of the lines to the longer line (which is "(min / max)")
	std::string header_padding_formatter = "%-" + std::to_string(label.length() + longest_header_for_padding.length()) + "s";
	std::string line_formatter = header_padding_formatter + formatter_after_header;
	std::string header = label + suffix;

	char line_char[512];
	std::string test = "%s";
	snprintf(line_char, 512, line_formatter.c_str(), header.c_str(), args...);

	return std::string(line_char);
}

void ImGuiRenderer::draw_debug_panel()
{
	if (!ImGui::CollapsingHeader("Debug"))
		return;

	if (ImGui::Checkbox("Show NaNs", &m_renderer->get_render_settings().display_NaNs))
		m_render_window->set_render_dirty(true);
	ImGuiRenderer::show_help_marker("If true, NaNs that occur during the rendering will show up as pink pixels.");

	ImGui::TreePush("Debug tree");
	ImGui::TreePop();
}

void ImGuiRenderer::rescale_ui()
{
	ImGuiViewport* viewport = ImGui::GetMainViewport();
	float windowDpiScale = viewport->DpiScale;

	if (windowDpiScale > 1.0f)
	{
		ImGuiIO& io = ImGui::GetIO();

		// Scaling by the DPI -10% as judged more pleasing
		io.FontGlobalScale = windowDpiScale * 1.08f;
	}
}
