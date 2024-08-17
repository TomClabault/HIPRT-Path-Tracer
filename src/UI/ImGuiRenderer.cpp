/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompilerOptions.h"
#include "UI/ImGuiRenderer.h"
#include "UI/RenderWindow.h"

#include <chrono>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

ImGuiRenderer::ImGuiRenderer()
{
	ImGuiViewport* viewport = ImGui::GetMainViewport();
	float windowDpiScale = viewport->DpiScale;
	if (windowDpiScale > 1.0f)
		ImGui::GetStyle().ScaleAllSizes(windowDpiScale);
}

void ImGuiRenderer::WrappingTooltip(const std::string& text)
{
	ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 32.0f, 0.0f));
	ImGui::BeginTooltip();
	ImGui::PushTextWrapPos(0.0f);
	ImGui::Text("%s", text.c_str());
	ImGui::PopTextWrapPos();
	ImGui::EndTooltip();
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
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", render_settings.sample_number, m_render_window->get_samples_per_second(), m_renderer->m_render_width, m_renderer->m_render_height);

	if (render_settings.has_access_to_adaptive_sampling_buffers())
	{
		unsigned int converged_count = m_renderer->get_status_buffer_values().pixel_converged_count;
		unsigned int total_pixel_count = m_renderer->m_render_width * m_renderer->m_render_height;

		bool can_print_convergence = false;
		can_print_convergence |= render_settings.sample_number > render_settings.adaptive_sampling_min_samples;
		can_print_convergence |= render_settings.stop_pixel_noise_threshold > 0.0f;

		if (can_print_convergence)
		{
			ImGui::Text("Pixels converged: %d / %d - %.4f%%", converged_count, total_pixel_count, static_cast<float>(converged_count) / total_pixel_count * 100.0f);
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			{
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

				ImGuiRenderer::WrappingTooltip(text);
			}
		}
		else
		{
			ImGui::Text("Pixels converged: N/A");
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
				ImGuiRenderer::WrappingTooltip("Adaptive sampling hasn't kicked in yet... Convergence computation hasn't started.");
		}
	}
	else
	{
		ImGui::Text("Pixels converged: N/A");
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGuiRenderer::WrappingTooltip("Convergence is only computed when either adaptive sampling or the \"Pixel noise threshold\" render stopping condition is used.");
	}

	ImGui::Separator();

	if (ImGui::Button("Save viewport to PNG"))
		m_render_window->get_screenshoter()->write_to_png();

	ImGui::Separator();

	ImGui::PushItemWidth(16 * ImGui::GetFontSize());

	draw_render_settings_panel();
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

	std::vector<const char*> items = { "- Default", "- Denoiser blend", "- Denoiser - Normals", "- Denoiser - Denoised normals", "- Denoiser - Albedo", "- Denoiser - Denoised albedo" };
	if (render_settings.has_access_to_adaptive_sampling_buffers())
		items.push_back("- Adaptive sampling heatmap");

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
			m_application_settings->target_width = m_renderer->m_render_width;
			m_application_settings->target_height = m_renderer->m_render_height;
		}
	}

	ImGui::Separator();

	ImGui::BeginDisabled(m_application_settings->auto_sample_per_frame);
	ImGui::InputInt("Samples per frame", &render_settings.samples_per_frame);
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::Checkbox("Auto", &m_application_settings->auto_sample_per_frame);
	if (m_application_settings->auto_sample_per_frame)
	{
		ImGui::TreePush("Target GPU framerate tree");
		if (ImGui::InputFloat("Target GPU framerate", &m_application_settings->target_GPU_framerate))
			// Clamping to 1 FPS because going below that is dangerous in terms of driver timeouts
			m_application_settings->target_GPU_framerate = std::max(1.0f, m_application_settings->target_GPU_framerate);
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGuiRenderer::WrappingTooltip("The samples per frame will be automatically adjusted such that the GPU"
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

		if (ImGui::InputInt("Max Sample Count", &m_application_settings->max_sample_count))
			m_application_settings->max_sample_count = std::max(m_application_settings->max_sample_count, 0);

		if (ImGui::InputFloat("Max Render Time (s)", &m_application_settings->max_render_time))
			m_application_settings->max_render_time = std::max(m_application_settings->max_render_time, 0.0f);
		ImGui::Separator();

		static bool use_adaptive_sampling_threshold = false;
		ImGui::BeginDisabled(use_adaptive_sampling_threshold);
		if (ImGui::InputFloat("Pixel noise threshold", &render_settings.stop_pixel_noise_threshold))
			render_settings.stop_pixel_noise_threshold = std::max(0.0f, render_settings.stop_pixel_noise_threshold);
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGuiRenderer::WrappingTooltip("Cannot be set lower than the adaptive sampling threshold. 0.0 to disable.");

		// Having the stop pixel noise threshold lower than the adaptive sampling noise threshold
		// is impossible because the adaptive sampling will stop sampling the pixel before it can
		// converge enough to the stop pixels noise threshold (which is lower than the
		// adaptive sampling) so we're making sure the values are correct here
		if (render_settings.enable_adaptive_sampling && render_settings.stop_pixel_noise_threshold > 0.0f)
			render_settings.stop_pixel_noise_threshold = std::max(render_settings.stop_pixel_noise_threshold, render_settings.adaptive_sampling_noise_threshold);

		ImGui::EndDisabled(); // use_adaptive_sampling_threshold

		ImGui::SameLine();
		ImGui::Checkbox("Use adaptive sampling threshold", &use_adaptive_sampling_threshold);
		if (use_adaptive_sampling_threshold)
			// If we're using the adaptive sampling threshold, updating the stop pixel noise threshold with the adaptive sampling threshold
			render_settings.stop_pixel_noise_threshold = render_settings.adaptive_sampling_noise_threshold;

		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGuiRenderer::WrappingTooltip("If checked, the adaptive sampling noise threshold will be used.");

		bool update_converge_text = render_settings.stop_pixel_noise_threshold > 0.0f;
		ImGui::BeginDisabled(!update_converge_text);
		ImGui::TreePush("Tree pixel stop noise threshold");
		{
			if (ImGui::InputFloat("Pixel proportion", &render_settings.stop_pixel_percentage_converged))
				render_settings.stop_pixel_percentage_converged = std::max(0.0f, std::min(render_settings.stop_pixel_percentage_converged, 100.0f));
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			{
				std::string additional_info = "";
				if (render_settings.stop_pixel_noise_threshold == 0.0f)
					additional_info = "The stop pixel noise threshold must be > 0.0.";

				ImGuiRenderer::WrappingTooltip("The proportion of pixels that need to have converge to the noise threshold for the rendering to stop. In percentage [0, 100]." + additional_info);
			}
		}
		ImGui::EndDisabled();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		// Tree stop noise threshold
		ImGui::TreePop();

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
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
				ImGuiRenderer::WrappingTooltip("No envmap is loaded.");
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

			if (ImGui::Checkbox("Enable adaptive sampling", (bool*)&render_settings.enable_adaptive_sampling))
			{
				/*if (render_settings.enable_adaptive_sampling)
				m_render_window->get_display_view_system()->queue_display_view_change()*/
				m_render_window->set_render_dirty(true);
			}

			float adaptive_sampling_noise_threshold_before = render_settings.adaptive_sampling_noise_threshold;
			ImGui::BeginDisabled(!render_settings.enable_adaptive_sampling);
			if (ImGui::InputInt("Adaptive sampling minimum samples", &render_settings.adaptive_sampling_min_samples))
				m_render_window->set_render_dirty(true);
			if (ImGui::InputFloat("Adaptive sampling noise threshold", &render_settings.adaptive_sampling_noise_threshold))
			{
				render_settings.adaptive_sampling_noise_threshold = std::max(0.0f, render_settings.adaptive_sampling_noise_threshold);
			
				m_render_window->set_render_dirty(true);
			}
			ImGui::EndDisabled();

			ImGui::TreePop();
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
				if (ImGui::Checkbox("Use visibility in RIS target function", &render_settings.ris_settings.use_visibility_in_target_function))
				{
					kernel_options->set_macro(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, render_settings.ris_settings.use_visibility_in_target_function ? 1 : 0);
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
				ImGui::SeparatorText("Initial Candidates Pass");
				//if (ImGui::CollapsingHeader("Initial Candidates Sampling"))
				{
					ImGui::TreePush("ReSTIR DI - Initial Candidate Pass Tree");

					// Whether or not to use the visibility term in the target function used for
					// resampling the initial candidates
					static bool use_visibility_initial_candidates = ReSTIR_DI_InitialCandidatesUseVisiblityTargetFunction;
					if (ImGui::Checkbox("Use visibility in target function", &use_visibility_initial_candidates))
					{
						kernel_options->set_macro(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_CANDIDATES_VISIBILITY_TARGET_FUNCTION, use_visibility_initial_candidates ? 1 : 0);
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}

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

				//if (ImGui::CollapsingHeader("Spatial Reuse Pass"))
				ImGui::SeparatorText("Spatial Reuse Pass");
				{
					ImGui::TreePush("ReSTIR DI - Spatial Reuse Pass Tree");

					// Whether or not to use the visibility term in the target function used for
					// resampling the neighbors in the spatial reuse pass of ReSTIR DI
					static bool use_visibility_target_function = ReSTIR_DI_SpatialReuseUseVisiblityTargetFunction;
					if (ImGui::Checkbox("Use visibility in target function", &use_visibility_target_function))
					{
						kernel_options->set_macro(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_REUSE_VISIBILITY_TARGET_FUNCTION, use_visibility_target_function ? 1 : 0);
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}

					if (ImGui::SliderInt("Spatial Reuse Pass Count", &render_settings.restir_di_settings.spatial_pass.number_of_passes, 0, 64))
					{
						// Clamping
						render_settings.restir_di_settings.spatial_pass.number_of_passes = std::max(0, render_settings.restir_di_settings.spatial_pass.number_of_passes);

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
						render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count = std::max(0, render_settings.restir_di_settings.spatial_pass.spatial_reuse_neighbor_count);

						m_render_window->set_render_dirty(true);
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::SeparatorText("Bias correction");

					static bool bias_correction_use_visibility = ReSTIR_DI_SpatialReuseBiasUseVisiblity;
					if (ImGui::Checkbox("Use visibility in bias correction", &bias_correction_use_visibility))
					{
						kernel_options->set_macro(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_REUSE_BIAS_CORRECTION_USE_VISIBILITY, bias_correction_use_visibility? 1 : 0);
						m_renderer->recompile_kernels();

						m_render_window->set_render_dirty(true);
					}

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

			const char* items[] = { "- No envmap sampling", "- Envmap Sampling - Binary Search" };
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

void ImGuiRenderer::draw_objects_panel()
{
	if (!ImGui::CollapsingHeader("Objects"))
		return;
	ImGui::TreePush("Objects tree");

	std::vector<RendererMaterial> materials = m_renderer->get_materials();
	std::vector<std::string> material_names = m_renderer->get_material_names();

	bool material_changed = false;
	static int currently_selected_material = 0;

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
			kernel_options->set_macro("__USE_HWI__", 1);
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
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGuiRenderer::WrappingTooltip("Whether or not to enable hardware accelerated ray tracing (bbox & triangle intersections)");
		break;

	case AMD_UNSUPPORTED:
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGuiRenderer::WrappingTooltip("Hardware accelerated ray tracing is only supported on RDNA2+ GPUs.");
		break;

	case NVIDIA_UNSUPPORTED:
		if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGuiRenderer::WrappingTooltip("HIPRT cannot access NVIDIA's proprietary hardware accelerated ray-tracing. Feature unavailable.");
		break;
	}

	if (ImGui::InputFloat("GPU Stall Percentage", &m_application_settings->GPU_stall_percentage))
		m_application_settings->GPU_stall_percentage = std::max(0.0f, std::min(m_application_settings->GPU_stall_percentage, 99.9f));
	if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
		ImGuiRenderer::WrappingTooltip("How much percent of the time the GPU will be forced to be idle (not rendering anything). This feature is meant only for GPUs that get too hot to avoid burning your GPUs during prolonged renders.");

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
	rolling_window_size_changed |= ImGui::RadioButton("25", &rolling_window_size, 25); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("100", &rolling_window_size, 100); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("250", &rolling_window_size, 250); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("1000", &rolling_window_size, 1000);

	if (rolling_window_size_changed)
		m_render_window_perf_metrics->resize_window(rolling_window_size);

	float variance, min, max;
	variance = m_render_window_perf_metrics->get_variance(PerformanceMetricsComputer::SAMPLE_TIME_KEY);
	min = m_render_window_perf_metrics->get_min(PerformanceMetricsComputer::SAMPLE_TIME_KEY);
	max = m_render_window_perf_metrics->get_max(PerformanceMetricsComputer::SAMPLE_TIME_KEY);

	static float scale_min = min, scale_max = max;
	scale_min = m_render_window_perf_metrics->get_data_index(PerformanceMetricsComputer::SAMPLE_TIME_KEY) == 0 ? min : scale_min;
	scale_max = m_render_window_perf_metrics->get_data_index(PerformanceMetricsComputer::SAMPLE_TIME_KEY) == 0 ? max : scale_max;

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::PlotHistogram("",
		PerformanceMetricsComputer::data_getter,
		m_render_window_perf_metrics->get_data(PerformanceMetricsComputer::SAMPLE_TIME_KEY).data(),
		m_render_window_perf_metrics->get_value_count(PerformanceMetricsComputer::SAMPLE_TIME_KEY),
		/* value offset */0,
		"Sample time",
		scale_min, scale_max,
		/* size */ ImVec2(0, 80));
	static bool auto_rescale = true;
	ImGui::SameLine();
	if (ImGui::Button("Rescale") || auto_rescale)
	{
		scale_min = min;
		scale_max = max;
	}
	ImGui::SameLine();
	ImGui::Checkbox("Auto-rescale", &auto_rescale);

	ImGui::Text("Sample time (avg)      : %.3fms (%.1f FPS)", m_render_window_perf_metrics->get_average(PerformanceMetricsComputer::SAMPLE_TIME_KEY), 1000.0f / m_render_window_perf_metrics->get_average(PerformanceMetricsComputer::SAMPLE_TIME_KEY));
	ImGui::Text("Sample time (var)      : %.3fms", variance);
	ImGui::Text("Sample time (std dev)  : %.3fms", std::sqrt(variance));
	ImGui::Text("Sample time (min / max): %.3fms / %.3fms", min, max);

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	ImGui::TreePop();
}

void ImGuiRenderer::draw_debug_panel()
{
	if (!ImGui::CollapsingHeader("Debug"))
		return;

	if (ImGui::Checkbox("Show NaNs", &m_renderer->get_render_settings().display_NaNs))
		m_render_window->set_render_dirty(true);

	if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
		ImGuiRenderer::WrappingTooltip("If true, NaNs that occur during the rendering will show up as pink pixels.");

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
