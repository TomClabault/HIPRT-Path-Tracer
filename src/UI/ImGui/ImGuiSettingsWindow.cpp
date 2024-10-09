/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "HostDeviceCommon/RenderSettings.h"
#include "Renderer/GPURenderer.h"
#include "Threads/ThreadManager.h"
#include "UI/ImGui/ImGuiRenderer.h"
#include "UI/ImGui/ImGuiSettingsWindow.h"
#include "UI/RenderWindow.h"

#include <iostream>

const char* ImGuiSettingsWindow::TITLE = "Settings";
const float ImGuiSettingsWindow::BASE_SIZE = 630.0f;

void ImGuiSettingsWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;

	m_application_settings = render_window->get_application_settings();
	m_renderer = render_window->get_renderer();
	m_render_window_denoiser = render_window->get_denoiser();
	m_render_window_perf_metrics = m_render_window->get_performance_metrics();
}

void ImGuiSettingsWindow::draw()
{
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 0.0f));
	ImGui::Begin(ImGuiSettingsWindow::TITLE, nullptr, ImGuiWindowFlags_NoDecoration);

	draw_header();
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

	m_current_size = ImGui::GetWindowSize();

	ImGui::PopStyleVar(3);
	ImGui::End();
}

void ImGuiSettingsWindow::draw_header()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (render_settings.accumulate)
		ImGui::Text("Render time: %.3fs", m_render_window->get_current_render_time() / 1000.0f);
	else
		ImGui::Text("Frame time: %.3fms", m_render_window_perf_metrics->get_current_value(GPURenderer::FULL_FRAME_TIME_KEY));
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
			if (render_settings.accumulate)
				// No need to show the text if we're not accumulating
			{
				ImGui::Text("Pixels converged: N/A");
				ImGuiRenderer::show_help_marker("Adaptive sampling hasn't kicked in yet... Convergence computation hasn't started.");
			}
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
}

void ImGuiSettingsWindow::draw_render_settings_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (!ImGui::CollapsingHeader("Render Settings"))
		return;
	ImGui::TreePush("Render settings tree");

	ImGui::SeparatorText("Global Performance Presets");
	static int preset_selected = 0;
	std::vector<const char*> preset_items = { "None", "Fastest", "Fast", "Medium", "High Quality" };
	if (ImGui::Combo("Performance Preset", &preset_selected, preset_items.data(), preset_items.size()))
		apply_performance_preset(static_cast<ImGuiRendererPerformancePreset>(preset_selected));

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::SeparatorText("Viewport Settings");
	std::vector<const char*> items = { "- Default", "- Denoiser blend", "- Denoiser - Normals", "- Denoiser - Denoised normals", "- Denoiser - Albedo", "- Denoiser - Denoised albedo" };
	if (render_settings.has_access_to_adaptive_sampling_buffers())
	{
		items.push_back("- Pixel convergence heatmap");
		items.push_back("- Converged pixels map");
	}

	int display_view_selected = m_render_window->get_display_view_system()->get_current_display_view_type();
	if (ImGui::Combo("Display View", &display_view_selected, items.data(), items.size()))
		m_render_window->get_display_view_system()->queue_display_view_change(static_cast<DisplayViewType>(display_view_selected));

	static float resolution_scaling_current_widget_value = m_application_settings->render_resolution_scale;
	ImGui::BeginDisabled(m_application_settings->keep_same_resolution);
	ImGui::InputFloat("Resolution scale", &resolution_scaling_current_widget_value);

	if (resolution_scaling_current_widget_value != m_application_settings->render_resolution_scale)
	{
		ImGui::TreePush("Resolution scaling apply button tree");

		if (ImGui::Button("Apply"))
		{
			if (resolution_scaling_current_widget_value <= 0.0f)
				// Wrong resolution scaling factor, restoring to previous value
				resolution_scaling_current_widget_value = m_application_settings->render_resolution_scale;
			else
			{
				// Valid scaling factor
				m_application_settings->render_resolution_scale = resolution_scaling_current_widget_value;
				m_render_window->change_resolution_scaling(resolution_scaling_current_widget_value);
				m_render_window->set_render_dirty(true);
			}
		}

		ImGui::TreePop();
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

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::SeparatorText("General Settings");

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
			if (!render_settings.accumulate)
			{
				// Adding a shortcut button to re-enable accumulation
				if (ImGui::Button("Enable accumulation"))
				{
					render_settings.accumulate = true;
					m_render_window->set_render_dirty(true);
				}
			}
			ImGui::BeginDisabled(!render_settings.accumulate); // Cannot use stopping condition if not accumulating
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
			ImGui::EndDisabled(); // !render_settings.accumulate

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

		if (ImGui::SliderFloat("Minimum Light Contribution", &render_settings.minimum_light_contribution, 0.0f, 10.0f))
		{
			render_settings.minimum_light_contribution = std::max(0.0f, render_settings.minimum_light_contribution);
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("If a selected light (for direct lighting estimation) contributes at a given "
			" point less than this 'minimum_light_contribution' value then the light sample is discarded. "
			"This can improve performance at the cost of some bias depending on the scene.\n"
			"0.0f to disable");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		// Light clamping tree
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Nested dielectrics"))
	{
		ImGui::TreePush("Nested dielectrics tree");

		std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
		const char* items[] = { "- Automatic", "- With priorities" };
		if (ImGui::Combo("Nested dielectrics strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY), items, IM_ARRAYSIZE(items)))
		{
			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		static int nested_dielectrics_stack_size = NestedDielectricsStackSize;
		if (ImGui::SliderInt("Stack Size", &nested_dielectrics_stack_size, 3, 8))
			nested_dielectrics_stack_size = std::max(1, nested_dielectrics_stack_size);

		if (nested_dielectrics_stack_size != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION))
		{
			ImGui::TreePush("Apply button nested dielectric stack size");
			if (ImGui::Button("Apply"))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::NESTED_DIELETRCICS_STACK_SIZE_OPTION, nested_dielectrics_stack_size);

				m_renderer->recompile_kernels();
				m_renderer->resize_g_buffer_ray_volume_states();
				m_render_window->set_render_dirty(true);
			}
			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Alpha Testing"))
	{
		ImGui::TreePush("Alpha testing tree");

		if (ImGui::Checkbox("Do alpha testing", &render_settings.do_alpha_testing))
			m_render_window->set_render_dirty(true);

		ImGui::TreePop();
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiSettingsWindow::apply_performance_preset(ImGuiRendererPerformancePreset performance_preset)
{
	switch (performance_preset)
	{
	case PERF_PRESET_NONE:
		break;

	case PERF_PRESET_FASTEST:
	{
		m_application_settings->render_resolution_scale = 0.5f;
		m_application_settings->target_GPU_framerate = 25.0f;

		HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
		render_settings.nb_bounces = 1;
		render_settings.ris_settings.number_of_bsdf_candidates = 0;
		render_settings.ris_settings.number_of_light_candidates = 1;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_RIS_BSDF_AND_LIGHT);
		m_renderer->recompile_kernels();

		m_render_window->change_resolution_scaling(0.5f);
		m_render_window->set_render_dirty(true);

		break;
	}

	case PERF_PRESET_FAST:
	{
		m_application_settings->render_resolution_scale = 0.75f;
		m_application_settings->target_GPU_framerate = 15.0f;

		HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
		render_settings.nb_bounces = 2;
		render_settings.ris_settings.number_of_bsdf_candidates = 1;
		render_settings.ris_settings.number_of_light_candidates = 4;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_RIS_BSDF_AND_LIGHT);
		m_renderer->recompile_kernels();

		m_render_window->change_resolution_scaling(0.75f);
		m_render_window->set_render_dirty(true);

		break;
	}

	case PERF_PRESET_MEDIUM:
	{
		m_application_settings->render_resolution_scale = 1.0f;
		m_application_settings->target_GPU_framerate = 5.0f;

		HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
		render_settings.nb_bounces = 2;
		render_settings.ris_settings.number_of_bsdf_candidates = 1;
		render_settings.ris_settings.number_of_light_candidates = 8;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_RIS_BSDF_AND_LIGHT);
		m_renderer->recompile_kernels();

		m_render_window->change_resolution_scaling(1.0f);
		m_render_window->set_render_dirty(true);

		break;
	}

	case PERF_PRESET_HIGH_QUALITY:
	{
		m_application_settings->render_resolution_scale = 1.0f;
		m_application_settings->target_GPU_framerate = 5.0f;

		HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
		render_settings.nb_bounces = 4;
		render_settings.ris_settings.number_of_bsdf_candidates = 1;
		render_settings.ris_settings.number_of_light_candidates = 8;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY, LSS_RESTIR_DI);
		m_renderer->recompile_kernels();

		m_render_window->change_resolution_scaling(1.0f);
		m_render_window->set_render_dirty(true);

		break;
	}

	default:
		break;
	}
}

void ImGuiSettingsWindow::draw_camera_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	Camera& camera = m_renderer->get_camera();

	if (ImGui::CollapsingHeader("Camera"))
	{
		ImGui::TreePush("Camera tree");

		if (ImGui::Checkbox("Do ray jittering", &camera.do_jittering))
			m_render_window->set_render_dirty(true);

		static float camera_fov = camera.vertical_fov / M_PI * 180.0f;
		if (ImGui::SliderFloat("FOV", &camera_fov, 0.0f, 180.0f, "%.3fdeg", ImGuiSliderFlags_AlwaysClamp))
		{
			camera.set_FOV(camera_fov / 180.0f * M_PI);

			m_render_window->set_render_dirty(true);
		}

		if (ImGui::SliderFloat("Camera Speed", &camera.user_movement_speed_multiplier, 0.0f, 10.0f))
			camera.user_movement_speed_multiplier = std::max(0.0f, camera.user_movement_speed_multiplier);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::BeginDisabled(!render_settings.accumulate);
		ImGui::Checkbox("Render low resolution when interacting", &render_settings.allow_render_low_resolution);
		if (!render_settings.accumulate)
			ImGuiRenderer::add_tooltip("Cannot render at low resolution when not accumulating. If you want to render at "
				"a lower resolution, you can use the resolution scale in \"Render Settings\"for that.");
		ImGui::SliderInt("Render low resolution downscale", &render_settings.render_low_resolution_scaling, 1, 8);
		if (!render_settings.accumulate)
			ImGuiRenderer::add_tooltip("Cannot render at low resolution when not accumulating. If you want to render at "
				"a lower resolution, you can use the resolution scale in \"Render Settings\"for that.");
		ImGui::EndDisabled();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_environment_panel()
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
			float& rota_X = m_renderer->get_envmap().rotation_X;
			float& rota_Y = m_renderer->get_envmap().rotation_Y;
			float& rota_Z = m_renderer->get_envmap().rotation_Z;

			bool& animate_envmap = m_renderer->get_envmap().animate;
			float& animation_speed_X = m_renderer->get_envmap().animation_speed_X;
			float& animation_speed_Y = m_renderer->get_envmap().animation_speed_Y;
			float& animation_speed_Z = m_renderer->get_envmap().animation_speed_Z;

			ImGui::BeginDisabled(m_renderer->get_render_settings().accumulate);
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::Checkbox("Animate", &animate_envmap);

			if (animate_envmap)
			{
				ImGui::Text("Speeds are in degrees per second");
				ImGui::SliderFloat("Animation Speed X", &animation_speed_X, 0.0f, 360.0f);
				ImGui::SliderFloat("Animation Speed Y", &animation_speed_Y, 0.0f, 360.0f);
				ImGui::SliderFloat("Animation Speed Z", &animation_speed_Z, 0.0f, 360.0f);

				//float delta_time = m_render_window->get_UI_delta_time();
				//rota_X += animation_speed_X / 360.0f / (1000.0f / delta_time);
				//rota_Y += animation_speed_Y / 360.0f / (1000.0f / delta_time);
				//rota_Z += animation_speed_Z / 360.0f / (1000.0f / delta_time);

				//// Keeping only the fractional part. This effectively brings a value
				//// that went above 1 back to between 0 and 1 to keep the rotation between
				//// 0 and 360 degrees
				//rota_X = rota_X - static_cast<int>(rota_X);
				//rota_Y = rota_Y - static_cast<int>(rota_Y);
				//rota_Z = rota_Z - static_cast<int>(rota_Z);
			}

			ImGui::EndDisabled();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			bool rotation_changed = false;
			rotation_changed |= ImGui::SliderFloat("Envmap rotation X", &rota_X, 0.0f, 1.0f);
			rotation_changed |= ImGui::SliderFloat("Envmap rotation Y", &rota_Y, 0.0f, 1.0f);
			rotation_changed |= ImGui::SliderFloat("Envmap rotation Z", &rota_Z, 0.0f, 1.0f);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
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

void ImGuiSettingsWindow::draw_sampling_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Sampling"))
	{
		ImGui::TreePush("Sampling tree");

		if (ImGui::CollapsingHeader("Adaptive sampling"))
		{

			ImGui::TreePush("Adaptive sampling tree");

			if (!render_settings.accumulate)
			{
				if (ImGui::Button("Enable accumulation"))
				{
					render_settings.accumulate = true;
					m_render_window->set_render_dirty(true);
				}
			}

			// Cannot use adaptive sampling without accumulation
			ImGui::BeginDisabled(!render_settings.accumulate);

			if (ImGui::Checkbox("Enable adaptive sampling", (bool*)&render_settings.enable_adaptive_sampling))
				m_render_window->set_render_dirty(true);
			if (!render_settings.accumulate)
				ImGuiRenderer::add_tooltip("Cannot use adaptive sampling when accumulation is not on.");

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

			const char* items[] = { "- No direct light sampling", "- Uniform one light", "- BSDF Sampling", "- MIS (1 Light + 1 BSDF)", "- RIS BDSF + Light candidates", "- ReSTIR DI (Primary Hit Only)" };
			if (ImGui::Combo("Direct light sampling strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			// Display additional widgets to control the parameters of the direct light
			// sampling strategy chosen (the number of candidates for RIS for example)
			switch (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY))
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
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, use_visibility_ris_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("RIS # of BSDF candidates", &render_settings.ris_settings.number_of_bsdf_candidates, 0, 16))
				{
					// Clamping to 0
					render_settings.ris_settings.number_of_bsdf_candidates = std::max(0, render_settings.ris_settings.number_of_bsdf_candidates);

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("RIS # of light candidates", &render_settings.ris_settings.number_of_light_candidates, 0, 32))
				{
					// Clamping to 0
					render_settings.ris_settings.number_of_light_candidates = std::max(0, render_settings.ris_settings.number_of_light_candidates);

					m_render_window->set_render_dirty(true);
				}

				break;
			}

			case LSS_RESTIR_DI:
			{
				display_ReSTIR_DI_bias_status(global_kernel_options);

				if (ImGui::CollapsingHeader("General Settings"))
				{
					ImGui::TreePush("ReSTIR DI - General Settings Tree");

					{
						if (ImGui::SliderInt("M-cap", &render_settings.restir_di_settings.m_cap, 0, 48))
						{
							render_settings.restir_di_settings.m_cap = std::max(0, render_settings.restir_di_settings.m_cap);
							if (render_settings.accumulate)
								m_render_window->set_render_dirty(true);
						}

						ImGui::Dummy(ImVec2(0.0f, 20.0f));

						if (ImGui::Checkbox("Use Final Visibility", &render_settings.restir_di_settings.do_final_shading_visibility))
						{
							m_renderer->recompile_kernels();
							m_render_window->set_render_dirty(true);
						}

						ImGui::Dummy(ImVec2(0.0f, 20.0f));

						static bool use_heuristics_at_all = true;
						static bool use_normal_heuristic_backup = render_settings.restir_di_settings.use_normal_similarity_heuristic;
						static bool use_plane_distance_heuristic_backup = render_settings.restir_di_settings.use_plane_distance_heuristic;
						static bool use_roughness_heuristic_backup = render_settings.restir_di_settings.use_roughness_similarity_heuristic;
						if (ImGui::Checkbox("Use Heuristics for neighbor rejection", &use_heuristics_at_all))
						{
							if (!use_heuristics_at_all)
							{
								// Saving the usage of the heuristics for later restoration
								use_normal_heuristic_backup = render_settings.restir_di_settings.use_normal_similarity_heuristic;
								use_plane_distance_heuristic_backup = render_settings.restir_di_settings.use_plane_distance_heuristic;
								use_roughness_heuristic_backup = render_settings.restir_di_settings.use_roughness_similarity_heuristic;

								render_settings.restir_di_settings.use_normal_similarity_heuristic = false;
								render_settings.restir_di_settings.use_plane_distance_heuristic = false;
								render_settings.restir_di_settings.use_roughness_similarity_heuristic = false;
							}
							else
							{
								// Restoring heuristics usage to their backup values
								render_settings.restir_di_settings.use_normal_similarity_heuristic = use_normal_heuristic_backup;
								render_settings.restir_di_settings.use_plane_distance_heuristic = use_plane_distance_heuristic_backup;
								render_settings.restir_di_settings.use_roughness_similarity_heuristic = use_roughness_heuristic_backup;
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




							if (ImGui::Checkbox("Use Normal Similarity Heuristic", &render_settings.restir_di_settings.use_normal_similarity_heuristic))
								m_render_window->set_render_dirty(true);

							if (render_settings.restir_di_settings.use_normal_similarity_heuristic)
							{
								if (ImGui::SliderFloat("Normal Similarity Threshold", &render_settings.restir_di_settings.normal_similarity_angle_degrees, 0.0f, 360.0f, "%.3f deg", ImGuiSliderFlags_AlwaysClamp))
								{
									render_settings.restir_di_settings.normal_similarity_angle_precomp = std::cos(render_settings.restir_di_settings.normal_similarity_angle_degrees * M_PI / 180.0f);

									m_render_window->set_render_dirty(true);
								}
							}




							if (ImGui::Checkbox("Use Plane Distance Heuristic", &render_settings.restir_di_settings.use_plane_distance_heuristic))
								m_render_window->set_render_dirty(true);

							if (render_settings.restir_di_settings.use_plane_distance_heuristic)
								if (ImGui::SliderFloat("Plane Distance Threshold", &render_settings.restir_di_settings.plane_distance_threshold, 0.0f, 1.0f))
									m_render_window->set_render_dirty(true);




							if (ImGui::Checkbox("Use Roughness Heuristic", &render_settings.restir_di_settings.use_roughness_similarity_heuristic))
								m_render_window->set_render_dirty(true);

							if (render_settings.restir_di_settings.use_roughness_similarity_heuristic)
								if (ImGui::SliderFloat("Roughness Threshold", &render_settings.restir_di_settings.roughness_similarity_threshold, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
									m_render_window->set_render_dirty(true);



							// ReSTIR DI Heursitics Tree
							ImGui::TreePop();
						}
					}

					ImGui::TreePop();
					ImGui::Dummy(ImVec2(0.0f, 20.0f));
				}





				if (ImGui::CollapsingHeader("Initial Candidates Pass"))
				{
					ImGui::TreePush("ReSTIR DI - Initial Candidate Pass Tree");

					{
						static bool do_light_presampling = ReSTIR_DI_DoLightsPresampling;
						if (ImGui::Checkbox("Do Light Presampling", &do_light_presampling))
						{
							global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHTS_PRESAMPLING, do_light_presampling ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

							m_renderer->recompile_kernels();
							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("If checked, lights are presampled in a pre-process pass as proposed in"
							" [Rearchitecting Spatiotemporal Resampling for Production, Wyman, Panteleev, 2021]\n\n"
							"This improves performance in scenes with dozens of thousands / millions of"
							" lights by avoiding cache trashing because of the memory random walk that"
							" light sampling becomes with that many lights");

						static bool use_initial_target_function_visibility = ReSTIR_DI_InitialTargetFunctionVisibility;
						if (ImGui::Checkbox("Use visibility in target function", &use_initial_target_function_visibility))
						{
							global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY, use_initial_target_function_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
							m_renderer->recompile_kernels();

							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("Whether or not to use the visibility term in the target function used for "
							"resampling initial candidates");

						if (ImGui::SliderInt("# of BSDF initial candidates", &render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates, 0, 16))
						{
							// Clamping to 0
							render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates = std::max(0, render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates);

							m_render_window->set_render_dirty(true);
						}

						if (ImGui::SliderInt("# of initial light candidates", &render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates, 0, 32))
						{
							// Clamping to 0
							render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates = std::max(0, render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates);

							m_render_window->set_render_dirty(true);
						}

						ImGui::BeginDisabled(!m_renderer->has_envmap());
						if (ImGui::SliderFloat("Envmap candidate probability", &render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability, 0.0f, 1.0f))
						{
							render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability = hippt::clamp(0.0f, 1.0f, render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability);

							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("The probability to sample the envmap per each \"initial light candidates\"");
						ImGui::EndDisabled();
					}

					ImGui::TreePop();
					ImGui::Dummy(ImVec2(0.0f, 20.0f));
				}





				if (ImGui::CollapsingHeader("Visibility Reuse Pass"))
				{
					ImGui::TreePush("ReSTIR DI - Visibility Reuse Pass Tree");

					{
						static bool do_visibility_reuse = ReSTIR_DI_DoVisibilityReuse;
						if (ImGui::Checkbox("Do visibility reuse", &do_visibility_reuse))
						{
							global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE, do_visibility_reuse ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
							m_renderer->recompile_kernels();

							m_render_window->set_render_dirty(true);
						}
					}

					ImGui::TreePop();
					ImGui::Dummy(ImVec2(0.0f, 20.0f));
				}





				if (ImGui::CollapsingHeader("Temporal Reuse Pass"))
				{
					ImGui::TreePush("ReSTIR DI - Temporal Reuse Pass Tree");
					{
						if (render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass && render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
						{
							if (ImGui::Checkbox("Do Fused Spatiotemporal", &render_settings.restir_di_settings.do_fused_spatiotemporal))
							{
								render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested = true;

								m_render_window->set_render_dirty(true);
							}
							ImGuiRenderer::show_help_marker("If checked, the spatial and temporal pass will be fused into a single kernel call. "
								"This avoids a synchronization barrier between the temporal pass and the spatial pass "
								"and increases performance. Because the spatial must then resample without the output of the temporal pass, the spatial "
								"pass only resamples on the temporal reservoir buffer, not the temporal + initial candidates reservoir "
								"(which is the output of the temporal pass). This is usually imperceptible.");
						}

						if (ImGui::Checkbox("Do Temporal Reuse", &render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass))
						{
							m_render_window->set_render_dirty(true);

							if (!render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
								// Disabling fused spatiotemporal if we just disabled the temporal reuse
								render_settings.restir_di_settings.do_fused_spatiotemporal = false;
						}

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

							if (ImGui::Checkbox("Use Permutation Sampling", &render_settings.restir_di_settings.temporal_pass.use_permutation_sampling))
								m_render_window->set_render_dirty(true);
							ImGuiRenderer::show_help_marker("If true, the back-projected position of the current pixel (temporal neighbor position) will be shuffled"
								" to add temporal variations.");
						}

						ImGui::TreePop();
						ImGui::Dummy(ImVec2(0.0f, 20.0f));
					}
				}





				if (ImGui::CollapsingHeader("Spatial Reuse Pass"))
				{
					ImGui::TreePush("ReSTIR DI - Spatial Reuse Pass Tree");
					{
						if (render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass && render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
						{
							if (ImGui::Checkbox("Do Fused Spatiotemporal", &render_settings.restir_di_settings.do_fused_spatiotemporal))
							{
								render_settings.restir_di_settings.temporal_pass.temporal_buffer_clear_requested = true;

								m_render_window->set_render_dirty(true);
							}
							ImGuiRenderer::show_help_marker("If checked, the spatial and temporal pass will be fused into a single kernel call. "
								"This avois a synchronization barrier between the temporal pass and the spatial pass "
								"and increases performance. Because the spatial must then resample without the output of the temporal pass, the spatial "
								"pass only resamples on the temporal reservoir buffer, not the temporal + initial candidates reservoir "
								"(which is the output of the temporal pass). This is usually imperceptible.");
						}

						if (ImGui::Checkbox("Do Spatial Reuse", &render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass))
						{
							m_render_window->set_render_dirty(true);

							if (!render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
								// Disabling fused spatiotemporal if we just disabled the spatial reuse
								render_settings.restir_di_settings.do_fused_spatiotemporal = false;
						}

						if (render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
						{
							static bool use_spatial_target_function_visibility = ReSTIR_DI_SpatialTargetFunctionVisibility;
							if (ImGui::Checkbox("Use visibility in target function", &use_spatial_target_function_visibility))
							{
								global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY, use_spatial_target_function_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
								m_renderer->recompile_kernels();

								m_render_window->set_render_dirty(true);
							}
							ImGuiRenderer::show_help_marker("Whether or not to use the visibility term in the target function used for "
								"resampling spatial neighbors.");

							int max_neighbor_count = render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;
							if (render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost)
								max_neighbor_count = std::max(max_neighbor_count, render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count);
							static int partial_visibility_neighbor_count = max_neighbor_count;
							if (use_spatial_target_function_visibility)
							{
								ImGui::TreePush("VisibilitySpatialReuseLastPassOnly Tree");

								{
									if (ImGui::SliderInt("Partial Neighbor Visibility", &partial_visibility_neighbor_count, 0, max_neighbor_count, "%d", ImGuiSliderFlags_AlwaysClamp))
									{
										// Using -1 so that the user manipulates intuitive numbers between 0 and
										// 'render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count'
										// but the shader actually wants value between -1 and
										// 'render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count' for it to be meaningful
										render_settings.restir_di_settings.spatial_pass.neighbor_visibility_count = partial_visibility_neighbor_count;

										m_render_window->set_render_dirty(true);
									}
									ImGuiRenderer::show_help_marker("How many neighbors will actually use a visibility term, can be useful to balance "
										"performance/variance but lowering this value below the maximum amount of neighbors may actually reduce "
										"performance because the final shading pass will have more visibility tests to do: if all neighbors use "
										"visibility during spatial resampling, then the final shading pass can be certain that all neighbors "
										"already take occlusion into account and so the final shading pass doesn't compute visibility. "
										"However, if 1 or 2 neighbors do not include visibility for example, then the final shading pass will "
										"have to trace rays for these neighbors and this will slow down the final shading pass quite a bit.");

									if (ImGui::Checkbox("Only on the last pass", &render_settings.restir_di_settings.spatial_pass.do_visibility_only_last_pass))
										m_render_window->set_render_dirty(true);
									ImGuiRenderer::show_help_marker("If checked, the visibility in the resampling target function will only be used on the last spatial reuse pass");
								}
								ImGui::Dummy(ImVec2(0.0f, 20.0f));

								ImGui::TreePop();
							}


							if (ImGui::SliderInt("Spatial Reuse Pass Count", &render_settings.restir_di_settings.spatial_pass.number_of_passes, 1, 8))
							{
								// Clamping
								render_settings.restir_di_settings.spatial_pass.number_of_passes = std::max(1, render_settings.restir_di_settings.spatial_pass.number_of_passes);

								m_render_window->set_render_dirty(true);
							}

							if (ImGui::SliderInt("Spatial Reuse Radius (px)", &render_settings.restir_di_settings.spatial_pass.reuse_radius, 1, 64))
							{
								// Clamping
								render_settings.restir_di_settings.spatial_pass.reuse_radius = std::max(1, render_settings.restir_di_settings.spatial_pass.reuse_radius);

								m_render_window->set_render_dirty(true);
							}

							// Checking the value before the "Neighbor Reuse Count" slider is modified
							// so that we know whether or not we'll have to keep the
							// 'partial_visibility_neighbor_count' value updated for the "Partial Neighbor Visibility" slider
							bool will_need_to_update_partial_visibility = partial_visibility_neighbor_count == max_neighbor_count;
							if (ImGui::SliderInt("Neighbor Reuse Count", &render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count, 1, 16))
							{
								// Clamping
								render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count = std::max(1, render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count);

								// Updating the maximum
								max_neighbor_count = render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;
								if (render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost)
									max_neighbor_count = std::max(max_neighbor_count, render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count);

								bool reuse_count_is_the_max = max_neighbor_count == render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;
								reuse_count_is_the_max |= !render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost;
								if (will_need_to_update_partial_visibility && reuse_count_is_the_max)
								{
									// Also updating the partial visibility neighbor index slider if it was set to the maximum
									// amount of neighbors
									partial_visibility_neighbor_count = render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;
									render_settings.restir_di_settings.spatial_pass.neighbor_visibility_count = partial_visibility_neighbor_count;
								}

								if (render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count < render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count)
									// If disocclusion boost is now below the spatial neighbor count, bumping it up
									// because it makes no sense to have the disocclusion boost below the base
									// spatial neighbor count
									render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count = render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;

								m_render_window->set_render_dirty(true);
							}

							if (ImGui::Checkbox("Increase Disocclusion Reuse Count", &render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost))
							{
								m_render_window->set_render_dirty(true);
								if (render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost)
								{
									// We just enabled disocclusion boost

									// Recomputing the max neighbor with the disocclusion boost taken into account
									max_neighbor_count = std::max(max_neighbor_count, render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count);

									partial_visibility_neighbor_count = max_neighbor_count;
								}
								else
									// Disabled disocclusion boost, bringing the value back to its maximum before
									// disocclusion boost which is just the number of reused spatial neighbors
									partial_visibility_neighbor_count = render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;
										
								render_settings.restir_di_settings.spatial_pass.neighbor_visibility_count = partial_visibility_neighbor_count;
							}
							ImGuiRenderer::show_help_marker("If checked, the given number of neighbors will be reused for pixels that just got "
								"disoccluded due to camera movement (and thus that have no temporal history). This helps "
								"reduce noise in disoccluded regions.");
							if (render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost)
							{
								{
									ImGui::TreePush("Disocclusion boost tree");

									if (ImGui::SliderInt("Disoccluded Neighbor Reuse Count", &render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count, render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count, 16 + render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count))
									{
										m_render_window->set_render_dirty(true);

										// Updating the maximum
										max_neighbor_count = render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count;
										if (render_settings.restir_di_settings.spatial_pass.do_disocclusion_reuse_boost)
											max_neighbor_count = std::max(max_neighbor_count, render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count);

										if (will_need_to_update_partial_visibility)
										{
											// If the number of neighbors using visibility is set at the maximum, then we should
											// keep that value at the maximum as we modify the disoccluded neighbor reuse count
											max_neighbor_count = render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count;
											partial_visibility_neighbor_count = max_neighbor_count;
											render_settings.restir_di_settings.spatial_pass.neighbor_visibility_count = max_neighbor_count;
										}
									}
									ImGuiRenderer::show_help_marker("How many neighbors a pixel will reuse if that pixel just got disoccluded.");

									if (render_settings.restir_di_settings.spatial_pass.neighbor_visibility_count == render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count)
										// If the user is using the visibility in the target function of all spatial neighbors,
										// modifying that maximum number should still keep the visibility target function count
										// to the maximum
										render_settings.restir_di_settings.spatial_pass.neighbor_visibility_count = std::max(render_settings.restir_di_settings.spatial_pass.disocclusion_reuse_count, render_settings.restir_di_settings.spatial_pass.reuse_neighbor_count);

									ImGui::TreePop();
								}
							}

							if (ImGui::Checkbox("Neighbor Samples Random Rotation", &render_settings.restir_di_settings.spatial_pass.do_neighbor_rotation))
								m_render_window->set_render_dirty(true);
							ImGuiRenderer::show_help_marker("If checked, spatial neighbors sampled (using the Hammersley point set) "
								"will be randomly rotated. Because neighbor locations are generated with a Hammersley point set "
								"(deterministic), not rotating them results in every pixel of every rendered image reusing the "
								"same neighbor locations which decreases reuse efficiency.");

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
								ImGuiRenderer::show_help_marker("Allows trading bias for rendering performance by "
									"spatially reusing converged neighbors only with a certain probability instead of never / always."
									"\n\n 0.0 nevers reuses converged neighbors. No bias but performance impact."
									"\n\n 1.0 always reuses converged neighbors. Biased but no performance impact.");
							}
							ImGui::EndDisabled();

							if (ImGui::Checkbox("Debug Neighbor Reuse Positions", &render_settings.restir_di_settings.spatial_pass.debug_neighbor_location))
								m_render_window->set_render_dirty(true);
							ImGuiRenderer::show_help_marker("If checked, neighbor in the spatial reuse pass will be hardcoded to always be "
								"15 pixels to the right, not in a circle. This makes spotting bias easier when debugging.");
						}
					}

					ImGui::TreePop();
					ImGui::Dummy(ImVec2(0.0f, 20.0f));
				}





				if (ImGui::CollapsingHeader("Bias correction"))
				{
					ImGui::TreePush("Bias correction tree ReSTIR DI");

					{
						const char* bias_correction_mode_items[] = {
							"- 1/M Weights (Biased)",
							"- 1/Z Weights (Unbiased)",
							"- MIS-like Weights (Unbiased)",
							"- MIS Weights GBH (Unbiased)",
							"- Pairwise MIS Weights (Unbiased)",
							"- Pairwise MIS Weights Defensive (Unbiased)",
						};
						if (ImGui::Combo("Bias Correction Weights", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS), bias_correction_mode_items, IM_ARRAYSIZE(bias_correction_mode_items)))
						{
							m_renderer->recompile_kernels();

							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("What weights to use to resample reservoirs");

						bool disable_confidence_weights = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_M
							|| global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z;

						ImGui::BeginDisabled(disable_confidence_weights);
						if (ImGui::Checkbox("Use Confidence Weights", &render_settings.restir_di_settings.use_confidence_weights))
							m_render_window->set_render_dirty(true);
						std::string confidence_weight_help_string = "Whether or not to use confidence weights when resampling the samples. Confidence weights allow proper temporal reuse.";
						if (disable_confidence_weights)
							confidence_weight_help_string += "\n\nDisabled because 1/M or 1/Z weights use confidence weights by design.";
						ImGuiRenderer::show_help_marker(confidence_weight_help_string);
						ImGui::EndDisabled();

						// No visibility bias correction for 1/M weights
						bool bias_correction_visibility_disabled = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_M;
						static bool bias_correction_use_visibility = ReSTIR_DI_BiasCorrectionUseVisibility;
						ImGui::BeginDisabled(bias_correction_visibility_disabled);
						if (ImGui::Checkbox("Use visibility in bias correction", &bias_correction_use_visibility))
						{
							global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY, bias_correction_use_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
							m_renderer->recompile_kernels();

							m_render_window->set_render_dirty(true);
						}
						if (bias_correction_visibility_disabled)
							ImGuiRenderer::show_help_marker("Visibility bias correction cannot be used with 1/M weights.");
						ImGui::EndDisabled();
					}

					ImGui::TreePop();
					ImGui::Dummy(ImVec2(0.0f, 20.0f));
				}





				if (ImGui::CollapsingHeader("Later Bounces Sampling Strategy"))
				{
					ImGui::TreePush("Later Bounces tree");

					{
						const char* second_bounce_items[] = { "- Uniform one light", "- BSDF Sampling", "- MIS (1 Light + 1 BSDF)", "- RIS BDSF + Light candidates" };
						if (ImGui::Combo("Direct Lighting Strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY), second_bounce_items, IM_ARRAYSIZE(second_bounce_items)))
						{
							m_renderer->recompile_kernels();
							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("What direct lighting strategy to use for bounces that come after the first one (camera ray hit) since ReSTIR DI only applies on the first bounce.");

						switch (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY))
						{
						case RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT:
							break;

						case RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF:
							break;

						case RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT:
						{
							static bool use_visibility_ris_target_function = RISUseVisiblityTargetFunction;
							if (ImGui::Checkbox("Use visibility in RIS target function", &use_visibility_ris_target_function))
							{
								global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, use_visibility_ris_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
								m_renderer->recompile_kernels();

								m_render_window->set_render_dirty(true);
							}

							if (ImGui::SliderInt("RIS # of BSDF candidates", &render_settings.ris_settings.number_of_bsdf_candidates, 0, 16))
							{
								// Clamping to 0
								render_settings.ris_settings.number_of_bsdf_candidates = std::max(0, render_settings.ris_settings.number_of_bsdf_candidates);

								m_render_window->set_render_dirty(true);
							}

							if (ImGui::SliderInt("RIS # of light candidates", &render_settings.ris_settings.number_of_light_candidates, 0, 32))
							{
								// Clamping to 0
								render_settings.ris_settings.number_of_light_candidates = std::max(0, render_settings.ris_settings.number_of_light_candidates);

								m_render_window->set_render_dirty(true);
							}

							break;
						}

						default:
							break;
						}
					}

					ImGui::TreePop();

					break;
				}
			}

			break;

			default:
				break;
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Envmap lighting"))
		{
			ImGui::TreePush("Envmap sampling tree");

			const char* items[] = { "- No envmap importance sampling", "- Importance Sampling - Binary Search", "- Importance Sampling - Alias Table " };
			if (ImGui::Combo("Envmap sampling strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
			{
				ThreadManager::start_thread("RecomputeEnvmapSamplingStructure", [this]() {
					m_renderer->get_envmap().recompute_sampling_data_structure(m_renderer.get());
					});

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);

				ThreadManager::join_threads("RecomputeEnvmapSamplingStructure");
			}

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) != ESS_NO_SAMPLING)
			{
				// If we do have an importance sampling strategy
				static bool do_envmap_bsdf_mis = EnvmapSamplingDoBSDFMIS;
				if (ImGui::Checkbox("Do MIS with BSDF", &do_envmap_bsdf_mis))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS, do_envmap_bsdf_mis ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("");
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::display_ReSTIR_DI_bias_status(std::shared_ptr<GPUKernelCompilerOptions> kernel_options)
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

	if ((kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE
		|| (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE && render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass))
		&& kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY) == KERNEL_OPTION_FALSE)
	{
		std::string prefix;
		if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE)
			prefix = " - Initial ";
		else if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE && render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
			prefix = " - Spatial ";

		bias_reasons.push_back(prefix + "target function visibility without\n"
			"    visibility in bias correction");
		hover_explanations.push_back("When using the visibility term in the target function used to "
			"produce initial candidate samples (or temporally/spatially resample), all remaining samples are unoccluded.\n"
			"Temporal & spatial reuse passes will then only resample on unoccluded samples.\n"
			"If not accounting for visibility when counting valid neighbors (visibility in bias correction), we may determine "
			"that a neighbor could have produced the picked sample when actually, it couldn't "
			"because from the neighbor's point of view, the sample could have been occluded "
			"(visibility term in target function).\n"
			"This overestimates the number of valid neighbors and results in darkening.\n\n");
	}

	if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE) == KERNEL_OPTION_FALSE
		&& kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_FALSE
		&& kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY) == KERNEL_OPTION_TRUE
		&& (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_1_OVER_Z
			|| kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS
			|| kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS) == RESTIR_DI_BIAS_CORRECTION_PAIRWISE_MIS_DEFENSIVE))
	{
		bias_reasons.push_back("- Visibility in bias correction without\n"
			"visibility reuse (or initial candidates visibility)");
		hover_explanations.push_back("When taking visibility into account in the counting of "
			"valid neighbors (visibility in bias correction), we're going to assume that if the picked sample (from resampling "
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
			"This is an issue with 1/Z weights (and pairwise-MIS) because MIS-like and proper MIS "
			"(generalized balance heuristic/GBH) weights do not blindly overweight a sample as "
			"1/Z does (and then hopes that we divide by Z accordingly).");
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

	if (!render_settings.restir_di_settings.do_final_shading_visibility)
	{
		bias_reasons.push_back("- Not using final shading visibility");
		hover_explanations.push_back("Not using visibility during the final shading of samples "
			"produced by ReSTIR leads to \"missing\" shadows and an overall brightening of the "
			"scene because light samples are assumed unoccluded when they actually aren't.");
	}

	if (!bias_reasons.empty())
	{
		ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Biased");
		ImGui::TreePush("Bias reasons");

		for (int i = 0; i < bias_reasons.size(); i++)
		{
			ImGui::Text("%s", bias_reasons[i].c_str());
			ImGuiRenderer::add_tooltip(hover_explanations[i].c_str());
			ImGuiRenderer::show_help_marker(hover_explanations[i].c_str());

		}
		ImGui::TreePop();

	}
	else
		ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Unbiased");
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiSettingsWindow::draw_objects_panel()
{
	if (!ImGui::CollapsingHeader("Objects"))
		return;
	ImGui::TreePush("Objects tree");

	// Keeping a backup of the materials. Useful when modifying the global emissive factor
	// of objects in the scene because we want the global factor to affect the original
	// emission of the materials, not the emission that has already been multiplied by
	// a previous factor: this would lead to a buggy exponential growth of the emission
	static std::vector<RendererMaterial> original_materials = m_renderer->get_materials();

	std::vector<RendererMaterial> materials = m_renderer->get_materials();
	std::vector<std::string> material_names = m_renderer->get_material_names();

	bool material_changed = false;
	static int currently_selected_material = 0;

	std::vector<const char*> items = { "- None", "- Lambertian BRDF", "- Oren Nayar BRDF", "- Disney BSDF" };
	if (ImGui::Combo("All Objects BSDF Override", m_renderer->get_global_compiler_options()->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::BSDF_OVERRIDE), items.data(), items.size()))
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

		static float global_emissive_objects_factor = 1.0f;
		if (ImGui::SliderFloat("Global Emissive Objects Factor", &global_emissive_objects_factor, 0.0f, 10.0f))
		{
			for (int n = 0; n < original_materials.size(); n++)
			{
				if (original_materials[n].is_emissive())
				{
					materials[n].set_emission(original_materials[n].get_original_emission() * global_emissive_objects_factor);

					materials[n].make_safe();
					materials[n].precompute_properties();
				}
			}

			// TODO we would need to recompute the alias table for the emissive lights here
			m_renderer->update_materials(materials);
			m_render_window->set_render_dirty(true);
		}
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

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
		std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_global_compiler_options();
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
		material_changed |= ImGui::SliderFloat("Transmission", &material.specular_transmission, 0.0f, 1.0f);

		if (material.specular_transmission > 0.0f)
		{
			material_changed |= ImGui::SliderFloat("Absorption distance", &material.absorption_at_distance, 0.0f, 20.0f);
			material_changed |= ImGui::ColorEdit3("Absorption color", (float*)&material.absorption_color);

			ImGui::BeginDisabled(kernel_options->get_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITIES);
			material_changed |= ImGui::SliderInt("Dielectric priority", &material.dielectric_priority, 1, StackPriorityEntry::PRIORITY_MAXIMUM);
			if (kernel_options->get_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITIES)
				ImGuiRenderer::show_help_marker("Disabled because not using nested dielectrics with priorities.");
			ImGui::EndDisabled();
		}

		ImGui::Separator();
		// Displaying original emission
		ImGui::BeginDisabled(material.emission_texture_index > 0);
		ColorRGB32F material_emission = material.get_emission() / material.emission_strength;
		if (ImGui::ColorEdit3("Emission", (float*)&material_emission, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float))
		{
			material.set_emission(material_emission / material.emission_strength);

			material_changed = true;
		}
		ImGui::EndDisabled();
		if (material.emission_texture_index > 0)
			ImGuiRenderer::show_help_marker("Disabled because the emission of this material is controlled by a texture");

		material_changed |= ImGui::SliderFloat("Emission Strength", &material.emission_strength, 0.0f, 10.0f);
		material_changed |= ImGui::SliderFloat("Opacity", &material.alpha_opacity, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);

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

void ImGuiSettingsWindow::draw_denoiser_panel()
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

void ImGuiSettingsWindow::draw_post_process_panel()
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

void ImGuiSettingsWindow::draw_performance_settings_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (!ImGui::CollapsingHeader("Performance Settings"))
		return;

	ImGui::TreePush("Performance settings tree");

	ImGui::Text("Device: %s", m_renderer->get_device_properties().name);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_global_compiler_options();
	HardwareAccelerationSupport hwi_supported = m_renderer->device_supports_hardware_acceleration();

	ImGui::SeparatorText("General Settings");

	static bool use_hardware_acceleration = kernel_options->has_macro("__USE_HWI__");
	ImGui::BeginDisabled(hwi_supported != HardwareAccelerationSupport::SUPPORTED);
	if (ImGui::Checkbox("Use ray tracing hardware acceleration", &use_hardware_acceleration))
	{
		kernel_options->set_macro_value("__USE_HWI__", use_hardware_acceleration);

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

	ImGui::SeparatorText("Kernel Settings");
	ImGui::TreePush("Shared/global stack Traversal Options Tree");

	{
		// List of exceptions because these kernels do not trace any rays
		static std::unordered_set<std::string> exceptions = { ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID };
		static std::vector<std::string> kernel_names;
		static std::map<std::string, GPUKernel*> kernels = m_renderer->get_kernels();
		if (kernel_names.empty())
		{
			for (const auto& name_to_kernel : kernels)
				if (exceptions.find(name_to_kernel.first) == exceptions.end())
					kernel_names.push_back(name_to_kernel.first);
		}

		static std::string selected_kernel_name = GPURenderer::CAMERA_RAYS_KERNEL_ID;
		static GPUKernel* selected_kernel = kernels[selected_kernel_name];
		static GPUKernelCompilerOptions* selected_kernel_options = &selected_kernel->get_kernel_options();

		if (ImGui::BeginCombo("Kernel", selected_kernel_name.c_str()))
		{
			for (const std::string& kernel_name : kernel_names)
			{
				const bool is_selected = (selected_kernel_name == kernel_name);
				if (ImGui::Selectable(kernel_name.c_str(), is_selected))
				{
					selected_kernel_name = kernel_name;
					selected_kernel = kernels[selected_kernel_name];
					selected_kernel_options = &selected_kernel->get_kernel_options();
				}

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}




		ImGui::TreePush("Kernel selection for stack size");

		{
			static std::unordered_map<std::string, bool> use_shared_stack_traversal;
			if (use_shared_stack_traversal.find(selected_kernel_name) == use_shared_stack_traversal.end())
				use_shared_stack_traversal[selected_kernel_name] = selected_kernel_options->get_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL);
			bool& use_shared_stack_traversal_bool = use_shared_stack_traversal[selected_kernel_name];

			if (ImGui::Checkbox("Use shared/global stack BVH traversal", &use_shared_stack_traversal_bool))
			{
				selected_kernel_options->set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL, use_shared_stack_traversal_bool ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("If checked, shared memory + a globally allocated buffer will be used for the BVH "
				"traversal of rays.\n"
				"This incurs an additional cost in VRAM but improves traversal performance.");





			if (use_shared_stack_traversal_bool)
			{
				static std::unordered_map<std::string, int> pending_stack_size_changes;
				if (pending_stack_size_changes.find(selected_kernel_name) == pending_stack_size_changes.end())
					pending_stack_size_changes[selected_kernel_name] = selected_kernel_options->get_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE);
				int& pending_stack_size = pending_stack_size_changes[selected_kernel_name];

				if (ImGui::InputInt("Shared stack size", &pending_stack_size))
					pending_stack_size = std::max(0, pending_stack_size);

				ImGuiRenderer::show_help_marker("Fast shared memory stack used for the BVH traversal of \"global\" rays (rays that search for a closest hit with no maximum distance)\n\n"
					"Allocating more of this speeds up the BVH traversal but reduces the amount of L1 cache available to "
					"the rest of the shader which thus reduces its performance. A tradeoff must be made.\n\n"
					"If this shared memory stack isn't large enough for traversing the BVH, then "
					"it is complemented by using the global stack buffer. If both combined aren't enough "
					"for the traversal, then artifacts start showing up in renders.\n\n"
					"Note that setting this value to 0 disables the shared stack usage but still uses the global buffer "
					"for traversal. This approach is still better that not using any of these two memories at all (this "
					"becomes the case when the checkboxes above are not checked.)");

				if (pending_stack_size != selected_kernel_options->get_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE))
				{
					// If the user has modified the size of the shared stack, showing a button to apply the changes 
					// (not applying the changes everytime because this requires a recompilation of basically all shaders and that's heavy)

					ImGui::TreePush("Apply button shared stack size");
					if (ImGui::Button("Apply"))
					{
						selected_kernel_options->set_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE, pending_stack_size);
						m_renderer->recompile_kernels();
						m_render_window->set_render_dirty(true);
					}
					ImGui::TreePop();
				}
			}
		}

		ImGui::TreePop();


		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputInt("Global stack per-thread size", &m_renderer->get_render_data().global_traversal_stack_buffer_size))
		{
			m_renderer->get_render_data().global_traversal_stack_buffer_size = std::max(0, m_renderer->get_render_data().global_traversal_stack_buffer_size);
			m_render_window->set_render_dirty(true);
		}

		ImGuiRenderer::show_help_marker("Size of the global stack buffer for each thread. Used for complementing the shared memory stack allocated in the kernels."
			"A good value for this parameter is scene-complexity dependent.\n\n"
			"A lower value will use less VRAM but will start introducing artifacts if the value is too low due "
			"to insufficient stack size for the BVH traversal.\n\n"
			"16 seems to be a good value to start with. If lowering this value improves performance, then that "
			"means that the BVH traversal is starting to suffer (the traversal is incomplete --> improved performance) "
			"and rendering artifacts will start to show up.");

		std::string size_string = "Global Stack Buffer VRAM Usage: ";
		size_string += std::to_string(m_renderer->get_render_data().global_traversal_stack_buffer_size * std::ceil(m_renderer->m_render_resolution.x / 8.0f) * 8.0f * std::ceil(m_renderer->m_render_resolution.y / 8.0f) * 8.0f * sizeof(int) / 1000000.0f);
		size_string += " MB";
		ImGui::Text("%s", size_string.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	ImGui::SeparatorText("Lighting Settings");
	ImGui::TreePush("Lighting Settings Performance Tree");
	{
		if (ImGui::SliderFloat("Minimum Light Contribution", &render_settings.minimum_light_contribution, 0.0f, 10.0f))
		{
			render_settings.minimum_light_contribution = std::max(0.0f, render_settings.minimum_light_contribution);
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("If a selected light (for direct lighting estimation) contributes at a given "
			" point less than this 'minimum_light_contribution' value then the light sample is discarded. "
			"This can improve performance at the cost of some bias depending on the scene.\n"
			"0.0f to disable");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	ImGui::TreePop();
}

void ImGuiSettingsWindow::draw_performance_metrics_panel()
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

	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::CAMERA_RAYS_KERNEL_ID, "Camera rays pass");
	if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI)
	{
		if (m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_LIGHTS_PRESAMPLING) == KERNEL_OPTION_TRUE)
			draw_perf_metric_specific_panel(m_render_window_perf_metrics, ReSTIRDIRenderPass::RESTIR_DI_LIGHTS_PRESAMPLING_KERNEL_ID, "ReSTIR Light Presampling");

		draw_perf_metric_specific_panel(m_render_window_perf_metrics, ReSTIRDIRenderPass::RESTIR_DI_INITIAL_CANDIDATES_KERNEL_ID, "ReSTIR Initial Candidates");

		if (render_settings.restir_di_settings.do_fused_spatiotemporal)
		{
			draw_perf_metric_specific_panel(m_render_window_perf_metrics, ReSTIRDIRenderPass::RESTIR_DI_SPATIOTEMPORAL_REUSE_KERNEL_ID, "ReSTIR Spatio-Temporal Reuse");
			if (render_settings.restir_di_settings.spatial_pass.number_of_passes > 1)
			{
				std::string spatial_reuse_text = "ReSTIR " + std::to_string(render_settings.restir_di_settings.spatial_pass.number_of_passes - 1) + " Spatial Reuse";
				draw_perf_metric_specific_panel(m_render_window_perf_metrics, ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID, spatial_reuse_text);
			}
		}
		else
		{
			if (render_settings.restir_di_settings.temporal_pass.do_temporal_reuse_pass)
				draw_perf_metric_specific_panel(m_render_window_perf_metrics, ReSTIRDIRenderPass::RESTIR_DI_TEMPORAL_REUSE_KERNEL_ID, "ReSTIR Temporal Reuse"); if (render_settings.restir_di_settings.spatial_pass.number_of_passes > 1)
				if (render_settings.restir_di_settings.spatial_pass.do_spatial_reuse_pass)
				{
					std::string spatial_reuse_text = "ReSTIR " + std::to_string(render_settings.restir_di_settings.spatial_pass.number_of_passes) + " Spatial Reuse";
					draw_perf_metric_specific_panel(m_render_window_perf_metrics, ReSTIRDIRenderPass::RESTIR_DI_SPATIAL_REUSE_KERNEL_ID, spatial_reuse_text);
				}
		}
	}
	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::PATH_TRACING_KERNEL_ID, "Path Tracing Pass");
	ImGui::Separator();
	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::FULL_FRAME_TIME_KEY, "Total Sample Time");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	ImGui::TreePop();
}

void ImGuiSettingsWindow::draw_perf_metric_specific_panel(std::shared_ptr<PerformanceMetricsComputer> perf_metrics, const std::string& perf_metrics_key, const std::string& label)
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
std::string ImGuiSettingsWindow::format_perf_metrics_tooltip_line(const std::string& label, const std::string& suffix, const std::string& longest_header_for_padding, const std::string& formatter_after_header, const Args& ...args)
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

void ImGuiSettingsWindow::draw_debug_panel()
{
	if (!ImGui::CollapsingHeader("Debug"))
		return;

	if (ImGui::Checkbox("Show NaNs", &m_renderer->get_render_settings().display_NaNs))
		m_render_window->set_render_dirty(true);
	ImGuiRenderer::show_help_marker("If true, NaNs that occur during the rendering will show up as pink pixels.");

	ImGui::TreePush("Debug tree");
	ImGui::TreePop();
}
