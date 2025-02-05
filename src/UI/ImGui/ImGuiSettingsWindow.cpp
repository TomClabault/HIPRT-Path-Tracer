/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompiler.h"
#include "HostDeviceCommon/RenderSettings.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/FillGBufferRenderPass.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Scene/CameraAnimation.h"
#include "Threads/ThreadManager.h"
#include "UI/ImGui/ImGuiRenderer.h"
#include "UI/ImGui/ImGuiSettingsWindow.h"
#include "UI/RenderWindow.h"

#include <iostream>

extern GPUKernelCompiler g_gpu_kernel_compiler;

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

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::SeparatorText("General render settings");
	draw_render_settings_panel();
	draw_camera_panel();
	draw_environment_panel();
	draw_sampling_panel();
	draw_denoiser_panel();
	draw_post_process_panel();
	draw_quality_panel();

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::SeparatorText("Other settings");
	draw_performance_settings_panel();
	draw_performance_metrics_panel();
	draw_shader_kernels_panel();
	draw_debug_panel();

	// For a little bit of space at the very bottom of the window
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

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
		ImGui::Text("Frame time (GPU): %.3fms", m_render_window_perf_metrics->get_current_value(GPURenderer::ALL_RENDER_PASSES_TIME_KEY));
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", render_settings.sample_number, m_render_window->get_samples_per_second(), m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);
	float time_before_viewport_refresh_ms = m_render_window->get_time_ms_before_viewport_refresh();
	if (!m_render_window->is_rendering_done())
	{
		// Only displaying the refresh timer if we actually need to wait before refreshin'
		// And also, not displaying that if the rendering is done

		float time_before_refresh_seconds = time_before_viewport_refresh_ms / 1000.0f;
		if (time_before_refresh_seconds > 0.0f)
			ImGui::Text("Viewport refresh in: %.3fs", time_before_refresh_seconds);
		else
		{
			// Time is < 0.0f i.e. the timer has expired and we're waiting for a refresh
			if (m_renderer->get_gmon_render_pass()->is_render_pass_used() && m_renderer->get_gmon_render_pass()->recomputation_requested())
				// If we're waiting for GMoN, indicating it
				ImGui::Text("Viewport refresh in: 0.000s --- Waiting for GMoN");
			else
				// If we're not waiting for GMoN, just clampign so that we don't display negative values
				ImGui::Text("Viewport refresh in: %.3fs", std::max(0.0f, time_before_refresh_seconds));
		}
	}
	else
		// If the rendering is done, displaying 0.000s
		ImGui::Text("Viewport refresh in: 0.000s");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
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

	if (ImGui::InputInt("Debug seed", &render_settings.restir_gi_settings.debug_seed))
		m_render_window->set_render_dirty(true);
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
	display_view_selector();

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
	ImGuiRenderer::show_help_marker("Keeps approximately the same render resolution when "
									"resizing the application's window.");

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
	draw_russian_roulette_options();

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (ImGui::CollapsingHeader("Render stopping condition"))
	{
		ImGui::TreePush("Stopping condition tree");
		{
			if (ImGui::InputInt("Max Sample Count", &m_application_settings->max_sample_count))
				m_application_settings->max_sample_count = std::max(m_application_settings->max_sample_count, 0);
			if (m_renderer->get_gmon_render_pass()->is_render_pass_used())
			{
				// Using GMoN

				unsigned int number_of_sets = m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT);
				if (m_application_settings->max_sample_count % number_of_sets != 0)
				{
					ImGui::TreePush("Number of samples not divisible GMoN tree");

					// But the maximum number of samples isn't divisible by the number of sets
					ImGui::Text("Warning: ");
					std::string warning_text = "Currently using GMoN (\"Post-processing\" panel) but the number of "
						"maximum samples entered here isn't divisible by the number of GMoN sets. This means that "
						"what's displayed in the viewport will only be " 
						+ std::to_string(std::max(1u, m_application_settings->max_sample_count / number_of_sets)) + " samples instead of "
						+ std::to_string(m_application_settings->max_sample_count) + ".\n\n"
						""
						"You click the button to the right to round up the maximum number of samples to one that is "
						"divisible by the number of GMoN sets (" 
						+ std::to_string(m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT)) + ")";
					ImGuiRenderer::show_help_marker(warning_text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));

					ImGui::SameLine();
					if (ImGui::Button("Round up"))
						m_application_settings->max_sample_count = std::ceil(m_application_settings->max_sample_count / static_cast<float>(number_of_sets)) * number_of_sets;

					ImGui::TreePop();
				}
			}

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
			ImGui::Checkbox("Use pixel noise threshold stopping condition", &render_settings.enable_pixel_stop_noise_threshold);
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

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiSettingsWindow::draw_russian_roulette_options()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (ImGui::Checkbox("Do Russian Roulette", &render_settings.use_russian_roulette))
		m_render_window->set_render_dirty(true);

	const char* items[] = { "- Max throughput", "- Arnold, Langlands, 2014" };
	if (ImGui::Combo("Termination method", (int*)&render_settings.path_russian_roulette_method, items, IM_ARRAYSIZE(items)))
		m_render_window->set_render_dirty(true);

	static bool min_depth_modified = false;
	if (!min_depth_modified)
		render_settings.russian_roulette_min_depth = std::min(5, render_settings.nb_bounces / 2);
	if (ImGui::SliderInt("RR Min Depth", &render_settings.russian_roulette_min_depth, 0, render_settings.nb_bounces + 1))
	{
		m_render_window->set_render_dirty(true);
		min_depth_modified = true;
	}
	ImGuiRenderer::show_help_marker("After how many bounces can russian roulette kick in? "
									"For example, 0 means that the camera ray hits, and then the next bounce "
									"is already susceptible to russian roulette kill. 1 would mean that the first "
									"bounce is never going to be cutoff by the russian roulette.");
	if (ImGui::SliderFloat("RR Throughput Clamp", &render_settings.russian_roulette_throughput_clamp, 1.0f, 20.0f))
		m_render_window->set_render_dirty(true);
	ImGuiRenderer::show_help_marker("After applying russian roulette (dividing by the continuation probability) "
									"the energy added to the ray throughput is clamped to this maximum value.\n"
									"\n"
									"This is biased and darkens the image the lower the threshold but it helps "
									"reduce variance and fireflies introduced by the russian roulette --> faster "
									"convergence.\n"
									"\n"
									"0 for no clamping.");
}

void ImGuiSettingsWindow::display_view_selector()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	std::shared_ptr<DisplayViewSystem> display_view_system = m_render_window->get_display_view_system();

	static std::vector<std::pair<const char*, DisplayViewType>> display_string_to_type = {
		{ "- Default", DisplayViewType::DEFAULT },
		{ "- GMoN blend", DisplayViewType::GMON_BLEND },
		{ "- Denoiser blend", DisplayViewType::DENOISED_BLEND },
		{ "- Denoiser - Normals", DisplayViewType::DISPLAY_DENOISER_NORMALS },
		{ "- Denoiser - Albedo", DisplayViewType::DISPLAY_DENOISER_ALBEDO },
		{ "- Pixel convergence heatmap", DisplayViewType::PIXEL_CONVERGENCE_HEATMAP },
		{ "- Converged pixels map", DisplayViewType::PIXEL_CONVERGED_MAP },
		{ "- White Furnace Threshold", DisplayViewType::WHITE_FURNACE_THRESHOLD }
	};

	std::vector<const char*> items;
	for (auto view_string_to_DisplayViewType : display_string_to_type)
		items.push_back(view_string_to_DisplayViewType.first);

	int display_view_selected_index = display_view_system->get_current_display_view_type();

	if (ImGui::BeginCombo("Display View", items[display_view_selected_index]))
	{
		for (int i = 0; i < items.size(); i++)
		{
			const bool is_selected = (display_view_selected_index == i);
			bool display_view_is_disabled = display_view_disabled(display_string_to_type[i].second);

			if (display_view_is_disabled)
				ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
			if (ImGui::Selectable(items[i], is_selected))
			{
				display_view_selected_index = i;

				if (display_view_is_disabled)
					// If we clicked on a display that was disabled, there is an
					// action to do to enable all the necessary for the display view to work
					display_view_disabled_action(display_string_to_type[i].second);
				display_view_system->queue_display_view_change(static_cast<DisplayViewType>(display_view_selected_index));
				m_render_window->set_force_viewport_refresh(true);
			}
			if (display_view_is_disabled)
			{
				ImGui::PopStyleColor();
				display_view_tooltip(display_string_to_type[i].second);
			}

			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	DisplaySettings& display_settings = display_view_system->get_display_settings();
	DisplayViewType display_view_type_selected = static_cast<DisplayViewType>(display_view_selected_index);
	// Adding some more UI elements for certain display views
	switch (display_view_type_selected)
	{
		case DisplayViewType::WHITE_FURNACE_THRESHOLD:
			ImGui::Checkbox("Use low threshold", &display_settings.white_furnace_display_use_low_threshold);
			ImGuiRenderer::show_help_marker("If checked, the white furnace threshold shader will display "
											"pixel that lose energy as green. Pixels will not be highlighted "
											"if unchecked");
			ImGui::Checkbox("Use high threshold", &display_settings.white_furnace_display_use_high_threshold);
			ImGuiRenderer::show_help_marker("If checked, the white furnace threshold shader will display "
											"pixel that gain energy as red. Pixels will not be highlighted "
											"if unchecked");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			break;
	}
}

bool ImGuiSettingsWindow::display_view_disabled(DisplayViewType display_view_type)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	switch (display_view_type)
	{
	case DisplayViewType::PIXEL_CONVERGED_MAP:
	case DisplayViewType::PIXEL_CONVERGENCE_HEATMAP:
		return !render_settings.has_access_to_adaptive_sampling_buffers();

	case DisplayViewType::GMON_BLEND:
		return !m_renderer->get_gmon_render_pass()->is_render_pass_used();

	case DisplayViewType::DENOISED_BLEND:
		return !m_application_settings->enable_denoising;

	default:
		break;
	}

	return false;
}

void ImGuiSettingsWindow::display_view_tooltip(DisplayViewType display_view_type)
{
	switch (display_view_type)
	{
	case DisplayViewType::PIXEL_CONVERGED_MAP:
	case DisplayViewType::PIXEL_CONVERGENCE_HEATMAP:
		ImGuiRenderer::add_tooltip("This display view is unavailabe because adaptive sampling isn't in use. Click to enable adaptive sampling.");
		return;

	case DisplayViewType::GMON_BLEND:
		ImGuiRenderer::add_tooltip("This display view is disabled because GMoN isn't in use. Click to enable GMoN.");
		return;

	case DisplayViewType::DENOISED_BLEND:
		ImGuiRenderer::add_tooltip("This display view is disabled because the denoiser isn't enabled. Click to enable the denoiser.");
		return;

	default:
		break;
	}
}

void ImGuiSettingsWindow::display_view_disabled_action(DisplayViewType display_view_type)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	switch (display_view_type)
	{
	case DisplayViewType::PIXEL_CONVERGED_MAP:
	case DisplayViewType::PIXEL_CONVERGENCE_HEATMAP:
		render_settings.enable_adaptive_sampling = true;
			
		m_render_window->set_render_dirty(true);

		return;

	case DisplayViewType::GMON_BLEND:
		// Enabling GMoN
		m_renderer->get_gmon_render_pass()->get_gmon_data().using_gmon = true;
		toggle_gmon();

		return;

	case DisplayViewType::DENOISED_BLEND:
		ImGuiRenderer::add_tooltip("This display view is disabled because the denoiser isn't enabled. Click to enable the denoiser.");
		return;

	default:
		break;
	}
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
	draw_camera_panel_static("Camera", m_render_window, m_renderer);
}

void ImGuiSettingsWindow::draw_camera_panel_static(const std::string& panel_title, RenderWindow* render_window, std::shared_ptr<GPURenderer> renderer)
{
	HIPRTRenderSettings& render_settings = renderer->get_render_settings();
	Camera& camera = renderer->get_camera();

	if (ImGui::CollapsingHeader(panel_title.c_str()))
	{
		ImGui::TreePush("Camera tree");

		ImGui::SeparatorText("Transformation");
		if (ImGui::DragFloat3("Position", reinterpret_cast<float*>(&camera.m_translation)))
			render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Settings");
		if (ImGui::Checkbox("Do ray jittering", &camera.do_jittering))
			render_window->set_render_dirty(true);

		static float camera_fov = camera.vertical_fov * M_INV_PI * 180.0f;
		if (ImGui::SliderFloat("FOV", &camera_fov, 0.0f, 180.0f, "%.3fdeg", ImGuiSliderFlags_AlwaysClamp))
		{
			camera.set_FOV(camera_fov / 180.0f * M_PI);

			render_window->set_render_dirty(true);
		}

		if (ImGui::SliderFloat("Camera Speed", &camera.user_movement_speed_multiplier, 0.0f, 10.0f))
			camera.user_movement_speed_multiplier = std::max(0.0f, camera.user_movement_speed_multiplier);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::BeginDisabled(!render_settings.accumulate);
		ImGui::Checkbox("Render low resolution when interacting", &render_settings.allow_render_low_resolution);
		if (!render_settings.accumulate)
			ImGuiRenderer::add_tooltip("Cannot render at low resolution when not accumulating. If you want to render at "
				"a lower resolution, you can use the resolution scale in \"Render Settings\"for that.");
		ImGui::SliderInt("Low resolution scale", &render_settings.render_low_resolution_scaling, 1, 8);
		if (!render_settings.accumulate)
			ImGuiRenderer::add_tooltip("Cannot render at low resolution when not accumulating. If you want to render at "
				"a lower resolution, you can use the resolution scale in \"Render Settings\"for that.");
		ImGui::EndDisabled();





		static int selected_object = 0;
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("Center camera on object");
		if (ImGui::BeginListBox("##center_on_object", ImVec2(-FLT_MIN, 7 * ImGui::GetTextLineHeightWithSpacing())))
		{
			const std::vector<std::string>& mesh_names = renderer->get_mesh_names();
			const std::vector<std::string>& material_names = renderer->get_material_names();
			for (int n = 0; n < mesh_names.size(); n++)
			{
				const bool is_selected = (selected_object == n);

				const std::string& mesh_name = mesh_names[n];
				const std::string& material_name = material_names[renderer->get_mesh_material_indices()[n]];
				std::string object_text = mesh_name + " (" + material_name + ")";
				if (ImGui::Selectable(object_text.c_str(), is_selected))
				{
					selected_object = n;

					float3 object_center = renderer->get_mesh_bounding_boxes()[n].get_center();
				}

				// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndListBox();
		}

		if (ImGui::Button("Center"))
		{
			camera.look_at_object(renderer->get_mesh_bounding_boxes()[selected_object]);

			render_window->set_render_dirty(true);
		}

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
	HIPRTRenderData& render_data = m_renderer->get_render_data();
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
			if (ImGui::InputInt("Minimum samples", &render_settings.adaptive_sampling_min_samples))
				m_render_window->set_render_dirty(true);
			if (ImGui::InputFloat("Noise threshold", &render_settings.adaptive_sampling_noise_threshold))
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

		//ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::SliderInt("Disabvle direct", &render_settings.enable_direct, 0, 1))
			m_render_window->set_render_dirty(true);
		if (ImGui::CollapsingHeader("Emissive geometry sampling"))
		{
			ImGui::TreePush("Direct lighting sampling tree");

			bool disabled = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_RESTIR_DI;
			ImGui::BeginDisabled(disabled);
			if (ImGui::SliderInt("NEE Samples", &render_settings.number_of_nee_samples, 1, 8))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker(std::string("How many light samples to take and shade per each vertex of the "
				"ray's path.\n"
				"\n"
				"Said otherwise, we're going to run next-event estimation that many "
				"times per each intersection point along the ray.\n"
				"\n"
				"This is good because this amortizes camera rays and bounce rays i.e. "
				"we get better shading quality for as many camera rays and bounce rays.\n"
				"\n"
				"With ReSTIR DI this only applies to the secondary bounces shading.") +
				(disabled ? std::string("\n\nDisabled because not supported by ReSTIR DI") : ""));
			ImGui::EndDisabled();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			const char* items[] = { "- No direct light sampling", "- Uniform one light", "- BSDF Sampling", "- MIS (1 Light + 1 BSDF)", "- RIS BDSF + Light candidates", "- ReSTIR DI (Primary Hit Only)" };
			if (ImGui::Combo("Sampling strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
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
				ImGui::Dummy(ImVec2(0.0f, -20.0f));
				break;

			case LSS_UNIFORM_ONE_LIGHT:
				ImGui::Dummy(ImVec2(0.0f, -20.0f));
				break;

			case LSS_MIS_LIGHT_BSDF:
				ImGui::Dummy(ImVec2(0.0f, -20.0f));
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
						if (ImGui::Checkbox("Use Final Visibility", &render_settings.restir_di_settings.do_final_shading_visibility))
							m_render_window->set_render_dirty(true);

						ImGui::Dummy(ImVec2(0.0f, 20.0f));
						if (ImGui::SliderInt("M-cap", &render_settings.restir_di_settings.m_cap, 0, 48))
						{
							render_settings.restir_di_settings.m_cap = std::max(0, render_settings.restir_di_settings.m_cap);
							if (render_settings.accumulate)
								m_render_window->set_render_dirty(true);
						}

						draw_ReSTIR_neighbor_heuristics_panel(render_settings.restir_di_settings);
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





				draw_ReSTIR_temporal_reuse_panel(render_settings.restir_di_settings, [this, &render_settings]() {
					if (render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass && render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
					{
						if (ImGui::Checkbox("Do Fused Spatiotemporal", &render_settings.restir_di_settings.do_fused_spatiotemporal))
						{
							render_settings.restir_di_settings.common_temporal_pass.temporal_buffer_clear_requested = true;

							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("If checked, the spatial and temporal pass will be fused into a single kernel call. "
							"This avoids a synchronization barrier between the temporal pass and the spatial pass "
							"and increases performance. Because the spatial must then resample without the output of the temporal pass, the spatial "
							"pass only resamples on the temporal reservoir buffer, not the temporal + initial candidates reservoir "
							"(which is the output of the temporal pass). This is usually imperceptible.");
					}

					if (ImGui::Checkbox("Do Temporal Reuse", &render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass))
					{
						m_render_window->set_render_dirty(true);

						if (!render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
							// Disabling fused spatiotemporal if we just disabled the temporal reuse
							render_settings.restir_di_settings.do_fused_spatiotemporal = false;
					}
				});

				draw_ReSTIR_spatial_reuse_panel<false>([&render_settings, this] () {
					if (render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass && render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass)
					{
						if (ImGui::Checkbox("Do Fused Spatiotemporal", &render_settings.restir_di_settings.do_fused_spatiotemporal))
						{
							render_settings.restir_di_settings.common_temporal_pass.temporal_buffer_clear_requested = true;

							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("If checked, the spatial and temporal pass will be fused into a single kernel call. "
							"This avois a synchronization barrier between the temporal pass and the spatial pass "
							"and increases performance. Because the spatial must then resample without the output of the temporal pass, the spatial "
							"pass only resamples on the temporal reservoir buffer, not the temporal + initial candidates reservoir "
							"(which is the output of the temporal pass). This is usually imperceptible.");
					}

					if (ImGui::Checkbox("Do Spatial Reuse", &render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass))
					{
						m_render_window->set_render_dirty(true);

						if (!render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
							// Disabling fused spatiotemporal if we just disabled the spatial reuse
							render_settings.restir_di_settings.do_fused_spatiotemporal = false;
					}
				});




				draw_ReSTIR_bias_correction_panel<false>();




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

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) != LSS_BSDF && global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) != LSS_NO_DIRECT_LIGHT_SAMPLING)
			{
				ImGui::Dummy(ImVec2(0.0f, 20.0f));

				draw_next_event_estimation_plus_plus_panel();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Envmap sampling"))
		{
			ImGui::TreePush("Envmap sampling tree");

			const char* items[] = { "- No envmap importance sampling", "- Importance Sampling - Binary Search", "- Importance Sampling - Alias Table " };
			if (ImGui::Combo("Sampling strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
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

		if (ImGui::CollapsingHeader("Path sampling"))
		{
			ImGui::TreePush("Path sampling tree");

			const char* items[] = { "- BSDF sampling", "- ReSTIR GI" };
			if (ImGui::Combo("Sampling strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			switch (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY))
			{
			case PSS_RESTIR_GI:
			{
				ImGui::TreePush("ReSTIR GI options tree");

				if (ImGui::CollapsingHeader("General Settings"))
				{
					ImGui::TreePush("ReSTIR GI - General Settings Tree");
					{
						ImGui::Dummy(ImVec2(0.0f, 20.0f));
						if (ImGui::SliderInt("M-cap", &render_settings.restir_di_settings.m_cap, 0, 48))
						{
							render_settings.restir_di_settings.m_cap = std::max(0, render_settings.restir_di_settings.m_cap);
							if (render_settings.accumulate)
								m_render_window->set_render_dirty(true);
						}

						draw_ReSTIR_neighbor_heuristics_panel(render_settings.restir_gi_settings);
					}
					ImGui::TreePop();
					ImGui::Dummy(ImVec2(0.0f, 20.0f));
				}

				draw_ReSTIR_temporal_reuse_panel(render_settings.restir_gi_settings, [&render_settings, this]() {
					if (ImGui::Checkbox("Do Temporal Reuse", &render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass))
						m_render_window->set_render_dirty(true);
				});
				draw_ReSTIR_spatial_reuse_panel<true>([&render_settings, this]() {
					if (ImGui::Checkbox("Do Spatial Reuse", &render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass))
						m_render_window->set_render_dirty(true);
				});

				draw_ReSTIR_bias_correction_panel<true>();

				if (ImGui::CollapsingHeader("Debug"))
				{
					ImGui::TreePush("ReSTIR GI options tree");

					const char* debug_view_items[] = { "No debug view", "- Final reservoir UCW", "- Final reservoir target function", "- Final reservoir weight sum"};
					if (ImGui::Combo("Debug view", (int*)&render_settings.restir_gi_settings.debug_view, debug_view_items, IM_ARRAYSIZE(debug_view_items)))
						m_render_window->set_render_dirty(true);
					if (ImGui::SliderFloat("Debug view scale factor", &render_settings.restir_gi_settings.debug_view_scale_factor, 0.0f, 3.0f))
						m_render_window->set_render_dirty(true);

					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
			default:
				break;
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Material sampling"))
		{
			ImGui::TreePush("Sampling Materials Tree");
			draw_principled_bsdf_energy_conservation();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Principled BSDF Diffuse Lobe");
			const char* items[] = { "- Lambertian", "- Oren-Nayar" };
			if (ImGui::Combo("Diffuse Lobe", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE), items, IM_ARRAYSIZE(items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("GGX");

			std::vector<const char*> ggx_sampling_items = { "- VNDF", "- VNDF Spherical Caps" };
			if (ImGui::Combo("GGX Sampling Method", m_renderer->get_global_compiler_options()->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::GGX_SAMPLE_FUNCTION), ggx_sampling_items.data(), ggx_sampling_items.size()))
			{
				m_renderer->recompile_kernels();

				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("How to sample the GGX NDF");

			std::vector<const char*> masking_shadowing_items = { "- Smith height-correlated", "- Smith height-uncorrelated" };
			if (ImGui::Combo("GGX Masking-Shadowing", (int*)&render_data.bsdfs_data.GGX_masking_shadowing, masking_shadowing_items.data(), masking_shadowing_items.size()))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("Which masking-shadowing term to use with the GGX NDF.");

			if (render_data.bsdfs_data.GGX_masking_shadowing == GGXMaskingShadowingFlavor::HeightUncorrelated && global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION) == KERNEL_OPTION_TRUE)
			{
				ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Warning: ");
				ImGui::SameLine();
				ImGui::Text("Multiple-scattering energy compensation look-up tables \n"
							"were not precomputed for smith height-uncorrelated \n"
							"masking-shadowing term.\n"
							"\n"
							"Energy conservation is not guaranteed.");
			}

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_ReSTIR_neighbor_heuristics_panel(ReSTIRCommonSettings& common_settings)
{
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	static bool use_heuristics_at_all = true;
	static bool use_normal_heuristic_backup = common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic;
	static bool use_plane_distance_heuristic_backup = common_settings.neighbor_similarity_settings.use_plane_distance_heuristic;
	static bool use_roughness_heuristic_backup = common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic;
	if (ImGui::Checkbox("Use Heuristics for neighbor rejection", &use_heuristics_at_all))
	{
		if (!use_heuristics_at_all)
		{
			// Saving the usage of the heuristics for later restoration
			use_normal_heuristic_backup = common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic;
			use_plane_distance_heuristic_backup = common_settings.neighbor_similarity_settings.use_plane_distance_heuristic;
			use_roughness_heuristic_backup = common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic;

			common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic = false;
			common_settings.neighbor_similarity_settings.use_plane_distance_heuristic = false;
			common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		}
		else
		{
			// Restoring heuristics usage to their backup values
			common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic = use_normal_heuristic_backup;
			common_settings.neighbor_similarity_settings.use_plane_distance_heuristic = use_plane_distance_heuristic_backup;
			common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = use_roughness_heuristic_backup;
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




		if (ImGui::Checkbox("Use Normal Similarity Heuristic", &common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic))
			m_render_window->set_render_dirty(true);

		if (common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic)
		{
			if (ImGui::SliderFloat("Normal Similarity Threshold", &common_settings.neighbor_similarity_settings.normal_similarity_angle_degrees, 0.1f, 90.0f, "%.3f deg", ImGuiSliderFlags_AlwaysClamp))
			{
				common_settings.neighbor_similarity_settings.normal_similarity_angle_precomp = std::cos(common_settings.neighbor_similarity_settings.normal_similarity_angle_degrees * M_PI / 180.0f);

				m_render_window->set_render_dirty(true);
			}
		}




		if (ImGui::Checkbox("Use Plane Distance Heuristic", &common_settings.neighbor_similarity_settings.use_plane_distance_heuristic))
			m_render_window->set_render_dirty(true);

		if (common_settings.neighbor_similarity_settings.use_plane_distance_heuristic)
			if (ImGui::SliderFloat("Plane Distance Threshold", &common_settings.neighbor_similarity_settings.plane_distance_threshold, 0.0f, 1.0f))
				m_render_window->set_render_dirty(true);




		if (ImGui::Checkbox("Use Roughness Heuristic", &common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic))
			m_render_window->set_render_dirty(true);

		if (common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic)
			if (ImGui::SliderFloat("Roughness Threshold", &common_settings.neighbor_similarity_settings.roughness_similarity_threshold, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp))
				m_render_window->set_render_dirty(true);



		// ReSTIR DI Heursitics Tree
		ImGui::TreePop();
	}
}

template <typename CommonReSTIRSettings>
void ImGuiSettingsWindow::draw_ReSTIR_temporal_reuse_panel(CommonReSTIRSettings& restir_settings, std::function<void(void)> draw_before_panel)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (ImGui::CollapsingHeader("Temporal Reuse Pass"))
	{
		ImGui::PushID(&restir_settings);
		ImGui::TreePush("ReSTIR - Temporal Reuse Pass Tree");
		{
			draw_before_panel();

			if (restir_settings.common_temporal_pass.do_temporal_reuse_pass)
			{
				// Same line as "Do Temporal Reuse"
				ImGui::SameLine();
				if (ImGui::Button("Reset Temporal Reservoirs"))
				{
					restir_settings.common_temporal_pass.temporal_buffer_clear_requested = true;
					m_render_window->set_render_dirty(true);
				}

				bool last_frame_g_buffer_needed = true;
				last_frame_g_buffer_needed &= !render_settings.accumulate;
				last_frame_g_buffer_needed &= restir_settings.common_temporal_pass.do_temporal_reuse_pass;

				if (ImGui::SliderInt("Max temporal neighbor search count", &restir_settings.common_temporal_pass.max_neighbor_search_count, 0, 16))
				{
					// Clamping
					restir_settings.common_temporal_pass.max_neighbor_search_count = std::max(0, restir_settings.common_temporal_pass.max_neighbor_search_count);

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("Temporal neighbor search radius", &restir_settings.common_temporal_pass.neighbor_search_radius, 0, 16))
				{
					// Clamping
					restir_settings.common_temporal_pass.neighbor_search_radius = std::max(0, restir_settings.common_temporal_pass.neighbor_search_radius);

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::Checkbox("Use Permutation Sampling", &restir_settings.common_temporal_pass.use_permutation_sampling))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("If true, the back-projected position of the current pixel (temporal neighbor position) will be shuffled"
					" to add temporal variations.");

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				if (ImGui::SliderInt("M-cap", &restir_settings.m_cap, 0, 48))
				{
					restir_settings.m_cap = std::max(0, restir_settings.m_cap);
					if (render_settings.accumulate)
						m_render_window->set_render_dirty(true);
				}
			}

			ImGui::TreePop();
			ImGui::PopID();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}
	}
}

template <bool IsReSTIRGI>
void ImGuiSettingsWindow::draw_ReSTIR_spatial_reuse_panel(std::function<void(void)> draw_before_panel)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	ReSTIRCommonSpatialPassSettings& restir_settings = IsReSTIRGI ? render_settings.restir_gi_settings.common_spatial_pass : render_settings.restir_di_settings.common_spatial_pass;
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Spatial Reuse Pass"))
	{
		ImGui::PushID(&restir_settings);
		ImGui::TreePush("ReSTIR - Spatial Reuse Pass Tree");
		{
			draw_before_panel();

			if (restir_settings.do_spatial_reuse_pass)
			{
				static bool use_spatial_target_function_visibility = IsReSTIRGI ? ReSTIR_GI_SpatialTargetFunctionVisibility : ReSTIR_DI_SpatialTargetFunctionVisibility;
				if (ImGui::Checkbox("Use visibility in target function", &use_spatial_target_function_visibility))
				{
					global_kernel_options->set_macro_value(IsReSTIRGI ? GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY : GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY, use_spatial_target_function_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Whether or not to use the visibility term in the target function used for "
					"resampling spatial neighbors.");

				int max_neighbor_count = restir_settings.reuse_neighbor_count;
				if (restir_settings.do_disocclusion_reuse_boost)
					max_neighbor_count = std::max(max_neighbor_count, restir_settings.disocclusion_reuse_count);
				static int partial_visibility_neighbor_count = max_neighbor_count;
				if (use_spatial_target_function_visibility)
				{
					ImGui::TreePush("VisibilitySpatialReuseLastPassOnly Tree");

					{
						if (ImGui::SliderInt("Partial Neighbor Visibility", &partial_visibility_neighbor_count, 0, max_neighbor_count, "%d", ImGuiSliderFlags_AlwaysClamp))
						{
							// Using -1 so that the user manipulates intuitive numbers between 0 and
							// 'restir_settings.reuse_neighbor_count'
							// but the shader actually wants value between -1 and
							// 'restir_settings.reuse_neighbor_count' for it to be meaningful
							restir_settings.neighbor_visibility_count = partial_visibility_neighbor_count;

							m_render_window->set_render_dirty(true);
						}
						ImGuiRenderer::show_help_marker("How many neighbors will actually use a visibility term, can be useful to balance "
							"performance/variance but lowering this value below the maximum amount of neighbors may actually reduce "
							"performance because the final shading pass will have more visibility tests to do: if all neighbors use "
							"visibility during spatial resampling, then the final shading pass can be certain that all neighbors "
							"already take occlusion into account and so the final shading pass doesn't compute visibility. "
							"However, if 1 or 2 neighbors do not include visibility for example, then the final shading pass will "
							"have to trace rays for these neighbors and this will slow down the final shading pass quite a bit.");

						if (ImGui::Checkbox("Only on the last pass", &restir_settings.do_visibility_only_last_pass))
							m_render_window->set_render_dirty(true);
						ImGuiRenderer::show_help_marker("If checked, the visibility in the resampling target function will only be used on the last spatial reuse pass");
					}
					ImGui::Dummy(ImVec2(0.0f, 20.0f));

					ImGui::TreePop();
				}


				if (ImGui::SliderInt("Spatial Reuse Pass Count", &restir_settings.number_of_passes, 1, 8))
				{
					// Clamping
					restir_settings.number_of_passes = std::max(1, restir_settings.number_of_passes);

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("Spatial Reuse Radius (px)", &restir_settings.reuse_radius, 1, 64))
				{
					if (!restir_settings.debug_neighbor_location)
						// Clamping if not debugging (we do allow negative values when debugging)
						restir_settings.reuse_radius = std::max(1, restir_settings.reuse_radius);

					m_render_window->set_render_dirty(true);
				}

				// Checking the value before the "Neighbor Reuse Count" slider is modified
				// so that we know whether or not we'll have to keep the
				// 'partial_visibility_neighbor_count' value updated for the "Partial Neighbor Visibility" slider
				bool will_need_to_update_partial_visibility = partial_visibility_neighbor_count == max_neighbor_count;
				if (ImGui::SliderInt("Neighbor Reuse Count", &restir_settings.reuse_neighbor_count, 0, 16))
				{
					// Updating the maximum
					max_neighbor_count = restir_settings.reuse_neighbor_count;
					if (restir_settings.do_disocclusion_reuse_boost)
						max_neighbor_count = std::max(max_neighbor_count, restir_settings.disocclusion_reuse_count);

					bool reuse_count_is_the_max = max_neighbor_count == restir_settings.reuse_neighbor_count;
					reuse_count_is_the_max |= !restir_settings.do_disocclusion_reuse_boost;
					if (will_need_to_update_partial_visibility && reuse_count_is_the_max)
					{
						// Also updating the partial visibility neighbor index slider if it was set to the maximum
						// amount of neighbors
						partial_visibility_neighbor_count = restir_settings.reuse_neighbor_count;
						restir_settings.neighbor_visibility_count = partial_visibility_neighbor_count;
					}

					if (restir_settings.disocclusion_reuse_count < restir_settings.reuse_neighbor_count)
						// If disocclusion boost is now below the spatial neighbor count, bumping it up
						// because it makes no sense to have the disocclusion boost below the base
						// spatial neighbor count
						restir_settings.disocclusion_reuse_count = restir_settings.reuse_neighbor_count;

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::Checkbox("Increase Disocclusion Reuse Count", &restir_settings.do_disocclusion_reuse_boost))
				{
					m_render_window->set_render_dirty(true);
					if (restir_settings.do_disocclusion_reuse_boost)
					{
						// We just enabled disocclusion boost

						// Recomputing the max neighbor with the disocclusion boost taken into account
						max_neighbor_count = std::max(max_neighbor_count, restir_settings.disocclusion_reuse_count);

						partial_visibility_neighbor_count = max_neighbor_count;
					}
					else
						// Disabled disocclusion boost, bringing the value back to its maximum before
						// disocclusion boost which is just the number of reused spatial neighbors
						partial_visibility_neighbor_count = restir_settings.reuse_neighbor_count;

					restir_settings.neighbor_visibility_count = partial_visibility_neighbor_count;
				}
				ImGuiRenderer::show_help_marker("If checked, the given number of neighbors will be reused for pixels that just got "
					"disoccluded due to camera movement (and thus that have no temporal history). This helps "
					"reduce noise in disoccluded regions.");
				if (restir_settings.do_disocclusion_reuse_boost)
				{
					{
						ImGui::TreePush("Disocclusion boost tree");

						if (ImGui::SliderInt("Disoccluded Neighbor Reuse Count", &restir_settings.disocclusion_reuse_count, restir_settings.reuse_neighbor_count, 16 + restir_settings.reuse_neighbor_count))
						{
							m_render_window->set_render_dirty(true);

							// Updating the maximum
							max_neighbor_count = restir_settings.reuse_neighbor_count;
							if (restir_settings.do_disocclusion_reuse_boost)
								max_neighbor_count = std::max(max_neighbor_count, restir_settings.disocclusion_reuse_count);

							if (will_need_to_update_partial_visibility)
							{
								// If the number of neighbors using visibility is set at the maximum, then we should
								// keep that value at the maximum as we modify the disoccluded neighbor reuse count
								max_neighbor_count = restir_settings.disocclusion_reuse_count;
								partial_visibility_neighbor_count = max_neighbor_count;
								restir_settings.neighbor_visibility_count = max_neighbor_count;
							}
						}
						ImGuiRenderer::show_help_marker("How many neighbors a pixel will reuse if that pixel just got disoccluded.");

						if (restir_settings.neighbor_visibility_count == restir_settings.reuse_neighbor_count)
							// If the user is using the visibility in the target function of all spatial neighbors,
							// modifying that maximum number should still keep the visibility target function count
							// to the maximum
							restir_settings.neighbor_visibility_count = std::max(restir_settings.disocclusion_reuse_count, restir_settings.reuse_neighbor_count);

						ImGui::TreePop();
					}
				}

				if (ImGui::Checkbox("Neighbor Samples Random Rotation", &restir_settings.do_neighbor_rotation))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("If checked, spatial neighbors sampled (using the Hammersley point set) "
					"will be randomly rotated. Because neighbor locations are generated with a Hammersley point set "
					"(deterministic), not rotating them results in every pixel of every rendered image reusing the "
					"same neighbor locations which decreases reuse efficiency.");

				ImGui::BeginDisabled(!render_settings.enable_adaptive_sampling);
				if (ImGui::Checkbox("Allow Reuse of Converged Neighbors", &restir_settings.allow_converged_neighbors_reuse))
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
				if (restir_settings.allow_converged_neighbors_reuse)
				{
					if (ImGui::SliderFloat("Converged Neighbor Reuse Probability", &restir_settings.converged_neighbor_reuse_probability, 0.0f, 1.0f))
						m_render_window->set_render_dirty(true);
					ImGuiRenderer::show_help_marker("Allows trading bias for rendering performance by "
						"spatially reusing converged neighbors only with a certain probability instead of never / always."
						"\n\n 0.0 nevers reuses converged neighbors. No bias but performance impact."
						"\n\n 1.0 always reuses converged neighbors. Biased but no performance impact.");
				}
				ImGui::EndDisabled();

				if (ImGui::Checkbox("Debug Neighbor Reuse Positions", &restir_settings.debug_neighbor_location))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("If checked, neighbor in the spatial reuse pass will be hardcoded to always be "
					"15 pixels to the right, not in a circle. This makes spotting bias easier when debugging.");
			}
		}

		ImGui::TreePop();
		ImGui::PopID();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

template <bool IsReSTIRGI>
void ImGuiSettingsWindow::draw_ReSTIR_bias_correction_panel()
{
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
	ReSTIRCommonSettings* restir_settings;
	if constexpr (IsReSTIRGI)
		restir_settings = &m_renderer->get_render_settings().restir_gi_settings;
	else
		restir_settings = &m_renderer->get_render_settings().restir_di_settings;

	if (ImGui::CollapsingHeader("Bias correction"))
	{
		ImGui::PushID(restir_settings);
		ImGui::TreePush("Bias correction tree ReSTIR");

		{
			const char* bias_correction_mode_items[] = {
				"- 1/M Weights (Biased)",
				"- 1/Z Weights (Unbiased)",
				"- MIS-like Weights (Unbiased)",
				"- MIS Weights GBH (Unbiased)",
				"- Pairwise MIS Weights (Unbiased)",
				"- Pairwise MIS Weights Defensive (Unbiased)",
			};

			int* bias_correction_weights_option_pointer = global_kernel_options->get_raw_pointer_to_macro_value(IsReSTIRGI ? GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_WEIGHTS : GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_WEIGHTS);
			if (ImGui::Combo("Bias Correction Weights", bias_correction_weights_option_pointer, bias_correction_mode_items, IM_ARRAYSIZE(bias_correction_mode_items)))
			{
				m_renderer->recompile_kernels();

				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("What weights to use to resample reservoirs");

			bool disable_confidence_weights = *bias_correction_weights_option_pointer == (IsReSTIRGI ? RESTIR_GI_BIAS_CORRECTION_1_OVER_M : RESTIR_DI_BIAS_CORRECTION_1_OVER_M)
										   || *bias_correction_weights_option_pointer == (IsReSTIRGI ? RESTIR_GI_BIAS_CORRECTION_1_OVER_Z : RESTIR_DI_BIAS_CORRECTION_1_OVER_Z);

			ImGui::BeginDisabled(disable_confidence_weights);
			if (ImGui::Checkbox("Use Confidence Weights", &restir_settings->use_confidence_weights))
				m_render_window->set_render_dirty(true);
			std::string confidence_weight_help_string = "Whether or not to use confidence weights when resampling the samples. Confidence weights allow proper temporal reuse.";
			if (disable_confidence_weights)
				confidence_weight_help_string += "\n\nDisabled because 1/M or 1/Z weights use confidence weights by design.";
			ImGuiRenderer::show_help_marker(confidence_weight_help_string);
			ImGui::EndDisabled();

			// No visibility bias correction for 1/M weights
			bool bias_correction_visibility_disabled = *bias_correction_weights_option_pointer == (IsReSTIRGI ? RESTIR_GI_BIAS_CORRECTION_1_OVER_M : RESTIR_DI_BIAS_CORRECTION_1_OVER_M);
			static bool bias_correction_use_visibility = IsReSTIRGI ? ReSTIR_GI_BiasCorrectionUseVisibility : ReSTIR_DI_BiasCorrectionUseVisibility;
			ImGui::BeginDisabled(bias_correction_visibility_disabled);
			if (ImGui::Checkbox("Use visibility in bias correction", &bias_correction_use_visibility))
			{
				int* bias_correction_use_visibility_option_pointer = global_kernel_options->get_raw_pointer_to_macro_value(IsReSTIRGI ? GPUKernelCompilerOptions::RESTIR_GI_BIAS_CORRECTION_USE_VISIBILITY : GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY);
				*bias_correction_use_visibility_option_pointer = bias_correction_use_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE;

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			if (bias_correction_visibility_disabled)
				ImGuiRenderer::show_help_marker("Visibility bias correction cannot be used with 1/M weights.");
			ImGui::EndDisabled();
		}

		ImGui::TreePop();
		ImGui::PopID();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

void ImGuiSettingsWindow::draw_next_event_estimation_plus_plus_panel()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Next Event Estimation++"))
	{
		ImGui::TreePush("Use NEE++ Tree");

		static bool use_nee_plus_plus = DirectLightUseNEEPlusPlus;
		if (ImGui::Checkbox("Use NEE++", &use_nee_plus_plus))
		{
			kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS, use_nee_plus_plus ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Whether or not to use NEE++ [Guo et al., 2020] features at all.");

		if (use_nee_plus_plus)
		{
			ImGui::TreePush("NEE++ Settings Tree");

			static unsigned int max_cell_records = 0;

			{
				ImGui::Text("Visiblity map update in: %.3fs", m_renderer->get_nee_plus_plus_data().milliseconds_before_finalizing_accumulation / 1000.0f);
				ImGui::SameLine();
				if (ImGui::Button("Refresh##vis_map"))
					m_renderer->get_nee_plus_plus_data().milliseconds_before_finalizing_accumulation = 0.0f;

				unsigned int traced = 0;
				unsigned int total = 0;

				ImGui::Text("Shadow rays traced: %.3f%%", m_renderer->get_nee_plus_plus_data().shadow_rays_actually_traced_cpu/ (float)m_renderer->get_nee_plus_plus_data().total_shadow_ray_queries_cpu * 100.0f);
				ImGui::SameLine();
				std::string button_text = render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics ? "Stop" : "Resume";
				if (ImGui::Button(button_text.c_str()))
					render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics = !render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics;

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
			}


			{

				if (ImGui::Checkbox("Update visibility map", &render_data.nee_plus_plus.update_visibility_map))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("If checked, the visibility map will continue accumulating visibility "
					"information as the rendering progresses");
				ImGui::SliderInt("Update max samples", &m_renderer->get_nee_plus_plus_data().stop_update_samples, 1, 96);
				ImGuiRenderer::show_help_marker("After this many samples, the update of the visibility will automatically "
					"stop to save some performance because accumulating forever isn't necessary.");
				ImGui::Dummy(ImVec2(0.0f, 20.0f));

				static bool use_nee_plus_plus_rr = DirectLightUseNEEPlusPlusRR;
				if (ImGui::Checkbox("Use NEE++ Russian Roulette", &use_nee_plus_plus_rr))
				{
					kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE, use_nee_plus_plus_rr ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Implementation of NEE++, [Guo et al., 2020].\n"
					"If checked, the voxel-to-voxel visibility estimate of NEE++ will be used to "
					"stochastically determine whether or not attempt at all to trace a shadow at "
					"a light during next-event-estimation.");

				if (use_nee_plus_plus_rr)
				{
					ImGui::TreePush("NEE++ RR options tree");

					if (ImGui::Checkbox("Use NEE++ RR for emissives", &render_data.nee_plus_plus.enable_nee_plus_plus_RR_for_emissives))
						m_render_window->set_render_dirty(true);

					if (ImGui::Checkbox("Use NEE++ RR for envmap", &render_data.nee_plus_plus.enable_nee_plus_plus_RR_for_envmap))
						m_render_window->set_render_dirty(true);

					ImGui::TreePop();
				}

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
			}

			{
				static bool use_cube_grid = true;
				ImGui::Checkbox("Use cubic grid", &use_cube_grid);
				bool size_changed = false;
				if (use_cube_grid)
				{
					static int grid_size = m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap.x;
					if (ImGui::SliderInt("Grid size (X, Y & Z)", &grid_size, 2, 30))
					{
						m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap.x = grid_size;
						m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap.y = grid_size;
						m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap.z = grid_size;

						size_changed = true;
					}
				}
				else
				{
					ImGui::PushItemWidth(4 * ImGui::GetFontSize());
					size_changed |= ImGui::SliderInt("##Grid_sizeX", &m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap.x, 2, 30);
					ImGui::SameLine();
					size_changed |= ImGui::SliderInt("##Grid_sizeY", &m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap.y, 2, 30);
					ImGui::SameLine();
					size_changed |= ImGui::SliderInt("Grid size (X/Y/Z)", &m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap.z, 2, 30);

					// Back to default size
					ImGui::PushItemWidth(16 * ImGui::GetFontSize());
				}

				if (size_changed)
				{
					// Clamping
					m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap = hippt::clamp(make_int3(2, 2, 2), make_int3(30, 30, 30), m_renderer->get_nee_plus_plus_data().grid_dimensions_no_envmap);

					m_renderer->reset_nee_plus_plus();
					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderFloat("Confidence threshold", &render_data.nee_plus_plus.confidence_threshold, 0.0f, 1.0f))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("If a voxel-to-voxel unocclusion probability is higher than that, "
					"the voxel will be considered unoccluded and so a shadow ray will be traced. This is to "
					"avoid trusting voxel that have a low probability of being unoccluded\n\n"
					""
					"0.0f basically disables NEE++ as any entry of the visibility map will require a shadow ray.\n\n"
					""
					"Higher values yield higher performance but also higher variance (and the tradeoff doesn't seem "
					"worth it, hence the very low default value which means that we only allow ourselves "
					"to save shadow rays when we have a very high probability that the two voxels are occluded.");
				ImGui::Text("VRAM Usage: %.3fMB", m_renderer->get_nee_plus_plus_data().get_vram_usage_bytes() / 1000000.0f);

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
			}

			{
				static bool display_shadow_rays = DirectLightNEEPlusPlusDisplayShadowRaysDiscarded;
				if (ImGui::Checkbox("Display shadow rays discarded", &display_shadow_rays))
				{
					m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED, display_shadow_rays ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("With this debug view enabled, every black pixel is a pixel which discarded its "
					"shadow ray thanks to NEE++ russian roulette.\n"
					"A colored pixel didn't discard its shadow ray.");
				if (display_shadow_rays)
				{
					ImGui::TreePush("Display shadow rays tree");

					static int shadow_ray_bounce_to_display = DirectLightNEEPlusPlusDisplayShadowRaysDiscardedBounce;
					if (ImGui::SliderInt("Bounce to display", &shadow_ray_bounce_to_display, 0, m_renderer->get_render_settings().nb_bounces))
					{
						m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED_BOUNCE, shadow_ray_bounce_to_display);
						m_renderer->recompile_kernels();
					}

					ImGui::TreePop();
				}

				if (ImGui::Button("Clear visibility map"))
				{
					m_renderer->reset_nee_plus_plus();
					m_render_window->set_render_dirty(true);
				}
			}

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_principled_bsdf_energy_conservation()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	HIPRTRenderData& render_data = m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	ImGui::SeparatorText("BSDF Energy Conservation Settings");

	static bool do_energy_conservation = PrincipledBSDFDoEnergyCompensation;
	if (ImGui::Checkbox("Do energy conservation", &do_energy_conservation))
	{
		global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION, do_energy_conservation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
		m_renderer->recompile_kernels();
		m_render_window->set_render_dirty(true);
	}
	ImGuiRenderer::show_help_marker("Global toggle to completely enable/disable any forms "
		"of energy compensation in all the materials using the Principled BSDF");

	if (do_energy_conservation)
	{
		ImGui::TreePush("Energy conservation options tree");

		static bool do_bsdf_energy_conservation = PrincipledBSDFEnforceStrongEnergyConservation;
		if (ImGui::Checkbox("Enforce BSDF strong energy conservation", &do_bsdf_energy_conservation))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENFORCE_ENERGY_CONSERVATION, do_bsdf_energy_conservation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("If checked, this will enable the strong energy conservation & preservation "
			"of the BSDF such that materials using this option neither lose or gain any amount of energy.\n\n"

			"This is however very computationally expensive and must also be enabled on a per material basis.\n"
			"The per-material option can be found in the \"Other properties\" tab of the material editor.\n"
			"This is usually only needed on clearcoated materials (but even then, the energy loss\n"
			"due to the absence of multiple scattering between the clearcoat layer and the BSDF below "
			"may be acceptable).\n\n"

			"Non-clearcoated materials can already ensure perfect (modulo implementation quality) energy "
			"conservation/preservation with the precomputed LUTs [Turquin, 2019] "
			"\"Use GGX Multiple Scattering\" option in \"Sampling\" --> \"Materials\".\n\n"

			"Note that even if no materials use the option in your scene, disabling this option may"
			" still be benefitial for performance as it adds quite a bit of register pressure. Disabling "
			" the option has the effect of literally removing all the code of this option from the shaders.");
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		{
			static bool do_glass_energy_compensation = PrincipledBSDFDoGlassEnergyCompensation;
			if (ImGui::Checkbox("Do glass lobe energy compensation", &do_glass_energy_compensation))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION, do_glass_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Global toggle on whether or not objects in the scene that use "
				"the Principled BSDF should do energy compensation for the glass layer."
				""
				"Implementation of [Practical multiple scattering compensation for microfacet models, Turquin, 2019].");
		}

		{
			static bool do_clearcoat_energy_compensation = PrincipledBSDFDoClearcoatEnergyCompensation;
			if (ImGui::Checkbox("Do clearcoat lobe energy compensation", &do_clearcoat_energy_compensation))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION, do_clearcoat_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Global toggle on whether or not objects in the scene that use "
				"the Principled BSDF should do energy compensation for the clearcoat layer.\n\n"
				""
				"Energy compensation on the clearcoat layer is an approximation but works very well in common cases.");
		}

		{
			static bool do_specular_energy_compensation = PrincipledBSDFDoSpecularEnergyCompensation;
			if (ImGui::Checkbox("Do specular/diffuse lobe energy compensation", &do_specular_energy_compensation))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION, do_specular_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Global toggle on whether or not objects in the scene that use "
				"the Principled BSDF should do energy compensation for the glossy (specular/diffuse) layer."
				""
				"Implementation of [Practical multiple scattering compensation for microfacet models, Turquin, 2019].");
		}

		{
			static bool do_metallic_energy_compensation = PrincipledBSDFDoMetallicEnergyCompensation;
			if (ImGui::Checkbox("Do metallic lobe energy compensation", &do_metallic_energy_compensation))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION, do_metallic_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Global toggle on whether or not objects in the scene that use "
				"the Principled BSDF should do energy compensation for the metallic layer."
				""
				"Implementation of [Practical multiple scattering compensation for microfacet models, Turquin, 2019].");

			if (do_metallic_energy_compensation)
			{
				ImGui::TreePush("Fresnel multiscatter tree");

				static bool use_multiple_scattering_fresnel = PrincipledBSDFDoMetallicFresnelEnergyCompensation;
				if (ImGui::Checkbox("Do GGX Multiple scattering fresnel", &use_multiple_scattering_fresnel))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION, use_multiple_scattering_fresnel ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Implementation of [Practical multiple scattering compensation for microfacet models, Turquin, 2019]"
					" for GGX energy compensation. The multiple scattering fresnel term takes into account the Fresnel "
					"reflection/transmission effect when the rays bounce multiple times on the microsurface. This is responsible "
					"for the increase in saturation of the color of conductors due to multiple scattering in-between the microfacets.");

				ImGui::TreePop();
			}
		}

		bool setting_changed = false;
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Max bounces");
		ImGui::Text("");
		ImGuiRenderer::show_help_marker("After what bounce to stop doing energy conservation (depending on the type of material)\n\n"
			""
			"0 means that energy conservation will only be done on the first hit (of camera rays) for example.\n\n"
			""
			"For glass, a value of 4 is usually enough to avoid losing too much energy when looking straight at a rough glass object.\n\n"
			""
			"For metals, 0 (only on the first hit) is also probably good enough except in some specific cases where rays get trapped "
			"(on a Mitsuba knob for example) where 4+ bounces may be required for decent results.\n\n"
			""
			"For clearcoated and specular materials, 0 is enough for smooth clearcoat/specular layers. "
			"For high roughness clearcoats/specular layers, the situation is the same as for metals: "
			"0 should be good enough as long as there are not too many concentrated inter-reflections "
			"(in which case, a higher value, 4+, is going to be preferred).");
		setting_changed |= ImGui::SliderInt("Glass energy conservation max bounce", &render_data.bsdfs_data.glass_energy_compensation_max_bounce, 0, render_settings.nb_bounces);
		setting_changed |= ImGui::SliderInt("Clearcoat energy conservation max bounce", &render_data.bsdfs_data.clearcoat_energy_compensation_max_bounce, 0, render_settings.nb_bounces);
		setting_changed |= ImGui::SliderInt("Specular/diffuse energy conservation max bounce", &render_data.bsdfs_data.glossy_base_energy_compensation_max_bounce, 0, render_settings.nb_bounces);
		setting_changed |= ImGui::SliderInt("Metallic energy conservation max bounce", &render_data.bsdfs_data.metal_energy_compensation_max_bounce, 0, render_settings.nb_bounces);
		if (setting_changed)
			m_render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("Energy compensation roughness threshold");
		if (ImGui::SliderFloat("", &render_data.bsdfs_data.energy_compensation_roughness_threshold, 0.0f, 1.0f))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker("Below this roughness, energy compensation will not be applied.\n\n"
			""
			"Generally speaking, the darkening of the material due to missing energy compensation is barely visible below 0.15f roughness.\n\n"
			""
			"0.0f disables the threshold and energy compensation will always be applied.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::Checkbox("Use hardware texture interpolation", &render_data.bsdfs_data.use_hardware_tex_interpolation))
		{
			m_renderer->init_GGX_glass_Ess_texture(render_data.bsdfs_data.use_hardware_tex_interpolation ? hipFilterModeLinear : hipFilterModePoint);
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Using the hardware for texture interpolation is faster but less precise than doing manual interpolation in the shader.");

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
		|| (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE && render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass))
		&& kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_BIAS_CORRECTION_USE_VISIBILITY) == KERNEL_OPTION_FALSE)
	{
		std::string prefix;
		if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE)
			prefix = " - Initial ";
		else if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE && render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
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
		&& render_settings.restir_di_settings.common_spatial_pass.allow_converged_neighbors_reuse
		&& render_settings.restir_di_settings.common_spatial_pass.converged_neighbor_reuse_probability > 0.0f)
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

void ImGuiSettingsWindow::draw_denoiser_panel()
{
	if (!ImGui::CollapsingHeader("Denoiser"))
		return;

	ImGui::TreePush("Denoiser tree");

	if (ImGui::Checkbox("Enable denoiser", &m_application_settings->enable_denoising))
		m_render_window->get_display_view_system()->queue_display_view_change(m_application_settings->enable_denoising ? DisplayViewType::DENOISED_BLEND : DisplayViewType::DEFAULT);
	if (ImGui::Checkbox("Use OpenGL Interop AOV Buffers", &m_application_settings->denoiser_use_interop_buffers))
	{
		m_renderer->set_use_denoiser_AOVs_interop_buffers(m_application_settings->denoiser_use_interop_buffers);
		m_render_window->set_render_dirty(true);
	}
	ImGuiRenderer::show_help_marker("If checked, a little bit of path tracing performance will be gained (on AMD GPUs at least) at the expense of "
		"a good bit of performance if displaying \"- Denoiser - Normals\" or \" - Denoiser - Albedo\" in the viewport.\n\n"
		""
		"You want this option checked only if you're visualizing the denoiser normals or denoiser albedo basically.");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

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
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	DisplaySettings& display_settings = m_render_window->get_display_view_system()->get_display_settings();
	ImGui::Checkbox("Only denoise when rendering is done", &m_application_settings->denoise_when_rendering_done);
	ImGui::SliderInt("Denoiser sample skip", &m_application_settings->denoiser_sample_skip, 1, 128);
	ImGui::SliderFloat("Denoiser blend", &display_settings.denoiser_blend, 0.0f, 1.0f);
	ImGui::EndDisabled();

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::Text("Denoising time: %.3fms", m_application_settings->last_denoised_duration / 1000.0f);

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiSettingsWindow::draw_post_process_panel()
{
	if (!ImGui::CollapsingHeader("Post-processing"))
		return;
	ImGui::TreePush("Post-processing tree");

	HIPRTRenderData& render_data = m_renderer->get_render_data();

	if (ImGui::CollapsingHeader("Tone-mapping"))
	{
		ImGui::TreePush("Tonemapping post processing tree");

		DisplaySettings& display_settings = m_render_window->get_display_view_system()->get_display_settings();

		bool changed = false;
		changed |= ImGui::Checkbox("Do tonemapping", &display_settings.do_tonemapping);
		changed |= ImGui::SliderFloat("Gamma", &display_settings.tone_mapping_gamma, 1.0f, 2.4f);
		changed |= ImGui::SliderFloat("Exposure", &display_settings.tone_mapping_exposure, 0.0f, 3.0f);
		if (changed)
			m_render_window->set_force_viewport_refresh(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
	std::shared_ptr<GMoNRenderPass> gmon_render_pass = m_renderer->get_gmon_render_pass();
	GMoNGPUData& gmon_data = gmon_render_pass->get_gmon_data();

	if (!render_data.render_settings.accumulate)
	{
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("Warning: ");
		ImGuiRenderer::show_help_marker("GMoN cannot be used without enabling accumulation.", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
	}
	ImGui::BeginDisabled(!render_data.render_settings.accumulate);
	if (ImGui::CollapsingHeader("GMoN"))
	{
		ImGui::TreePush("GMoN tree post processing");

		if (ImGui::Checkbox("Use GMoN", &gmon_data.using_gmon))
			toggle_gmon();

		ImGuiRenderer::show_help_marker("Use GMoN for fireflies elimination.\n"
			"The algorithm computes the median of means of the pixels as an estimator "
			"that is more robust than the simple mean usually used to average samples.\n"
			"The algorithm is unbiased as long as enough samples are accumulated. If not "
			"enough samples are accumulated, the firefly elimination tends to be a bit too "
			"strong and the image will probably end up darker than expected, especially on high-variance scenes.\n\n"
			""
			"Implementation following [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]");

		if (gmon_data.using_gmon)
		{
			ImGui::Text("VRAM Usage: %.3fMB", gmon_render_pass->get_VRAM_usage_bytes() / 1000000.0f);

			bool gmon_mode_changed = false;
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::Text("GMoN Mode");
			gmon_mode_changed |= ImGui::RadioButton("Median of Means", ((int*)&render_data.buffers.gmon_estimator.gmon_mode), 0); ImGui::SameLine();
			gmon_mode_changed |= ImGui::RadioButton("Binary G-MoN", ((int*)&render_data.buffers.gmon_estimator.gmon_mode), 1); ImGui::SameLine();
			gmon_mode_changed |= ImGui::RadioButton("Adaptive G-MoN", ((int*)&render_data.buffers.gmon_estimator.gmon_mode), 2);
			if (gmon_mode_changed)
				m_render_window->set_render_dirty(true);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			static int number_of_sets = GMoNMSetsCount;
			if (ImGui::SliderInt("Number of sets (M)", &number_of_sets, 3, 31))
			{
				number_of_sets = hippt::clamp(3, 31, number_of_sets);

				if (!(number_of_sets & 1))
					// number_of_sets is even but we only want odd
					number_of_sets--;
			}
			ImGuiRenderer::show_help_marker("How many sets (M variable in the GMoN paper, [Buisine et al., 2021]).\n\n"
				""
				"As a general rule: more sets eliminate fireflies the best but more sets require more samples per "
				"pixel to avoid too much darkening, especially on high-variance scene. If your scene is very "
				"easy to render, you probably don't need many sets (less than 15). If your scene has high "
				"variance caustics, you're probably going to need a lot of samples per pixel and so a large "
				"number of sets will be fine anyways.\n\n"
				"Said otherwise: if you're noticing too much darkening, try reducing the number of sets or "
				"try accumulating more samples per pixel.\n\n"
				""
				"Yet another way top choose the number of sets: keep that number as low as possible as long as it removes the fireflies.");
			// If the user modified the number of sets, displaying an "Apply" button
			if (number_of_sets != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT))
			{
				ImGui::SameLine();
				if (ImGui::Button("Apply"))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT, number_of_sets);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
			}

			if (ImGui::SliderFloat("GMoN blend factor", &gmon_data.gmon_blend_factor, 0.0f, 1.0f))
			{
				gmon_data.gmon_auto_blend_factor = false;
				m_render_window->set_force_viewport_refresh(true);
			}
			ImGui::SameLine();
			ImGui::Checkbox("Auto", &gmon_data.gmon_auto_blend_factor);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (m_render_window->get_display_view_system()->get_current_display_view_type() != DisplayViewType::GMON_BLEND)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("The display view currently in used isn't \"GMoN blend\" so the output of GMoN cannot be visualized.", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
		}

		ImGui::TreePop();
	}
	ImGui::EndDisabled();

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::TreePop();
}

void ImGuiSettingsWindow::toggle_gmon()
{
	std::shared_ptr<GMoNRenderPass> gmon_render_pass = m_renderer->get_gmon_render_pass();
	bool gmon_now_enabled = gmon_render_pass->get_gmon_data().using_gmon;
	if (m_render_window->get_display_view_system()->get_current_display_view_type() == DisplayViewType::DEFAULT && gmon_now_enabled)
		// We just enabled GMoN, automatically switching to the GMoN view for convenience
		m_render_window->get_display_view_system()->queue_display_view_change(DisplayViewType::GMON_BLEND);

	if (gmon_now_enabled && !gmon_render_pass->get_all_kernels()[GMoNRenderPass::COMPUTE_GMON_KERNEL]->has_been_compiled())
		// The GMoN kernel hasn't been compiled yet, compiling it
		m_renderer->recompile_kernels();

	m_render_window->set_render_dirty(true);
}

void ImGuiSettingsWindow::draw_quality_panel()
{
	if (!ImGui::CollapsingHeader("Quality settings"))
		return;

	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	ImGui::TreePush("Quality settings tree");

	if (ImGui::CollapsingHeader("Nested dielectrics"))
	{
		ImGui::TreePush("Nested dielectrics tree");

		std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

		static int nested_dielectrics_stack_size = NestedDielectricsStackSize;
		if (ImGui::SliderInt("Stack Size", &nested_dielectrics_stack_size, 3, 8))
			nested_dielectrics_stack_size = std::max(1, nested_dielectrics_stack_size);
		ImGui::Text("Max nested dielectrics: %d", nested_dielectrics_stack_size - 3);
		ImGuiRenderer::show_help_marker("How many nested dielectrics objects can be present in the scene with the "
			"current nested dielectrics stack size");

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
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		if (ImGui::SliderInt("Max bounce", &render_settings.alpha_testing_indirect_bounce, 0, render_settings.nb_bounces))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker("At what bounce to stop doing alpha testing.\n\n"
			""
			"A value of 0 means that alpha testing isn't done at bounce 0 which means that even camera "
			"rays do not do alpha testing --> alpha testing is disabled.\n\n"
			""
			"A value of 1 means that camera rays do alpha testing but the next bounce rays do not do alpha "
			"testing.\n\n"
			""
			"Shadow rays for NEE are also affected by this setting.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f)); 
			ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Normal mapping"))
	{
		ImGui::TreePush("Normal mapping tree");

		if (ImGui::Checkbox("Do normal mapping", &render_settings.do_normal_mapping))
			m_render_window->set_render_dirty(true);

		ImGui::TreePop();
	}

	ImGui::TreePop();
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

	if (ImGui::CollapsingHeader("General Settings"))
	{
		ImGui::TreePush("Perf settings general settings tree");

		if (ImGui::InputFloat("GPU Stall Percentage", &m_application_settings->GPU_stall_percentage))
			m_application_settings->GPU_stall_percentage = std::max(0.0f, std::min(m_application_settings->GPU_stall_percentage, 99.9f));
		ImGuiRenderer::show_help_marker("How much percent of the time the GPU will be forced to be idle (not rendering anything)."
										" This feature is basically only meant for GPUs that get too hot to avoid burning your GPUs during long renders if you have"
										" time to spare.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		draw_russian_roulette_options();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		static bool reuse_bsdf_mis_ray = ReuseBSDFMISRay;
		if (ImGui::Checkbox("Reuse MIS BSDF ray", &reuse_bsdf_mis_ray))
		{
			kernel_options->set_macro_value(GPUKernelCompilerOptions::REUSE_BSDF_MIS_RAY, reuse_bsdf_mis_ray ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("If checked, the BSDF ray shot for BSDF MIS during the evaluation of NEE will be reused "
			"for the next bounce.\n\n"
			""
			"There is virtually no point in disabling that option. This options i there only for "
			"performance comparisons with and without reuse");
		static bool do_direction_reuse = DoFirstBounceWarpDirectionReuse;
		if (ImGui::Checkbox("Warp BSDF sampled directions reuse", &do_direction_reuse))
		{
			kernel_options->set_macro_value(GPUKernelCompilerOptions::DO_FIRST_BOUNCE_WARP_DIRECTION_REUSE, do_direction_reuse ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
			m_renderer->recompile_kernels();

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Partial and experimental implementation of[Generate Coherent Rays Directly, Liu et al., 2024] "
			"for reuse sampled directions on the first hit accross the threads of warps");

		static bool delta_distrib_opti = PrincipledBSDFDeltaDistributionEvaluationOptimization;
		if (ImGui::Checkbox("BSDF delta distribution optimization", &delta_distrib_opti))
		{
			kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION, delta_distrib_opti ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
			m_renderer->recompile_kernels();

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("If this is true, then delta distribution lobes of the principled BSDF will not be evaluated "
			"if the incident light direction used for the evaluation doesn't come from sampling the "
			" delta distribution lobe itself.\n\n"
			""
			"For example, consider a clearcoat diffuse lobe. If bsdf_eval() is called with an "
			"incident light direction that was sampled from the diffuse lobe, the perfectly smooth clearcoat lobe "
			"is going to have its contribution evaluate to 0 because there is no chance that the sampled "
			"diffuse direction perfectly aligns with the delta of the smooth clearcoat lobe.\n\n"
			""
			"Same with all the other lobes that can be delta distributions.\n\n"
			""
			"There is basically no point in disabling that, this is just for performance comparisons.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Lighting Settings"))
	{
		ImGui::TreePush("Lighting Settings Performance Tree");

		ImGui::SeparatorText("Clamping");
		if (ImGui::SliderFloat("Direct lighting", &render_settings.direct_contribution_clamp, 0.0f, 10.0f))
		{
			render_settings.direct_contribution_clamp = std::max(0.0f, render_settings.direct_contribution_clamp);
			m_render_window->set_render_dirty(true);
		}
		if (ImGui::SliderFloat("Envmap ligthing", &render_settings.envmap_contribution_clamp, 0.0f, 10.0f))
		{
			render_settings.envmap_contribution_clamp = std::max(0.0f, render_settings.envmap_contribution_clamp);
			m_render_window->set_render_dirty(true);
		}
		if (ImGui::SliderFloat("Indirect ligthing", &render_settings.indirect_contribution_clamp, 0.0f, 10.0f))
		{
			render_settings.indirect_contribution_clamp = std::max(0.0f, render_settings.indirect_contribution_clamp);
			m_render_window->set_render_dirty(true);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
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
		draw_next_event_estimation_plus_plus_panel();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Ray-tracing settings"))
	{
		ImGui::TreePush("Ray-tracing settings tree");

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

		bool bvh_needs_rebuild = false;
		static int build_type_chosen = 0;
		std::vector<const char*> bvh_items = { "- SBVH", "- HPLOC", "- LBVH"};
		bvh_needs_rebuild |= ImGui::Combo("BVH Build", &build_type_chosen, bvh_items.data(), bvh_items.size());

		static bool do_triangle_splits = true;
		bvh_needs_rebuild |= ImGui::Checkbox("Do triangle splits", &do_triangle_splits);

		/*static bool do_triangle_pairing = true;
		bvh_needs_rebuild |= ImGui::Checkbox("Do triangle pairing", &do_triangle_pairing);*/

		static bool do_bvh_compaction = true;
		bvh_needs_rebuild |= ImGui::Checkbox("Do BVH compaction", &do_bvh_compaction);

		if (bvh_needs_rebuild)
		{
			hiprtBuildFlags build_flags = 0;
			switch (build_type_chosen)
			{
			case 0:
				// SBVH
				build_flags |= hiprtBuildFlagBitPreferHighQualityBuild;
				break;

			case 1:
				// HPLOC
				build_flags |= hiprtBuildFlagBitPreferBalancedBuild;
				break;

			case 2:
				// LBVH
				build_flags |= hiprtBuildFlagBitPreferFastBuild;
				break;
			}

			/*if (!do_triangle_pairing)
				build_flags |= hiprtBuildFlagBitDisableTrianglePairing;*/

			if (!do_triangle_splits)
				build_flags |= hiprtBuildFlagBitDisableSpatialSplits;

			m_renderer->rebuild_renderer_bvh(build_flags, do_bvh_compaction);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Kernel Settings"))
	{
		ImGui::TreePush("Shared/global stack Traversal Options Tree");

		// List of exceptions because these kernels do not trace any rays
		static std::vector<std::string> kernel_names;
		static std::map<std::string, std::shared_ptr<GPUKernel>> kernels = m_renderer->get_tracing_kernels();
		if (kernel_names.empty())
			// Filling the kernel names if not already done
			for (const auto& name_to_kernel : kernels)
				kernel_names.push_back(name_to_kernel.first);

		static std::string selected_kernel_name = FillGBufferRenderPass::FILL_GBUFFER_KERNEL;
		static std::shared_ptr<GPUKernel> selected_kernel = kernels[selected_kernel_name];
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

				ImGui::PushItemWidth(8 * ImGui::GetFontSize());
				if (ImGui::InputInt("Shared stack size", &pending_stack_size))
					pending_stack_size = std::max(0, pending_stack_size);
				ImGui::PopItemWidth();

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
		ImGui::PushItemWidth(8 * ImGui::GetFontSize());
		if (ImGui::InputInt("Global stack per-thread size", &m_renderer->get_render_data().global_traversal_stack_buffer_size))
		{
			m_renderer->get_render_data().global_traversal_stack_buffer_size = hippt::clamp(0, 128, m_renderer->get_render_data().global_traversal_stack_buffer_size);
			m_render_window->set_render_dirty(true);
		}
		ImGui::PopItemWidth();

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


	ImGui::Dummy(ImVec2(0.0f, 20.0f));
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

		m_render_window->set_render_dirty(true);
	}
	if (ImGui::Checkbox("Freeze random", (bool*)&render_settings.freeze_random))
		m_render_window->set_render_dirty(true);
	if (ImGui::InputInt("Samples per frame", &render_settings.samples_per_frame))
		// Clamping to 1
		render_settings.samples_per_frame = std::max(1, render_settings.samples_per_frame);
	ImGui::SameLine();
	ImGui::Checkbox("Auto", &m_application_settings->auto_sample_per_frame);

	bool rolling_window_size_changed = false;
	int rolling_window_size = m_render_window_perf_metrics->get_window_size();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::Text("Measures Window Size"); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("25", &rolling_window_size, 25); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("100", &rolling_window_size, 100); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("250", &rolling_window_size, 250); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("1000", &rolling_window_size, 1000);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	if (rolling_window_size_changed)
		m_render_window_perf_metrics->resize_window(rolling_window_size);

	RenderGraph& render_graph = m_renderer->get_render_graph();
	for (auto& name_to_render_pass : render_graph.get_render_passes())
	{
		const std::map<std::string, std::shared_ptr<GPUKernel>>& render_pass_kernels = name_to_render_pass.second->get_all_kernels();
		if (!render_pass_kernels.empty())
		{
			ImGui::SeparatorText(name_to_render_pass.first.c_str());

			ImGui::TreePush(name_to_render_pass.first.c_str());
			for (auto& name_to_kernel : render_pass_kernels)
				draw_perf_metric_specific_panel(m_render_window_perf_metrics, name_to_kernel.first, name_to_kernel.first);
			ImGui::TreePop();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

	}

	draw_perf_metric_specific_panel(m_render_window_perf_metrics, RenderWindow::PERF_METRICS_CPU_OVERHEAD_TIME_KEY, "CPU Overhead");
	ImGui::Separator();
	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::ALL_RENDER_PASSES_TIME_KEY, "Total sample time (GPU)");
	draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::FULL_FRAME_TIME_WITH_CPU_KEY, "Total sample time (+CPU)");
	if (m_debug_trace_kernel_selected != 0)
	{
		ImGui::Separator();
		draw_perf_metric_specific_panel(m_render_window_perf_metrics, GPURenderer::DEBUG_KERNEL_TIME_KEY, "Debug trace kernel");
	}

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

extern bool g_background_shader_compilation_enabled;
void ImGuiSettingsWindow::draw_shader_kernels_panel()
{
	if (ImGui::CollapsingHeader("Shaders/Kernels"))
	{
		ImGui::TreePush("Shaders kernels tree");
		std::string background_shader_compilation_button_string;
		if (g_background_shader_compilation_enabled)
			background_shader_compilation_button_string = "Stop background shader compilation";
		else
			background_shader_compilation_button_string = "Resume background shader compilation";

		if (ImGui::Button(background_shader_compilation_button_string.c_str()))
		{
			if (g_background_shader_compilation_enabled)
				m_renderer->stop_background_shader_compilation();
			else
				m_renderer->resume_background_shader_compilation();
		}
		ImGuiRenderer::show_help_marker("Click to " + (g_background_shader_compilation_enabled ? std::string("stop") : std::string("resume")) + " background shaders precompilation");

		if (ImGui::Button("Force shaders reload"))
		{
			m_renderer->recompile_kernels(false);
			m_render_window->set_render_dirty(true);
		}
		if (ImGui::Button("Clear shader cache"))
			std::filesystem::remove_all("shader_cache");
		ImGuiRenderer::show_help_marker("Completely clears the shader cache on the disk.");

		static GPUKernelCompiler::ShaderCacheUsageOverride shader_cache_use_override = g_gpu_kernel_compiler.get_shader_cache_usage_override();
		std::vector<const char*> shader_cache_override_values = { "No override", "Do not use shader cache", "Always use shader cache" };
		if (ImGui::Combo("Shader cache use override", (int*)&shader_cache_use_override, shader_cache_override_values.data(), shader_cache_override_values.size()))
			g_gpu_kernel_compiler.set_shader_cache_usage_override(shader_cache_use_override);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::CollapsingHeader("Kernels compilation statistics"))
		{
			ImGui::Text("Kernel [Registers, Shared Memory, Local Memory]");
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			// Computing the longest kernel name for aligning everything
			size_t longest_kernel_name = 0;
			for (auto kernel_name_to_kernel : m_renderer->get_all_kernels())
				longest_kernel_name = hippt::max(longest_kernel_name, kernel_name_to_kernel.first.length());
			std::string padding_formatter = "%-" + std::to_string(longest_kernel_name) + "s";

			for (auto kernel_name_to_kernel : m_renderer->get_all_kernels())
			{
				const std::string& kernel_name = kernel_name_to_kernel.first;
				const std::shared_ptr<GPUKernel> kernel = kernel_name_to_kernel.second;

				if (kernel->has_been_compiled())
				{
					int nb_reg = kernel->get_kernel_attribute(ORO_FUNC_ATTRIBUTE_NUM_REGS);
					int nb_shared = kernel->get_kernel_attribute(ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
					int nb_local = kernel->get_kernel_attribute(ORO_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);

					std::string text = padding_formatter + " [%d, %d, %d]";
					ImGui::Text(text.c_str(), kernel_name.c_str(), nb_reg, nb_shared, nb_local);
				}
				else
				{
					std::string text = padding_formatter + " [Not compiled]";
					ImGui::Text(text.c_str(), kernel_name.c_str());
				}
			}
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_debug_panel()
{
	if (!ImGui::CollapsingHeader("Debug"))
		return;

	ImGui::TreePush("Debug tree");

	if (ImGui::Checkbox("Show NaNs", &m_renderer->get_render_settings().display_NaNs))
		m_render_window->set_render_dirty(true);
	ImGuiRenderer::show_help_marker("If true, NaNs that occur during the rendering will show up as pink pixels.");

	if (ImGui::Checkbox("White furnace mode", &m_renderer->get_render_data().bsdfs_data.white_furnace_mode))
		m_render_window->set_render_dirty(true);
	if (m_renderer->get_render_data().bsdfs_data.white_furnace_mode)
	{
		ImGui::TreePush("White furnace tree");
		if (ImGui::Checkbox("Turn off emissives", &m_renderer->get_render_data().bsdfs_data.white_furnace_mode_turn_off_emissives))
			m_render_window->set_render_dirty(true);
		ImGui::TreePop();
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	std::vector<const char*> trace_kernel_items = { "None", "TraceTest" };
	if (ImGui::Combo("Override trace kernel", &m_debug_trace_kernel_selected, trace_kernel_items.data(), trace_kernel_items.size()))
	{
		if (m_debug_trace_kernel_selected != 0)
		{
			m_debug_trace_kernel_options = *m_renderer->get_global_compiler_options().get();
			m_debug_trace_kernel_options.set_macro_value("__USE_HWI__", 1);

			m_renderer->set_debug_trace_kernel(trace_kernel_items[m_debug_trace_kernel_selected], m_debug_trace_kernel_options);
			m_render_window->set_render_dirty(true);
		}
		else
		{
			// Disabling the debug trace kernel
			m_renderer->set_debug_trace_kernel("");
			m_render_window->set_render_dirty(true);
		}
	}

	ImGui::TreePush("DebugTraceKernelOptions");
	switch (m_debug_trace_kernel_selected)
	{
	case 1:
		ImGui::InputInt("BVH Traversal Shared Mem", m_debug_trace_kernel_options.get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE));
		if (ImGui::Button("Apply"))
			m_renderer->set_debug_trace_kernel(trace_kernel_items[m_debug_trace_kernel_selected], m_debug_trace_kernel_options);

		break;

	default:
		break;
	}
	ImGui::TreePop();

	ImGui::TreePop();
}
