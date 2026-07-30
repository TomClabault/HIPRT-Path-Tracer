/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Compiler/GPUKernelCompiler.h"
#include "Device/includes/BSDFs/MicrofacetRegularization.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRDIOptions.h"
#include "HostDeviceCommon/LightTreeSGSettings.h"
#include "HostDeviceCommon/RenderSettings.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/FillGBufferRenderPass.h"
#include "Renderer/RenderPasses/MegaKernelRenderPass.h"
#include "Scene/CameraAnimation.h"
#include "Threads/ThreadManager.h"
#include "UI/ImGui/ImGuiRenderer.h"
#include "UI/ImGui/ImGuiSettingsWindow.h"
#include "UI/RenderWindow.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <iostream>
#include <sstream>

#include "Threads/ThreadFunctions.h"

#include "Scene/SceneParser.h"

extern GPUKernelCompiler g_gpu_kernel_compiler;

static const std::vector<std::string>& assimp_supported_extensions()
{
	static std::vector<std::string> extensions;
	if (extensions.empty())
	{
		Assimp::Importer importer;
		std::string ext_list;
		importer.GetExtensionList(ext_list);

		std::istringstream ss(ext_list);
		std::string token;
		while (std::getline(ss, token, ';'))
			if (token.size() >= 2 && token[0] == '*')
				extensions.push_back(token.substr(1));
	}
	return extensions;
}

const char* ImGuiSettingsWindow::TITLE	   = "Render settings";
const float ImGuiSettingsWindow::BASE_SIZE = 630.0f;

void ImGuiSettingsWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;

	m_application_settings		 = render_window->get_application_settings();
	m_renderer					 = render_window->get_renderer();
	m_render_window_denoiser	 = render_window->get_denoiser();
	m_render_window_perf_metrics = m_render_window->get_performance_metrics();
}

void ImGuiSettingsWindow::set_status_text(const std::string& new_status_text)
{
	m_status_text = new_status_text;
}

std::string ImGuiSettingsWindow::get_status_text() const
{
	return m_status_text;
}

void ImGuiSettingsWindow::draw()
{
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 0.0f));
	ImGui::Begin(ImGuiSettingsWindow::TITLE, nullptr, ImGuiWindowFlags_NoDecoration);

	draw_header();

	// Scene file chooser
	{
		std::string current_scene_filepath = m_renderer->get_scene_filepath();
		std::string current_scene_filename = current_scene_filepath.empty() ? "None" : std::filesystem::path(current_scene_filepath).filename().string();

		static ImGuiComboFlags scene_combo_flags = ImGuiComboFlags_HeightLarge;
		if (ImGui::BeginCombo("Scene file", current_scene_filename.c_str(), scene_combo_flags))
		{
			static char scene_filter_buf[256] = "";

			if (ImGui::IsWindowAppearing())
			{
				scene_filter_buf[0] = '\0';

				ImGui::SetKeyboardFocusHere();
			}

			ImGui::InputTextWithHint("##scene_filter", "Filter...", scene_filter_buf, IM_ARRAYSIZE(scene_filter_buf));

			std::string filter_str = scene_filter_buf;
			std::transform(filter_str.begin(), filter_str.end(), filter_str.begin(), ::tolower);

			// File chooser first item
			if (ImGui::Selectable("..."))
			{
				const char* filters[]	= { "*" };
				std::string custom_path = Utils::open_file_dialog(filters, 1);

				if (!custom_path.empty())
					load_new_scene(custom_path);
			}

			ImGui::Separator();

			std::vector<std::string> scene_files;
			const std::vector<std::string>& supported_exts = assimp_supported_extensions();

			if (std::filesystem::exists(DATA_DIRECTORY "/GLTFs/"))
			{
				for (const auto& entry : std::filesystem::directory_iterator(DATA_DIRECTORY "/GLTFs/"))
				{
					if (entry.is_regular_file())
					{
						std::string ext = entry.path().extension().string();
						bool supported	= false;
						for (const std::string& supported_ext : supported_exts)
						{
							if (ext == supported_ext)
							{
								supported = true;
								break;
							}
						}

						if (supported)
							scene_files.push_back(entry.path().filename().string());
					}
				}
			}

			for (const std::string& filename : scene_files)
			{
				std::string lower_filename = filename;
				std::transform(lower_filename.begin(), lower_filename.end(), lower_filename.begin(), ::tolower);

				bool passes_filter = filter_str.empty() || lower_filename.find(filter_str) != std::string::npos;
				if (!passes_filter)
					continue;

				bool is_selected = (filename == current_scene_filename);

				if (ImGui::Selectable(filename.c_str(), is_selected))
				{
					load_new_scene(DATA_DIRECTORY "/GLTFs/" + filename);
				}

				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}

			ImGui::EndCombo();
		}
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::SeparatorText("General render settings");
	draw_render_settings_panel();
	draw_render_stopping_conditions_panel();
	draw_camera_panel();
	draw_environment_panel();
	draw_sampling_panel();
	draw_material_settings_panel();
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

	ImGui::Text("Status: %s", m_status_text.c_str());
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	if (render_settings.accumulate)
		ImGui::Text("Render time: %.3fs", m_render_window->get_current_render_time_ms() / 1000.0f);
	else
		ImGui::Text("Frame time (GPU): %.3fms", m_render_window_perf_metrics->get_current_value(GPURenderer::ALL_RENDER_PASSES_TIME_KEY));
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", render_settings.sample_number, m_render_window->get_samples_per_second(),
				m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y);
	float time_before_viewport_refresh_ms = m_render_window->get_time_ms_before_viewport_refresh();
	if (!m_render_window->is_rendering_done() && render_settings.accumulate)
	{
		// Only displaying the refresh timer if we actually need to wait before refreshin'
		// And also, not displaying that if the rendering is done

		float time_before_refresh_seconds = time_before_viewport_refresh_ms / 1000.0f;
		if (time_before_refresh_seconds > 0.0f)
			ImGui::Text("Viewport refresh in: %.3fs", time_before_refresh_seconds);
		else
		{
			// Time is < 0.0f i.e. the timer has expired and we're waiting for a refresh
			if (m_renderer->gmon_used() && m_renderer->get_gmon_render_pass()->recomputation_requested())
				// If we're waiting for GMoN, indicating it
				ImGui::Text("Viewport refresh in: 0.000s --- Waiting for GMoN");
			else
				// If we're not waiting for GMoN, just clampign so that we don't display negative values
				ImGui::Text("Viewport refresh in: %.3fs", std::max(0.0f, time_before_refresh_seconds));
		}
	}
	else if (render_settings.accumulate)
		// If the rendering is done, displaying 0.000s
		ImGui::Text("Viewport refresh in: 0.000s");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (render_settings.has_access_to_adaptive_sampling_buffers())
	{
		unsigned int converged_count   = m_renderer->get_status_buffer_values().pixel_converged_count;
		unsigned int total_pixel_count = m_renderer->m_render_resolution.x * m_renderer->m_render_resolution.y;

		bool can_print_convergence = false;
		can_print_convergence |= render_settings.sample_number > render_settings.adaptive_sampling_min_samples;
		can_print_convergence |= render_settings.stop_pixel_noise_threshold > 0.0f;

		if (can_print_convergence)
		{
			ImGui::Text("Pixels converged: %d / %d - %.4f%%", converged_count, total_pixel_count,
						static_cast<float>(converged_count) / total_pixel_count * 100.0f);

			// Adding some information on what noise threshold is being used
			std::string text = "Noise threshold: ";
			if (render_settings.enable_adaptive_sampling && render_settings.sample_number > render_settings.adaptive_sampling_min_samples)
			{
				if (render_settings.stop_pixel_noise_threshold > render_settings.adaptive_sampling_noise_threshold)
				{
					// If the pixel noise threshold is stronger, then the displayed convergence counter
					// is going to be according to the stop noise threshold so that's what we're adding in the tooltip
					// there
					text += std::format("{:.3f}", render_settings.stop_pixel_noise_threshold) + " (pixel noise threshold)\n";
					text += "Pixel proportion: " + std::format("{:.3f}", render_settings.stop_pixel_percentage_converged) + "%%";
				}
				else
					text += std::to_string(render_settings.adaptive_sampling_noise_threshold) + " (adaptive sampling)";
			}
			else if (render_settings.stop_pixel_noise_threshold > 0.0f)
			{
				text += std::format("{:.3f}", render_settings.stop_pixel_noise_threshold) + " (pixel noise threshold)\n";
				text += "Pixel proportion: " + std::format("{:.3f}", render_settings.stop_pixel_percentage_converged) + "%%";
			}

			ImGui::TreePush("Convergence info tree");
			ImGui::Text("%s", text.c_str());
			ImGui::TreePop();
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
		ImGuiRenderer::show_help_marker("Convergence is only computed when either adaptive sampling or the \"Pixel noise threshold\" render stopping condition "
										"is used.");
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (ImGui::Button("Save viewport to PNG"))
		m_render_window->get_screenshoter()->write_to_png();
	if (ImGui::Button("Copy viewport to clipboard"))
		Utils::copy_image_to_clipboard(m_render_window->get_screenshoter()->get_image(), false);

	ImGui::Separator();

	ImGui::PushItemWidth(16 * ImGui::GetFontSize());
}

void ImGuiSettingsWindow::draw_render_settings_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	// ImGui::PopItemWidth();
	if (!ImGui::CollapsingHeader("Render Settings"))
		return;
	ImGui::TreePush("Render settings tree");

	static int preset_selected			  = 0;
	std::vector<const char*> preset_items = { "Default",		"Reference path-tracer", "MIS NEE Path Tracer", "RIS NEE Path Tracer",
											  "ReSTIR DI Fast", "ReSTIR DI Efficiency",	 "ReSTIR GI",			"ReSTIR DI + GI" };
	std::vector<const char*> tooltips	  = {
		"No preset",
		"Reference, no NEE, brute-force path-tracer",
		"NEE with MIS (BSDF + Light sampling) at each vertex of the path",
		"NEE with RIS (N*BSDF + M*Light sampling) at each vertex of the path",
		"Direct lighting only (0 bounce) and ReSTIR DI. Fast settings for better framerates but converges slower than \"ReSTIR DI Efficiency\"",
		"Direct lighting only (0 bounce) and ReSTIR DI. Heavy settings for the fastest convergence rate",
		"5 bounces with ReSTIR GI and RIS at each vertex of the path",
		"\"ReSTIR DI Fast\" for the direct lighting + 5 bounces with ReSTIR GI and RIS at each vertex of the path",
	};

	ImGui::SeparatorText("Global settings presets");
	if (ImGuiRenderer::ComboWithTooltips("Rendering preset", &preset_selected, preset_items.data(), preset_items.size(), tooltips.data()))
		apply_performance_preset(static_cast<ImGuiRendererSettingsPreset>(preset_selected));

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
			m_application_settings->target_width  = m_renderer->m_render_resolution.x;
			m_application_settings->target_height = m_renderer->m_render_resolution.y;
		}
	}
	ImGuiRenderer::show_help_marker("Keeps approximately the same render resolution when "
									"resizing the application's window.");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::SeparatorText("General settings");

	if (ImGui::Checkbox("Accumulate", &render_settings.accumulate))
	{
		m_render_window->set_render_dirty(true);

		if (!render_settings.accumulate)
		{
			m_render_window->get_application_settings()->auto_sample_per_frame = false;
			render_settings.samples_per_frame								   = 1;
		}
	}

	if (ImGui::InputInt("Samples per frame", &render_settings.samples_per_frame))
	{
		// Clamping to 1
		render_settings.samples_per_frame = std::max(1, render_settings.samples_per_frame);
		// If the user manually changed to number of samples per frame, let's disable auto sample per frame
		// because the user probably doesn't want it
		m_application_settings->auto_sample_per_frame = false;
	}

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

	int nb_bounce_before_change = render_settings.nb_bounces;
	if (ImGui::InputInt("Max bounces", &render_settings.nb_bounces))
	{
		// Clamping to 0 in case the user input a negative number of bounces
		render_settings.nb_bounces = std::max(render_settings.nb_bounces, 0);

		m_render_window->set_render_dirty(true);
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	draw_russian_roulette_options();

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

void ImGuiSettingsWindow::draw_render_stopping_conditions_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (ImGui::CollapsingHeader("Render stopping condition"))
	{
		ImGui::TreePush("Stopping condition tree");
		{
			if (ImGui::InputInt("Max sample count", &m_application_settings->max_sample_count))
				m_application_settings->max_sample_count = std::max(m_application_settings->max_sample_count, 0);
			if (m_renderer->gmon_used())
			{
				// Using GMoN

				unsigned int number_of_sets = m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT);
				if (m_application_settings->max_sample_count % number_of_sets != 0)
				{
					ImGui::TreePush("Number of samples not divisible GMoN tree");

					// But the maximum number of samples isn't divisible by the number of sets
					std::string warning_text =
						"Currently using GMoN (\"Post-processing\" panel) but the number of "
						"maximum samples entered here isn't divisible by the number of GMoN sets. This means that "
						"what's displayed in the viewport will only be " +
						std::to_string(std::max(1u, m_application_settings->max_sample_count / number_of_sets)) + " samples instead of " +
						std::to_string(m_application_settings->max_sample_count) +
						".\n\n"
						""
						"You click the button to the right to round up the maximum number of samples to one that is "
						"divisible by the number of GMoN sets (" +
						std::to_string(m_renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT)) + ")";
					ImGuiRenderer::add_warning(warning_text);

					ImGui::SameLine();
					if (ImGui::Button("Round up"))
						m_application_settings->max_sample_count =
							std::ceil(m_application_settings->max_sample_count / static_cast<float>(number_of_sets)) * number_of_sets;

					ImGui::TreePop();
				}
			}

			if (ImGui::InputFloat("Max render time (s)", &m_application_settings->max_render_time))
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
			ImGui::Checkbox("Use pixel noise threshold stopping condition", &render_settings.use_pixel_stop_noise_threshold);
			ImGuiRenderer::show_help_marker("If enabled, stops the renderer after a certain proportion "
											"of pixels of the image have converged. \"converged\" is evaluated according to the "
											"threshold of the adaptive sampling if it is enabled. If adaptive sampling is not "
											"enabled, \"converged\" is defined by the \"Pixel noise threshold\" variance "
											"threshold below.");

			ImGui::BeginDisabled(!render_settings.use_pixel_stop_noise_threshold);
			{
				if (ImGui::InputFloat("Pixel proportion", &render_settings.stop_pixel_percentage_converged))
					render_settings.stop_pixel_percentage_converged = std::max(0.0f, std::min(render_settings.stop_pixel_percentage_converged, 100.0f));
				ImGuiRenderer::show_help_marker("The proportion of pixels that need to have converge "
												"to the noise threshold for the rendering to stop. In percentage [0, 100].");
			}
			ImGui::EndDisabled();

			ImGui::BeginDisabled(render_settings.enable_adaptive_sampling || !render_settings.use_pixel_stop_noise_threshold);
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

				ImGui::InputInt("Minimum sample count", &m_application_settings->pixel_stop_noise_threshold_min_sample_count);
				ImGuiRenderer::show_help_marker("How many samples to render before evaluating the number of pixels that have reached "
												"the noise threshold.\n\n"
												""
												"This setting only applies to the \"pixel stop noise threshold\" feature.\n"
												"It does not apply to adaptive sampling.\n"
												"Adaptive sampling has its own minimum sample count.");

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
}

void ImGuiSettingsWindow::draw_russian_roulette_options()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (ImGui::Checkbox("Do Russian Roulette", &render_settings.do_russian_roulette))
		m_render_window->set_render_dirty(true);

	const char* items[] = { "- Max throughput", "- Arnold, Langlands, 2014" };
	if (ImGui::Combo("Termination method", (int*)&render_settings.path_russian_roulette_method, items, IM_ARRAYSIZE(items)))
		m_render_window->set_render_dirty(true);

	static bool min_depth_modified = false;
	if (!min_depth_modified)
		render_settings.russian_roulette_min_depth = std::min(5, render_settings.nb_bounces / 2);
	if (ImGui::SliderInt("RR min depth", &render_settings.russian_roulette_min_depth, 0, render_settings.nb_bounces + 1))
	{
		m_render_window->set_render_dirty(true);
		min_depth_modified = true;
	}
	ImGuiRenderer::show_help_marker("After how many bounces can russian roulette kick in? "
									"For example, 0 means that the camera ray hits, and then the next bounce "
									"is already susceptible to russian roulette kill. 1 would mean that the first "
									"bounce is never going to be cutoff by the russian roulette.");
	if (ImGui::SliderFloat("RR throughput clamp", &render_settings.russian_roulette_throughput_clamp, 1.0f, 20.0f))
		m_render_window->set_render_dirty(true);
	ImGuiRenderer::show_help_marker("After applying russian roulette (dividing by the continuation probability) "
									"the energy added to the ray throughput is clamped to this maximum value.\n"
									"\n"
									"This is biased and darkens the image the lower the threshold but it helps "
									"reduce variance and fireflies introduced by the russian roulette --> faster "
									"convergence.\n"
									"\n"
									"0 for no clamping.");

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.0f, 0.0f, 1.0f));		   // Red
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.2f, 0.2f, 1.0f)); // Lighter red when hovered
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.0f, 0.0f, 1.0f));  // Darker red when clicked
	if (ImGui::Button("Reset render"))
		m_render_window->set_render_dirty(true);
	ImGui::PopStyleColor(3);
}

void ImGuiSettingsWindow::display_view_selector()
{
	HIPRTRenderSettings& render_settings				   = m_renderer->get_render_settings();
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

	if (ImGui::BeginCombo("Display view", items[display_view_selected_index]))
	{
		for (int i = 0; i < items.size(); i++)
		{
			const bool is_selected		  = (display_view_selected_index == i);
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

	DisplaySettings& display_settings		   = display_view_system->get_display_settings();
	DisplayViewType display_view_type_selected = static_cast<DisplayViewType>(display_view_selected_index);
	// Adding some more UI elements for certain display views
	switch (display_view_type_selected)
	{
	case DisplayViewType::WHITE_FURNACE_THRESHOLD:
		bool viewport_update_needed = false;

		viewport_update_needed |= ImGui::Checkbox("Use low threshold", &display_settings.white_furnace_display_use_low_threshold);
		ImGuiRenderer::show_help_marker("If checked, the white furnace threshold shader will display "
										"pixel that lose energy as green. Pixels will not be highlighted "
										"if unchecked");
		viewport_update_needed |= ImGui::Checkbox("Use high threshold", &display_settings.white_furnace_display_use_high_threshold);
		ImGuiRenderer::show_help_marker("If checked, the white furnace threshold shader will display "
										"pixel that gain energy as red. Pixels will not be highlighted "
										"if unchecked");

		if (viewport_update_needed)
			m_render_window->set_force_viewport_refresh(true);

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
		return !m_renderer->gmon_used();

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
		m_renderer->get_gmon_render_pass()->get_gmon_data().use_gmon = true;
		toggle_gmon();

		return;

	case DisplayViewType::DENOISED_BLEND:
		ImGuiRenderer::add_tooltip("This display view is disabled because the denoiser isn't enabled. Click to enable the denoiser.");
		return;

	default:
		break;
	}
}

void ImGuiSettingsWindow::apply_performance_preset(ImGuiRendererSettingsPreset performance_preset)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	switch (performance_preset)
	{
	case SETTINGS_PRESET_DEFAULT:
		break;

	case SETTINGS_PRESET_REFERENCE_BRUTE_FORCE_PATH_TRACER:
		render_settings.do_alpha_testing			= true;
		render_settings.direct_contribution_clamp	= 0.0f;
		render_settings.indirect_contribution_clamp = 0.0f;
		render_settings.envmap_contribution_clamp	= 0.0f;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_NO_DIRECT_LIGHT_SAMPLING);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, ESS_NO_SAMPLING);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_BSDF);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION, KERNEL_OPTION_FALSE);
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);

		break;

	case SETTINGS_PRESET_MIS_NEE_PATH_TRACER:
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_MIS_LIGHT_BSDF);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, ESS_ALIAS_TABLE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_BSDF);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION, KERNEL_OPTION_TRUE);
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);

		break;

	case SETTINGS_PRESET_RIS_NEE_PATH_TRACER:
		render_settings.ris_settings.number_of_bsdf_candidates	= 1;
		render_settings.ris_settings.number_of_light_candidates = 4;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_RIS_BSDF_AND_LIGHT);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, ESS_ALIAS_TABLE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_BSDF);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION, KERNEL_OPTION_TRUE);
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);

		break;

	case SETTINGS_PRESET_RESTIR_DI_FAST:
		render_settings.nb_bounces = 0;

		render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass   = true;
		render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location = false;
		render_settings.restir_di_settings.common_spatial_pass.number_of_passes		   = 1;
		render_settings.restir_di_settings.common_spatial_pass.reuse_neighbor_count	   = 3;
		// Reuse radius 1% of the resolution
		render_settings.restir_di_settings.common_spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * 0.01f;

		render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass = true;

		render_settings.restir_di_settings.neighbor_similarity_settings.use_normal_similarity_heuristic	   = true;
		render_settings.restir_di_settings.neighbor_similarity_settings.use_plane_distance_heuristic	   = true;
		render_settings.restir_di_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		render_settings.restir_di_settings.m_cap														   = 5;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_RESTIR_DI);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE, RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, ESS_ALIAS_TABLE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_BSDF);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION, KERNEL_OPTION_TRUE);
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);

		break;

	case SETTINGS_PRESET_RESTIR_DI_EFFICIENCY:
		render_settings.nb_bounces = 0;

		render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass   = true;
		render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location = false;
		render_settings.restir_di_settings.common_spatial_pass.number_of_passes		   = 2;
		render_settings.restir_di_settings.common_spatial_pass.reuse_neighbor_count	   = 8;
		// Reuse radius 1% of the resolution
		render_settings.restir_di_settings.common_spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * 0.01f;

		render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass = true;

		render_settings.restir_di_settings.neighbor_similarity_settings.use_normal_similarity_heuristic	   = true;
		render_settings.restir_di_settings.neighbor_similarity_settings.use_plane_distance_heuristic	   = true;
		render_settings.restir_di_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		render_settings.restir_di_settings.m_cap														   = 3;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_RESTIR_DI);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE, RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, ESS_ALIAS_TABLE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_BSDF);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION, KERNEL_OPTION_TRUE);
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);

		break;

	case SETTINGS_PRESET_RESTIR_GI:
		render_settings.nb_bounces								= 5;
		render_settings.ris_settings.number_of_bsdf_candidates	= 1;
		render_settings.ris_settings.number_of_light_candidates = 4;

		render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass   = true;
		render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location = false;
		render_settings.restir_di_settings.common_spatial_pass.number_of_passes		   = 2;
		render_settings.restir_di_settings.common_spatial_pass.reuse_neighbor_count	   = 8;
		// Reuse radius 1% of the resolution
		render_settings.restir_di_settings.common_spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * 0.01f;

		render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass = true;

		render_settings.restir_di_settings.neighbor_similarity_settings.use_normal_similarity_heuristic	   = true;
		render_settings.restir_di_settings.neighbor_similarity_settings.use_plane_distance_heuristic	   = true;
		render_settings.restir_di_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		render_settings.restir_di_settings.m_cap														   = 3;

		render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass   = true;
		render_settings.restir_gi_settings.common_spatial_pass.debug_neighbor_location = false;
		render_settings.restir_gi_settings.common_spatial_pass.number_of_passes		   = 2;
		render_settings.restir_gi_settings.common_spatial_pass.reuse_neighbor_count	   = 8;
		// Reuse radius 1% of the resolution
		render_settings.restir_gi_settings.common_spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * 0.01f;

		render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass = true;

		render_settings.restir_gi_settings.neighbor_similarity_settings.use_normal_similarity_heuristic	   = true;
		render_settings.restir_gi_settings.neighbor_similarity_settings.use_plane_distance_heuristic	   = true;
		render_settings.restir_gi_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		render_settings.restir_gi_settings.use_jacobian_rejection_heuristic								   = true;
		render_settings.restir_gi_settings.use_neighbor_sample_point_roughness_heuristic				   = true;
		render_settings.restir_gi_settings.m_cap														   = 3;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_RESTIR_DI);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE, RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_TYPE, RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_USE_VISIBILITY, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, ESS_ALIAS_TABLE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_RESTIR_GI);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION, KERNEL_OPTION_TRUE);
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);

		break;

	case SETTINGS_PRESET_RESTIR_DI_GI:
		render_settings.nb_bounces								= 5;
		render_settings.ris_settings.number_of_bsdf_candidates	= 1;
		render_settings.ris_settings.number_of_light_candidates = 4;

		render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass   = true;
		render_settings.restir_gi_settings.common_spatial_pass.debug_neighbor_location = false;
		render_settings.restir_gi_settings.common_spatial_pass.number_of_passes		   = 2;
		render_settings.restir_gi_settings.common_spatial_pass.reuse_neighbor_count	   = 8;
		// Reuse radius 1% of the resolution
		render_settings.restir_gi_settings.common_spatial_pass.reuse_radius =
			hippt::max(m_renderer->m_render_resolution.x, m_renderer->m_render_resolution.y) * 0.01f;

		render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass = true;

		render_settings.restir_gi_settings.neighbor_similarity_settings.use_normal_similarity_heuristic	   = true;
		render_settings.restir_gi_settings.neighbor_similarity_settings.use_plane_distance_heuristic	   = true;
		render_settings.restir_gi_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = false;
		render_settings.restir_gi_settings.use_jacobian_rejection_heuristic								   = true;
		render_settings.restir_gi_settings.use_neighbor_sample_point_roughness_heuristic				   = true;
		render_settings.restir_gi_settings.m_cap														   = 3;

		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_RIS_BSDF_AND_LIGHT);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_TYPE, RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_USE_VISIBILITY, KERNEL_OPTION_TRUE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY, KERNEL_OPTION_FALSE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY, ESS_ALIAS_TABLE);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_RESTIR_GI);
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION, KERNEL_OPTION_TRUE);
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);

		break;

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
	Camera& camera						 = renderer->get_camera();

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

		static float camera_fov = camera.vertical_fov * hippt::M_INV_PI * 180.0f;
		if (ImGui::SliderFloat("FOV", &camera_fov, 0.0f, 180.0f, "%.3fdeg", ImGuiSliderFlags_AlwaysClamp))
		{
			camera.set_FOV_radians(camera_fov / 180.0f * hippt::M_Pi);

			render_window->set_render_dirty(true);
		}

		if (ImGui::SliderFloat("Camera movement speed", &camera.user_movement_speed_multiplier, 0.0f, 10.0f))
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
			const std::vector<std::string>& mesh_names	   = renderer->get_mesh_names();
			const std::vector<std::string>& material_names = renderer->get_material_names();
			for (int n = 0; n < mesh_names.size(); n++)
			{
				const bool is_selected = (selected_object == n);

				const std::string& mesh_name	 = mesh_names[n];
				const std::string& material_name = material_names[renderer->get_mesh_material_indices()[n]];
				std::string object_text			 = mesh_name + " (" + material_name + ")";
				if (ImGui::Selectable(object_text.c_str(), is_selected))
				{
					selected_object = n;

					float3_t object_center = renderer->get_mesh_bounding_boxes()[n].get_center();
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

void ImGuiSettingsWindow::load_new_envmap(const std::string& filepath)
{
	Image32Bit envmap_image;

	ThreadManager::start_serial_thread(ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD, ThreadFunctions::read_envmap, std::ref(envmap_image), filepath, 4, true);
	m_renderer->set_envmap(envmap_image, filepath);
	ThreadManager::join_threads(ThreadManager::RENDERER_SET_ENVMAP);

	envmap_image.free();
}

void ImGuiSettingsWindow::load_new_scene(std::string filepath)
{
	Assimp::Importer assimp_importer;
	Scene parsed_scene;
	SceneParserOptions options(filepath);
	options.override_aspect_ratio = m_renderer->get_camera().aspect;

	SceneParser::parse_scene_file(filepath, assimp_importer, parsed_scene, options);

	m_renderer->set_camera(parsed_scene.camera);
	m_renderer->set_scene(parsed_scene);
	m_renderer->set_scene_filepath(filepath);

	ThreadManager::join_all_threads();

	m_render_window->set_render_dirty(true);
}

void ImGuiSettingsWindow::draw_environment_panel()
{
	bool render_made_piggy = false;

	if (ImGui::CollapsingHeader("Environment"))
	{
		ImGui::TreePush("Environment tree");

		WorldSettings& world_settings = m_renderer->get_render_data().world_settings;

		bool has_envmap = m_renderer->has_envmap();
		render_made_piggy |= ImGui::RadioButton("None", ((int*)&world_settings.ambient_light_type), 0);
		ImGui::SameLine();
		render_made_piggy |= ImGui::RadioButton("Use uniform lighting", ((int*)&world_settings.ambient_light_type), 1);
		ImGui::SameLine();
		ImGui::BeginDisabled(!has_envmap);
		render_made_piggy |= ImGui::RadioButton("Use envmap lighting", ((int*)&world_settings.ambient_light_type), 2);
		if (!has_envmap)
			// Showing a tooltip for why the envmap button is disabled
			ImGuiRenderer::show_help_marker("No envmap loaded.");
		ImGui::EndDisabled();

		if (world_settings.ambient_light_type == AmbientLightType::UNIFORM)
		{
			render_made_piggy |=
				ImGui::ColorEdit3("Uniform light color", (float*)&world_settings.uniform_light_color, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
		}
		else if (world_settings.ambient_light_type == AmbientLightType::ENVMAP)
		{
			std::string current_envmap_filepath = m_renderer->get_envmap().get_envmap_filepath();
			std::string current_envmap_filename = current_envmap_filepath.empty() ? "None" : std::filesystem::path(current_envmap_filepath).filename().string();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (ImGui::BeginCombo("Envmap file", current_envmap_filename.c_str()))
			{
				std::vector<std::string> skysphere_files;

				if (std::filesystem::exists(DATA_DIRECTORY "/Skyspheres/"))
					for (const auto& entry : std::filesystem::directory_iterator(DATA_DIRECTORY "/Skyspheres/"))
						if (entry.is_regular_file())
							skysphere_files.push_back(entry.path().filename().string());

				if (ImGui::Selectable("..."))
				{
					const char* filters[]	= { "*.hdr", "*.exr" };
					std::string custom_path = Utils::open_file_dialog(filters, 2);

					if (!custom_path.empty())
					{
						load_new_envmap(custom_path);
						render_made_piggy = true;
					}
				}

				ImGui::Separator();

				for (const std::string& filename : skysphere_files)
				{
					bool is_selected = (filename == current_envmap_filename);

					if (ImGui::Selectable(filename.c_str(), is_selected))
					{
						load_new_envmap(DATA_DIRECTORY "/Skyspheres/" + filename);
						render_made_piggy = true;
					}

					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}

				ImGui::EndCombo();
			}

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
			render_made_piggy |= ImGui::SliderFloat("Envmap intensity", (float*)&world_settings.envmap_intensity, 0.0f, 10.0f);
			ImGui::TreePush("Envmap intensity tree");
			render_made_piggy |= ImGui::Checkbox("Scale background intensity", (bool*)&world_settings.envmap_scale_background_intensity);
			if (world_settings.envmap_intensity != 1.0f && !world_settings.envmap_scale_background_intensity)
			{
				ImGuiRenderer::add_warning("Using a custom envmap intensity without scaling the background "
										   "intensity can result in discrepancies when looking at the envmap through a mirror or "
										   "transparent glass for example (glass with IOR 1.0f). In these rare cases, the envmap "
										   "will appear brighter through the objects than when viewed directly.\n"
										   "This is because scaling the envmap intensity without scaling how it appears to camera rays isn't "
										   "physically accurate.");
			}

			ImGui::TreePop();
		}

		// Ensuring no negative light color
		world_settings.uniform_light_color.clamp(0.0f, 1.0e38f);

		ImGui::TreePop();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}

	if (render_made_piggy)
		m_render_window->set_render_dirty(true);
}

void ImGuiSettingsWindow::draw_sampling_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
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
			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) == LSS_RESTIR_DI ||
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_GI)
			{
				ImGuiRenderer::add_warning("Adaptive sampling may not be efficient when used with ReSTIR. This is because "
										   "ReSTIR looks around the current pixel to find neighbor to reuse but with adaptive sampling enabled, most pixels "
										   "are not going to be sampled anymore. This means that these pixels contain stale samples that strongly biases "
										   "ReSTIR is reused.\n"
										   "Thus, these sample are not reused to avoid bias but this then drastically reduces the efficiency of ReSTIR and the "
										   "overall "
										   "setup becomes inefficient.");
			}

			float adaptive_sampling_noise_threshold_before = render_settings.adaptive_sampling_noise_threshold;
			ImGui::BeginDisabled(!render_settings.enable_adaptive_sampling);
			if (ImGui::InputInt("Minimum samples", &render_settings.adaptive_sampling_min_samples))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("How many samples to wait before adaptive sampling activates.\n\n"
											""
											"The general rule is to keep this value as low as possible without getting conspicuous black/unconverged pixels.");
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

		if (ImGui::CollapsingHeader("Emissive geometry sampling"))
		{
			ImGui::TreePush("Direct lighting sampling tree");

			if (ImGui::Checkbox("Enable DI", &render_settings.enable_direct_lighting))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker(std::string("Whether or not to integrate direct lighting (NEE) at the primary hit (G-buffer surface)."));
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			const char* items_base_strategy[]	 = { "- Uniform sampling", "- Power sampling", "- Light tree ATS (Conty & Kulla 2018)",
													 "- SG light tree (Tokuyoshi et al. 2024)", "- ReGIR + Cache cells (Experimental)" };
			const char* tooltips_base_strategy[] = {
				"All lights are sampled uniformly.",

				"Lights are sampled proportionally to their power.",

				"Implementation of [Importance Sampling of Many Lights with Adaptive Tree Splitting, Conty & Kulla, 2018]",

				"Implementation of [Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting, Tokuyoshi et al., 2024]",

				"Uses ReGIR to sample lights.\n\n"
				"Highly custom implementation of [Rendering many lights with grid - based reservoirs, Boksansky, 2021] + Disney's cache points: "
				"[Cache Points For Production-Scale Occlusion-Aware Many-Lights Sampling And Volumetric Scattering, Li et al. 2024]"
			};

			bool base_sampling_strategy_changed = ImGuiRenderer::ComboWithTooltips(
				"Light sampling strategy", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY),
				items_base_strategy, IM_ARRAYSIZE(items_base_strategy), tooltips_base_strategy);

			const char* items[]	   = { "- No direct light sampling",
									   "- Light sampling",
									   "- BSDF Sampling",
									   "- MIS (1 Light + 1 BSDF)",
									   "- RIS BDSF + Light candidates",
									   "- RISLTC BSDF + Light candidates",
									   "- LTC Shading",
									   "- ReSTIR DI (Primary hit only)" };
			const char* tooltips[] = {
				"No direct light sampling. Emission is only gathered if rays happen to bounce into the lights.",

				"Samples one random light in the scene without MIS. Efficient as long as there are not too many lights in the scene and no glossy/specular "
				"surfaces.",

				"Samples lights only using one BSDF sample.",

				"Samples one random light in the scene with MIS (Multiple Importance Sampling) : light sample + BRDF sample.",

				"Samples lights in the scene with RIS (Resampled Importance Sampling) with both BSDF and light candidates. The number of light or BSDF "
				"candidates can be controlled.",

				"Experimental and unfinished! Samples lights in the scene with RISLTC (Shah et. al, 2023) with both BSDF and light candidates. The number of "
				"light or BSDF candidates can "
				"be controlled.",

				"Experimental and unfinished! Uses Linearly Transformed Cosines to analytically shade lights. This is biased as shadowing is not taken into "
				"account. Not all BSDF lobe "
				"configurations are supported.",

				"Uses ReSTIR DI to sample direct lighting at the first bounce in the scene. Later bounces use another of the above strategies which can be "
				"changed in the ReSTIR DI settings.",
			};

			const bool no_direct_light_sampling_disabled = false;
			const bool uniform_one_light_disabled =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR;
			const bool bsdf_sampling_disabled = false;

			const bool regir		= global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR;
			const bool mis_disabled = regir;

			const bool ris_disabled			= false;
			const bool risltc_disabled		= regir;
			const bool ltc_shading_disabled = regir;
			const bool restir_di_disabled	= false;

			unsigned char disabled_items[] = { no_direct_light_sampling_disabled,
											   uniform_one_light_disabled,
											   bsdf_sampling_disabled,
											   mis_disabled,
											   ris_disabled,
											   risltc_disabled,
											   ltc_shading_disabled,
											   restir_di_disabled };
			// If the user chooses a combination of base sampling strategy + sampling technique that is forbidden,
			// we're going to fallback automatically to something that is allowed and this array gives the default
			// fallback for the techniques in the same order that they are in the 'items_base_strategy' array.
			int preferred_fallback_technique[] = { LSS_ONE_LIGHT, LSS_ONE_LIGHT, LSS_ONE_LIGHT, LSS_ONE_LIGHT, LSS_RIS_BSDF_AND_LIGHT };
			static_assert(IM_ARRAYSIZE(preferred_fallback_technique) == IM_ARRAYSIZE(items_base_strategy));

			bool nee_estimator_disabled = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PT;
			ImGui::BeginDisabled(nee_estimator_disabled);
			if (ImGuiRenderer::ComboWithTooltips("NEE Estimator",
												 global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR),
												 items, IM_ARRAYSIZE(items), tooltips, disabled_items))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			if (nee_estimator_disabled)
				ImGuiRenderer::add_tooltip("The NEE estimator is controlled by ReSTIR PT.");
			ImGui::EndDisabled(); // nee_estimator_disabled

			if (disabled_items[global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR)])
			{
				int preferred_base_strategy =
					preferred_fallback_technique[global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY)];
				if (disabled_items[preferred_base_strategy])
				{
					// If also the preferred technique is disabled, choosing the first enabled one
					for (int i = 1; i < IM_ARRAYSIZE(disabled_items); i++)
					{
						if (!disabled_items[i])
						{
							global_kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, i);

							m_renderer->recompute_emissives_sampling_data_structure();
							m_renderer->recompile_kernels();
							m_render_window->set_render_dirty(true);

							break;
						}
					}
				}
				else
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, preferred_base_strategy);

					m_renderer->recompute_emissives_sampling_data_structure();
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				// We just took care of recompiling the kernels and
				// everything so we don't need to do it again below
				base_sampling_strategy_changed = false;
			}

			if (base_sampling_strategy_changed)
			{
				// If the base light sampling strategy changed, we need to update the
				// kernels

				// Will recompute the alias table if necessary
				m_renderer->recompute_emissives_sampling_data_structure();

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			const char* items_triangle_sampling[]	 = { "- Uniform area", "- Solid angle", "- Projected solid angle" };
			const char* tooltips_triangle_sampling[] = {
				"Most basic sampling method, fastest but has the highest variance. Does not take the shading point into account",

				"Slower than uniformly sampling the area but has lower variance. Takes the geometry term into account but not the cosine "
				"term at the shading point",

				"Slower than sampling according to solid angle but has even lower variance. Takes the cosine term at the shading point into "
				"account on top of the geometry term."
			};
			if (ImGuiRenderer::ComboWithTooltips(
					"Triangle sampling strategy",
					global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY), items_triangle_sampling,
					IM_ARRAYSIZE(items_triangle_sampling), tooltips_triangle_sampling))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY) ==
				TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE)
			{
				ImGui::TreePush("Projected solid angle settings tree");

				if (ImGui::SliderFloat("Minimum solid angle", &render_settings.projected_solid_angle_sampling_threshold, 0.0f, 2.0f * hippt::M_Pi, "%.3f",
									   ImGuiSliderFlags_AlwaysClamp))
					m_render_window->set_render_dirty(true);

				ImGui::TreePop();
			}

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY) ==
					TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE ||
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY) ==
					TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE)
			{
				ImGui::TreePush("Solid angle triangle sampling use LTC tree");

				static bool use_ltc = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE_USE_LTC);
				if (ImGui::Checkbox("Use LTC sampling", &use_ltc))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE_USE_LTC,
														   use_ltc ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker(
					"If checked, the LTC-based method from [BRDF Importance Sampling for Polygonal Lights, Peters 2021] will be used "
					"for sampling a point on emissive triangle.This has for effect of taking the BRDF into account "
					"when sampling the point on the triangle, massively increasing the quality of the sampling on glossy surfaces.\n\n"
					"Warning: LTC-based sampling may be slightly biased due to the BRDF approximation error of LTCs.");

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE_USE_LTC) == KERNEL_OPTION_TRUE &&
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY) ==
						TRIANGLE_POINT_SAMPLING_STRATEGY_SOLID_ANGLE ||
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_STRATEGY) ==
					TRIANGLE_POINT_SAMPLING_STRATEGY_PROJECTED_SOLID_ANGLE)
				draw_ltc_settings_panel();

			switch (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY))
			{
			case LSS_BASE_REGIR:
				draw_ReGIR_settings_panel();

				break;

			case LSS_BASE_LIGHT_TREE_ATS:
				draw_light_tree_ATS_settings_panel();

				break;

			case LSS_BASE_LIGHT_TREE_SG:
				draw_light_tree_SG_settings_panel();

				break;
			}

			// Display additional widgets to control the parameters of the direct light
			// sampling strategy chosen (the number of candidates for RIS for example)
			switch (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR))
			{
			case LSS_NO_DIRECT_LIGHT_SAMPLING:
				break;

			case LSS_ONE_LIGHT:
				break;

			case LSS_MIS_LIGHT_BSDF:
				break;

			case LSS_RIS_BSDF_AND_LIGHT:
			{
				draw_ris_settings_panel();
				if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PT)
					draw_ReSTIR_PT_light_sampling_panel();

				break;
			}

			case LSS_RISLTC:
			{
				draw_risltc_settings_panel();

				break;
			}

			case LSS_RESTIR_DI:
			{
				draw_ReSTIR_DI_settings_panel();

				break;
			}

			break;

			default:
				break;
			}

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) != LSS_BSDF &&
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) != LSS_NO_DIRECT_LIGHT_SAMPLING)
				draw_next_event_estimation_plus_plus_panel();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Envmap sampling"))
		{
			ImGui::TreePush("Envmap sampling tree");

			// Disabled if no envmap loaded
			bool envmap_sampling_disabled = m_renderer->get_envmap().get_width() == 0;

			ImGui::BeginDisabled(envmap_sampling_disabled);
			const char* items[]	   = { "- No envmap sampling", "- Importance Sampling - Binary Search", "- Importance Sampling - Alias Table " };
			const char* tooltips[] = { "The envmap will not be importance sampled. Should behave okay for low frequency envmaps but this is going to be "
									   "extremely inefficient for high frequency envmaps.",
									   "Importance samples a texel of the environment map proportionally to its luminance using a binary search on the CDF "
									   "distributions of the envmap luminance. Good convergence.",
									   "Importance samples a texel of the environment map proportionally to its luminance using an alias table for constant "
									   "time sampling. Good convergence and faster than \"Binary Search\"." };

			if (ImGuiRenderer::ComboWithTooltips("Sampling strategy",
												 global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY),
												 items, IM_ARRAYSIZE(items), tooltips))
			{
				ThreadManager::start_thread("RecomputeEnvmapSamplingStructure",
											[this]() { m_renderer->get_envmap().recompute_sampling_data_structure(m_renderer.get()); });

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);

				ThreadManager::join_threads("RecomputeEnvmapSamplingStructure");
			}

			if (envmap_sampling_disabled)
				ImGuiRenderer::add_tooltip("Disabled because no envmap is loaded in the renderer.");

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) != ESS_NO_SAMPLING)
			{
				ImGui::Text("Sampling structure VRAM usage: %.3fMB", m_renderer->get_envmap().get_sampling_structure_VRAM_usage());
				ImGui::Dummy(ImVec2(0.0f, 20.0f));

				// If we do have an importance sampling strategy
				bool do_envmap_bsdf_mis = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS);
				if (ImGui::Checkbox("Do MIS with BSDF", &do_envmap_bsdf_mis))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BSDF_MIS,
														   do_envmap_bsdf_mis ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Whether or not to shoot a BSDF ray when sampling the envmap.\n\n"
												""
												"Useful on specular/glossy surfaces.");

				bool do_envmap_bilinear_filtering = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING);
				if (ImGui::Checkbox("Do bilinear filtering", &do_envmap_bilinear_filtering))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_DO_BILINEAR_FILTERING,
														   do_envmap_bilinear_filtering ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Whether or not to do bilinear filtering when sampling the envmap.\n\n"
												""
												"This is mostly useful when the camera is looking straigth at the envmap and we don't "
												"have camera ray jittering on: in this case, bilinear filtering will hide the "
												"pixelated look of the envmap.");
			}
			ImGui::EndDisabled();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Path sampling"))
		{
			ImGui::TreePush("Path sampling tree");

			const char* items[]	   = { "- BSDF sampling", "- ReSTIR GI", "- ReSTIR PT", "- ReSTIR PG" };
			const char* tooltips[] = {
				"Classical BSDF path tracing: sample the BSDF at each bounce for the next direction.",
				"Uses ReSTIR GI to resample a path to shade for the pixel. Biased (although barely noticeable by design of resampling full path trees instead "
				"of just paths as ReSTIR PT)",
				"Uses ReSTIR PT to resample a path to shade for the pixel. The difference with ReSTIR GI is that is resamples paths and not full path trees, "
				"guaranteeing unbiasedness.",
				"Uses ReSTIR Path Guiding piggy-backing on another ReSTIR path sampler to improve path sampling distributions over time with guiding.",
			};
			if (ImGuiRenderer::ComboWithTooltips("Sampling strategy",
												 global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY), items,
												 IM_ARRAYSIZE(items), tooltips))
			{
				if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PG)
				{
					// This is ReSTIR PG. Its enabled through ReSTIR GI/PT so let's enable ReSTIR PT
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY, PATH_SAMPLING_RESTIR_PT);

					// Automatically enabling ReSTIR PG still for convenience
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_PG_ENABLE, KERNEL_OPTION_TRUE);
				}

				if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PT)
					// ReSTIR PT always uses RIS for NEE
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR, LSS_RIS_BSDF_AND_LIGHT);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			switch (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY))
			{
			case PATH_SAMPLING_RESTIR_GI:
			{
				if (ImGui::CollapsingHeader("ReSTIR GI"))
				{
					ImGui::TreePush("ReSTIR GI tree");

					static float last_VRAM_usage = 0.0f;
					if (m_renderer->get_ReSTIR_GI_render_pass())
						last_VRAM_usage = m_renderer->get_ReSTIR_GI_render_pass()->get_VRAM_usage();
					ImGui::Text("VRAM Usage: %.3fMB", last_VRAM_usage);

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					if (ImGui::SliderInt("M-cap", &render_settings.restir_gi_settings.m_cap, 0, 255, "%d", ImGuiSliderFlags_AlwaysClamp))
					{
						render_settings.restir_gi_settings.m_cap = std::max(0, render_settings.restir_gi_settings.m_cap);
						if (render_settings.accumulate)
							m_render_window->set_render_dirty(true);
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));

					ImGui::PushItemWidth(12 * ImGui::GetFontSize());
					draw_ReSTIR_temporal_reuse_panel<ReSTIR_VARIANT_GI>(
						[&render_settings, this]()
						{
							if (ImGui::Checkbox("Do Temporal Reuse", &render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass))
								m_render_window->set_render_dirty(true);
						});
					draw_ReSTIR_spatial_reuse_panel<ReSTIR_VARIANT_GI>(
						[&render_settings, this]()
						{
							if (ImGui::Checkbox("Do spatial reuse", &render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass))
								m_render_window->set_render_dirty(true);
						});
					ImGui::PopItemWidth();

					if (ImGui::CollapsingHeader("Rejection Heuristics"))
					{
						ImGui::TreePush("ReSTIR GI - Rejection Heuristics Tree");

						draw_ReSTIR_neighbor_heuristics_panel<ReSTIR_VARIANT_GI>();

						ImGui::TreePop();
						ImGui::Dummy(ImVec2(0.0f, 20.0f));
					}

					draw_ReSTIR_bias_correction_panel<ReSTIR_VARIANT_GI>();

					if (ImGui::CollapsingHeader("Debug"))
					{
						ImGui::TreePush("ReSTIR GI options tree");

						if (ImGui::Checkbox("Debug neighbor reuse positions", &render_settings.restir_gi_settings.common_spatial_pass.debug_neighbor_location))
							m_render_window->set_render_dirty(true);
						ImGuiRenderer::show_help_marker("If checked, neighbor in the spatial reuse pass will be hardcoded to always be "
														"15 pixels to the right, not in a circle. This makes spotting bias easier when debugging.");
						if (render_settings.restir_gi_settings.common_spatial_pass.debug_neighbor_location)
						{
							ImGui::TreePush("Debug neighbor location vertical tree");

							ImGui::Text("Debug reuse direction");
							bool reuse_direction_changed = false;
							reuse_direction_changed |= ImGui::RadioButton(
								"Horizontally", ((int*)&render_settings.restir_gi_settings.common_spatial_pass.debug_neighbor_location_direction), 0);
							ImGui::SameLine();
							reuse_direction_changed |= ImGui::RadioButton(
								"Vertically", ((int*)&render_settings.restir_gi_settings.common_spatial_pass.debug_neighbor_location_direction), 1);
							ImGui::SameLine();
							reuse_direction_changed |= ImGui::RadioButton(
								"Diagonally", ((int*)&render_settings.restir_gi_settings.common_spatial_pass.debug_neighbor_location_direction), 2);

							if (reuse_direction_changed)
								m_render_window->set_render_dirty(true);

							ImGui::TreePop();
						}

						ImGui::Dummy(ImVec2(0.0f, 20.0f));
						std::vector<const char*> debug_view_items = { "- No debug view",
																	  "- Shade only initial candidates",
																	  "- Final reservoir UCW",
																	  "- Final reservoir target function",
																	  "- Final reservoir weight sum",
																	  "- Final reservoir M",
																	  "- Per pixel reuse radius",
																	  "- Valid directions percentage" };
						if (ImGui::Combo("Debug view", (int*)&render_settings.restir_gi_settings.debug_view, debug_view_items.data(), debug_view_items.size()))
						{
							int macro_value_before =
								global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED);
							if (render_settings.restir_gi_settings.debug_view == ReSTIRGIDebugView::GI_SHADE_ONLY_INITIAL_CANDIDATES)
								global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED,
																	   KERNEL_OPTION_TRUE);
							else
								global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_GI_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED,
																	   KERNEL_OPTION_FALSE);
							int macro_value_after =
								global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED);

							bool macro_option_changed = macro_value_before != macro_value_after;
							if (macro_option_changed)
								m_renderer->recompile_kernels();

							m_render_window->set_render_dirty(true);
						}
						if (ImGui::SliderFloat("Debug view scale factor", &render_settings.restir_gi_settings.debug_view_scale_factor, 0.0f, 1.0f))
							m_render_window->set_render_dirty(true);

						ImGui::TreePop(); // Debug tree
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop(); // ReSTIR GI Tree
				}

				break;
			}

			case PATH_SAMPLING_RESTIR_PT:
			{
				if (ImGui::CollapsingHeader("ReSTIR PT"))
				{
					ImGui::TreePush("ReSTIR PT tree");
					static float last_VRAM_usage = 0.0f;
					if (m_renderer->get_ReSTIR_PT_render_pass())
						last_VRAM_usage = m_renderer->get_ReSTIR_PT_render_pass()->get_VRAM_usage();
					ImGui::Text("VRAM Usage: %.3fMB", last_VRAM_usage);

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					if (ImGui::SliderInt("M-cap", &render_settings.restir_pt_settings.m_cap, 0, 255, "%d", ImGuiSliderFlags_AlwaysClamp))
					{
						render_settings.restir_pt_settings.m_cap = std::max(0, render_settings.restir_pt_settings.m_cap);
						if (render_settings.accumulate)
							m_render_window->set_render_dirty(true);
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));

					ImGui::PushItemWidth(12 * ImGui::GetFontSize());
					draw_ReSTIR_PT_initial_candidates_panel();
					draw_ReSTIR_temporal_reuse_panel<ReSTIR_VARIANT_PT>(
						[&render_settings, this]()
						{
							if (ImGui::Checkbox("Do Temporal Reuse", &render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass))
								m_render_window->set_render_dirty(true);
						});
					draw_ReSTIR_spatial_reuse_panel<ReSTIR_VARIANT_PT>(
						[&render_settings, this]()
						{
							if (ImGui::Checkbox("Do spatial reuse", &render_settings.restir_pt_settings.common_spatial_pass.do_spatial_reuse_pass))
								m_render_window->set_render_dirty(true);
						});
					ImGui::PopItemWidth();

					if (ImGui::CollapsingHeader("Rejection Heuristics"))
					{
						ImGui::TreePush("ReSTIR PT - Rejection Heuristics Tree");

						draw_ReSTIR_neighbor_heuristics_panel<ReSTIR_VARIANT_PT>();

						ImGui::TreePop();
						ImGui::Dummy(ImVec2(0.0f, 20.0f));
					}

					draw_ReSTIR_bias_correction_panel<ReSTIR_VARIANT_PT>();
					draw_ReSTIR_PT_SPMIS_settings_panel();

					if (ImGui::CollapsingHeader("Debug"))
					{
						ImGui::TreePush("ReSTIR PT options tree");

						if (ImGui::Checkbox("Debug neighbor reuse positions", &render_settings.restir_pt_settings.common_spatial_pass.debug_neighbor_location))
							m_render_window->set_render_dirty(true);
						ImGuiRenderer::show_help_marker("If checked, neighbor in the spatial reuse pass will be hardcoded to always be "
														"15 pixels to the right, not in a circle. This makes spotting bias easier when debugging.");
						if (render_settings.restir_pt_settings.common_spatial_pass.debug_neighbor_location)
						{
							ImGui::TreePush("Debug neighbor location vertical tree");

							ImGui::Text("Debug reuse direction");
							bool reuse_direction_changed = false;
							reuse_direction_changed |= ImGui::RadioButton(
								"Horizontally", ((int*)&render_settings.restir_pt_settings.common_spatial_pass.debug_neighbor_location_direction), 0);
							ImGui::SameLine();
							reuse_direction_changed |= ImGui::RadioButton(
								"Vertically", ((int*)&render_settings.restir_pt_settings.common_spatial_pass.debug_neighbor_location_direction), 1);
							ImGui::SameLine();
							reuse_direction_changed |= ImGui::RadioButton(
								"Diagonally", ((int*)&render_settings.restir_pt_settings.common_spatial_pass.debug_neighbor_location_direction), 2);

							if (reuse_direction_changed)
								m_render_window->set_render_dirty(true);

							ImGui::TreePop();
						}

						ImGui::Dummy(ImVec2(0.0f, 20.0f));
						const char* debug_view_items[] = { "- No debug view",
														   "- Shade only initial candidates",
														   "- Final reservoir UCW",
														   "- Final reservoir target function",
														   "- Final reservoir weight sum",
														   "- Final reservoir M",
														   "- Per pixel reuse radius",
														   "- Valid directions percentage",
														   "- SPMIS cells",
														   "- Cell variance" };
						if (ImGui::Combo("Debug view", (int*)&render_settings.restir_pt_settings.debug_view, debug_view_items, IM_ARRAYSIZE(debug_view_items)))
						{
							int macro_value_before =
								global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED);
							if (render_settings.restir_pt_settings.debug_view == ReSTIRPTDebugView::PT_SHADE_ONLY_INITIAL_CANDIDATES)
								global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_PT_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED,
																	   KERNEL_OPTION_TRUE);
							else
								global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_PT_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED,
																	   KERNEL_OPTION_FALSE);
							int macro_value_after =
								global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_DEBUG_VIEW_SHADE_ONLY_INITIAL_CANDIDATES_ENABLED);

							bool macro_option_changed = macro_value_before != macro_value_after;
							if (macro_option_changed)
								m_renderer->recompile_kernels();

							m_render_window->set_render_dirty(true);
						}
						if (ImGui::SliderFloat("Debug view scale factor", &render_settings.restir_pt_settings.debug_view_scale_factor, 0.0f, 1.0f))
							m_render_window->set_render_dirty(true);

						ImGui::TreePop(); // Debug tree
					}

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop(); // ReSTIR GI Tree
				}
			}

			default:
				break;
			}

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_GI ||
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PT)
				draw_ReSTIR_PG_settings_panel();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_material_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Material/BSDF settings"))
	{
		ImGui::TreePush("Sampling Materials Tree");
		draw_principled_bsdf_energy_conservation();

		if (ImGui::CollapsingHeader("Principled BSDF diffuse lobe"))
		{
			ImGui::TreePush("Principled bsdf diffuse lobe tree");

			const char* items[] = { "- Lambertian", "- Oren-Nayar" };
			if (ImGui::Combo("Diffuse Lobe", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DIFFUSE_LOBE),
							 items, IM_ARRAYSIZE(items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			bool luminance_sampling = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_DIFFUSE_LUMINANCE);
			if (ImGui::Checkbox("Luminance sampling", &luminance_sampling))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_DIFFUSE_LUMINANCE,
													   luminance_sampling ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("If true, the diffuse lobe sampling probability will be additionally weighted by its luminance.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Principled BSDF metallic lobe"))
		{
			ImGui::TreePush("Principled bsdf metallic lobe tree");

			static bool sample_cosine_weighted =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_METALLIC_SAMPLE_COSINE_WEIGHTED);
			if (ImGui::Checkbox("Sample cosine weighted##metallic", &sample_cosine_weighted))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_METALLIC_SAMPLE_COSINE_WEIGHTED,
													   sample_cosine_weighted ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to sample the metallic lobe of the BSDF cosine-weighted (instead of VNDF sampling) or not.\n\n"
											"Cosine-weighted sampling is going to have massively lower variance at high roughnesses (> 0.7) and it is faster "
											"to sample on top of that.");

			if (sample_cosine_weighted)
			{
				ImGui::TreePush("Cosine-weighted metallic sampling settings tree");

				if (ImGui::SliderFloat("Roughness threshold", &render_data.bsdfs_data.metallic_sample_cosine_weighted_roughness_threshold, 0.0f, 1.0f))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("If the roughness of the metallic lobe of the Principled BSDF is higher or equal to this threshold, the "
												"metallic lobe will be sampled using cosine-weighted hemisphere sampling instead of GGX VNDF sampling.");

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Principled BSDF glossy lobe"))
		{
			ImGui::TreePush("Principled bsdf glossy lobe tree");

			bool sample_glossy_based_on_fresnel =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL);
			if (ImGui::Checkbox("Fresnel-based sampling##glossy", &sample_glossy_based_on_fresnel))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_GLOSSY_BASED_ON_FRESNEL,
													   sample_glossy_based_on_fresnel ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to sample the glossy/diffuse base layer of the BSDF based on the fresnel or not.\n\n"
											""
											"This means that the diffuse layer will be sampled more often at normal incidence since this is where "
											"the specular layer reflects close to no light.\n\n"
											""
											"At grazing angle however, where the specular layer reflects the most light(and so the diffuse layer "
											"below isn't reached by that light that is reflected by the specular layer), it is the specular layer "
											"that will be sampled more often.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Principled BSDF coat lobe"))
		{
			ImGui::TreePush("Principled bsdf coat lobe tree");

			bool sample_coat_based_on_fresnel = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL);
			if (ImGui::Checkbox("Fresnel-based sampling##coat", &sample_coat_based_on_fresnel))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_SAMPLE_COAT_BASED_ON_FRESNEL,
													   sample_coat_based_on_fresnel ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Same as the glossy layer fresnel-based sampling but for the coat layer.");

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("GGX");

		std::vector<const char*> ggx_sampling_items = { "- VNDF", "- VNDF Spherical Caps" };
		if (ImGui::Combo("GGX Sampling Method",
						 m_renderer->get_global_compiler_options()->get_raw_pointer_to_macro_value(
							 GPUKernelCompilerOptions::PRINCIPLED_BSDF_ANISOTROPIC_GGX_SAMPLE_FUNCTION),
						 ggx_sampling_items.data(), ggx_sampling_items.size()))
		{
			m_renderer->recompile_kernels();

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("How to sample the GGX NDF");

		std::vector<const char*> masking_shadowing_items = { "- Smith height-correlated", "- Smith height-uncorrelated" };
		if (ImGui::Combo("GGX Masking-Shadowing", (int*)&render_data.bsdfs_data.GGX_masking_shadowing, masking_shadowing_items.data(),
						 masking_shadowing_items.size()))
		{
			// Reloading all the energy compensation textures because if we change the masking-shadowing term,
			// the precomputed directional albedo isn't correct anymore
			m_renderer->load_GGX_energy_compensation_textures();
			m_renderer->load_GGX_glass_energy_compensation_textures();
			m_renderer->load_glossy_dielectric_energy_compensation_textures();

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Which masking-shadowing term to use with the GGX NDF.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Microfacet model regularization");
		draw_microfacet_model_regularization_tree();

		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_ris_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	bool ris_disabled = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_PT;
	ImGui::BeginDisabled(ris_disabled);
	if (ImGui::CollapsingHeader("RIS Settings"))
	{
		ImGui::TreePush("RIS Settings tree");

		if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR)
		{
			ImGui::Text("The mix of BSDF/light samples is controlled\n"
						"by the ReGIR settings.");
		}
		else
		{
			bool use_visibility_ris_target_function = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION);
			if (ImGui::Checkbox("Use visibility in RIS target function", &use_visibility_ris_target_function))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION,
													   use_visibility_ris_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
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
		}

		ImGui::TreePop();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
	ImGui::EndDisabled(); // ris_disabled
	if (ris_disabled)
		ImGuiRenderer::add_tooltip("The RIS light sampling settings are controlled by ReSTIR PT.");
}

void ImGuiSettingsWindow::draw_risltc_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("RISLTC Settings"))
	{
		ImGui::TreePush("RISLTC Settings tree");

		if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR)
		{
			ImGui::Text("The mix of BSDF/light samples is controlled\n"
						"by the ReGIR settings.");
		}
		else
		{
			// if (ImGui::SliderInt("RIS # of BSDF candidates", &render_settings.ris_settings.number_of_bsdf_candidates, 0, 16))
			//{
			//	// Clamping to 0
			//	render_settings.ris_settings.number_of_bsdf_candidates = std::max(0, render_settings.ris_settings.number_of_bsdf_candidates);

			//	m_render_window->set_render_dirty(true);
			//}

			if (ImGui::SliderInt("RIS # of light candidates", &render_settings.risltc_settings.number_of_light_candidates, 0, 32))
			{
				// Clamping to 0
				render_settings.risltc_settings.number_of_light_candidates = std::max(0, render_settings.risltc_settings.number_of_light_candidates);

				m_render_window->set_render_dirty(true);
			}
		}

		ImGui::TreePop();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

void ImGuiSettingsWindow::draw_ReSTIR_DI_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	std::shared_ptr<ReSTIRDIRenderPass> restir_di_render_pass = std::dynamic_pointer_cast<ReSTIRDIRenderPass>(
		m_renderer->get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_render_pass(ReSTIRDIRenderPass::RESTIR_DI_RENDER_PASS_NAME));

	ImGui::BeginDisabled(!restir_di_render_pass);
	if (ImGui::CollapsingHeader("ReSTIR DI Settings") && restir_di_render_pass)
	{
		ImGui::TreePush("ReSTIR DI Settings tree");

		ImGui::Text("VRAM Usage: %.3fMB", restir_di_render_pass->get_VRAM_usage());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		display_ReSTIR_DI_bias_status(global_kernel_options);

		if (ImGui::Checkbox("Use Final Visibility", &render_settings.restir_di_settings.do_final_shading_visibility))
			m_render_window->set_render_dirty(true);

		if (ImGui::SliderInt("M-cap", &render_settings.restir_di_settings.m_cap, 0, 64, "%d", ImGuiSliderFlags_AlwaysClamp))
		{
			render_settings.restir_di_settings.m_cap = std::max(0, render_settings.restir_di_settings.m_cap);
			if (render_settings.accumulate)
				m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("0 disables the M-cap");

		if (ImGui::CollapsingHeader("Rejection Heuristics"))
		{
			ImGui::TreePush("ReSTIR DI - Rejection Heuristics Tree");

			draw_ReSTIR_neighbor_heuristics_panel<ReSTIR_VARIANT_DI>();

			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		if (ImGui::CollapsingHeader("Initial Candidates Pass"))
		{
			ImGui::TreePush("ReSTIR DI - Initial Candidate Pass Tree");

			{
				bool use_initial_target_function_visibility =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY);
				if (ImGui::Checkbox("Use visibility in target function", &use_initial_target_function_visibility))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY,
														   use_initial_target_function_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Whether or not to use the visibility term in the target function used for "
												"resampling initial candidates");

				const bool bsdf_samples_disabled_regir =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_STRATEGY) == LSS_BASE_REGIR;
				const bool bsdf_samples_disabled = bsdf_samples_disabled_regir;
				if (bsdf_samples_disabled_regir)
					ImGuiRenderer::add_warning("BSDF samples are disabled in ReSTIR DI because they are controlled by "
											   "the ReGIR settings (use BSDF MIS in ReGIR for BSDF samples).");
				ImGui::BeginDisabled(bsdf_samples_disabled);
				if (ImGui::SliderInt("# of BSDF initial candidates", &render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates,
									 0, 16))
				{
					// Clamping to 0
					render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates =
						std::max(0, render_settings.restir_di_settings.initial_candidates.number_of_initial_bsdf_candidates);

					m_render_window->set_render_dirty(true);
				}
				ImGui::EndDisabled();

				if (ImGui::SliderInt("# of initial light candidates", &render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates,
									 0, 32))
				{
					// Clamping to 0
					render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates =
						std::max(0, render_settings.restir_di_settings.initial_candidates.number_of_initial_light_candidates);

					m_render_window->set_render_dirty(true);
				}

				ImGui::BeginDisabled(!m_renderer->has_envmap());
				if (ImGui::SliderFloat("Envmap candidate probability", &render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability,
									   0.0f, 1.0f))
				{
					render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability =
						hippt::clamp(0.0f, 1.0f, render_settings.restir_di_settings.initial_candidates.envmap_candidate_probability);

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
				bool do_visibility_reuse = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE);
				if (ImGui::Checkbox("Do visibility reuse", &do_visibility_reuse))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE,
														   do_visibility_reuse ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}
			}

			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		draw_ReSTIR_temporal_reuse_panel<ReSTIR_VARIANT_DI>(
			[this, &render_settings]()
			{
				if (ImGui::Checkbox("Do Temporal Reuse", &render_settings.restir_di_settings.common_temporal_pass.do_temporal_reuse_pass))
					m_render_window->set_render_dirty(true);
			});

		ImGui::PushItemWidth(12 * ImGui::GetFontSize());
		draw_ReSTIR_spatial_reuse_panel<ReSTIR_VARIANT_DI>(
			[&render_settings, this]()
			{
				if (ImGui::Checkbox("Do spatial reuse", &render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass))
					m_render_window->set_render_dirty(true);
			});
		ImGui::PopItemWidth();

		draw_ReSTIR_bias_correction_panel<ReSTIR_VARIANT_DI>();
		if (ImGui::CollapsingHeader("Debug"))
		{
			ImGui::TreePush("ReSTIR DI debug options tree");

			if (ImGui::Checkbox("Debug neighbor reuse positions", &render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("If checked, neighbor in the spatial reuse pass will be hardcoded to always be "
											"15 pixels to the right, not in a circle. This makes spotting bias easier when debugging.");
			if (render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location)
			{
				ImGui::TreePush("Debug neighbor location vertical tree");

				ImGui::Text("Debug reuse direction");
				bool reuse_direction_changed = false;
				reuse_direction_changed |=
					ImGui::RadioButton("Horizontally", ((int*)&render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location_direction), 0);
				ImGui::SameLine();
				reuse_direction_changed |=
					ImGui::RadioButton("Vertically", ((int*)&render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location_direction), 1);
				ImGui::SameLine();
				reuse_direction_changed |=
					ImGui::RadioButton("Diagonally", ((int*)&render_settings.restir_di_settings.common_spatial_pass.debug_neighbor_location_direction), 2);

				if (reuse_direction_changed)
					m_render_window->set_render_dirty(true);

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Later Bounces Sampling Strategy"))
		{
			ImGui::TreePush("Later Bounces tree");

			{
				const char* second_bounce_items[] = { "- Uniform one light", "- BSDF Sampling", "- MIS (1 Light + 1 BSDF)", "- RIS BDSF + Light candidates" };
				if (ImGui::Combo("Direct Lighting Strategy",
								 global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY),
								 second_bounce_items, IM_ARRAYSIZE(second_bounce_items)))
				{
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("What direct lighting strategy to use for bounces that come after the first one (camera ray hit) since ReSTIR "
												"DI only applies on the first bounce.");
				ImGui::Dummy(ImVec2(0.0f, 20.0f));

				switch (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_LATER_BOUNCES_SAMPLING_STRATEGY))
				{
				case RESTIR_DI_LATER_BOUNCES_UNIFORM_ONE_LIGHT:
					break;

				case RESTIR_DI_LATER_BOUNCES_MIS_LIGHT_BSDF:
					break;

				case RESTIR_DI_LATER_BOUNCES_RIS_BSDF_AND_LIGHT:
				{
					bool use_visibility_ris_target_function =
						global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION);
					if (ImGui::Checkbox("Use visibility in RIS target function", &use_visibility_ris_target_function))
					{
						global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION,
															   use_visibility_ris_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
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
		}

		ImGui::TreePop(); // ReSTIR DI Settings tree
	}
	ImGui::EndDisabled(); // !restir_di_render_pass
}

void ImGuiSettingsWindow::draw_ReSTIR_PG_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	ReSTIRPGSettings& restir_pg_settings							= render_settings.restir_pg_settings;
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
	std::shared_ptr<ReSTIRPGRenderPass> restir_pg_render_pass		= std::dynamic_pointer_cast<ReSTIRPGRenderPass>(
		m_renderer->get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_render_pass(ReSTIRPGRenderPass::RESTIR_PG_RENDER_PASS_NAME));

	if (ImGui::CollapsingHeader("ReSTIR PG"))
	{
		ImGui::TreePush("ReSTIR PG options tree");

		bool use_pg = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_ENABLE) == KERNEL_OPTION_TRUE;
		if (ImGui::Checkbox("Enable ReSTIR PG", &use_pg))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_PG_ENABLE, use_pg ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		if (ImGui::CollapsingHeader("VMF Mixture"))
		{
			ImGui::TreePush("ReSTIR VMF Mixture tree");

			static int distribution_component_count = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT);
			ImGui::SliderInt("Mixture component count", &distribution_component_count, 1, 8);

			if (distribution_component_count != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT))
			{
				ImGui::TreePush("ReSTIR PG distribution component count apply button");

				if (ImGui::Button("Apply##ReSTIR PG distribution component count"))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT, distribution_component_count);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Sampling"))
		{
			ImGui::TreePush("ReSTIR PG Sampling tree");

			if (ImGui::SliderFloat("BSDF sampling proba", &render_data.render_settings.restir_pg_settings.bsdf_sampling_probability, 0.0f, 1.0f, "%.3f",
								   ImGuiSliderFlags_AlwaysClamp))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("The probability of sampling the BSDF when sampling the next path vertex direction with ReSTIR PG.\n\n"
											"ReSTIR PG samples the next path vertex direction from a mixture of the BSDF and the path guiding distribution. "
											"This slider controls the balance between those two distributions.\n\n"
											"A value of 1.0f means that only the BSDF will be sampled, so no path guiding, while a value of 0.0f means that "
											"only the path guiding distribution will be sampled.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Hash grid"))
		{
			ImGui::TreePush("ReSTIR PG Hash grid tree");

			if (ImGui::SliderFloat("Grid cell target projected size", &restir_pg_settings.hash_grid_target_projected_size, 5, 25))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("The target screen-space size (in pixels) that a grid cell should occupy on the screen.\n"
											"This has the effect of making the grid cells larger in the distance so that the projected size stays "
											"approximately constant.");

			if (ImGui::SliderFloat("Grid cell minimum size", &restir_pg_settings.hash_grid_cell_min_size, 0.1, 0.5))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("The minimum size of a grid cell in world space units");

			static int linear_probing_steps = ReSTIRPGHashGridCollisionResolveSteps;
			ImGui::SliderInt("Collision resolution steps", &linear_probing_steps, 1, 32);
			if (linear_probing_steps != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_HASH_GRID_COLLISION_RESOLVE_STEPS))
			{
				ImGui::TreePush("ReSTIR PG linear probing steps apply button");
				if (ImGui::Button("Apply"))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::RESTIR_PG_HASH_GRID_COLLISION_RESOLVE_STEPS, linear_probing_steps);

					m_render_window->set_render_dirty(true);
					m_renderer->recompile_kernels();
				}
				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Debug"))
		{
			ImGui::TreePush("ReSTIR PG Debug tree");

			const char* debug_view_items[] = { "- No debug view", "- Grid cells", "- Distribution direction", "- Distribution sharpness",
											   "- Distribution weight" };
			if (ImGui::Combo("Debug view", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE),
							 debug_view_items, IM_ARRAYSIZE(debug_view_items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE) == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_DIRECTION ||
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE) == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_SHARPNESS ||
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE) == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_WEIGHT)
			{
				ImGui::TreePush("Distribution component direction tree");

				if (ImGui::SliderInt("Component index", &render_data.render_settings.restir_pg_settings.debug_distribution_component_direction_number, 0,
									 global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DISTRIBUTION_COMPONENT_COUNT) - 1))
					m_render_window->set_render_dirty(true);

				if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PG_DEBUG_MODE) == RESTIR_PG_DEBUG_DISTRIBUTION_COMPONENT_SHARPNESS)
				{
					if (ImGui::SliderFloat("Sharpness normalize factor", &render_data.render_settings.restir_pg_settings.debug_normalization_factor, 0.5f,
										   50.0f))
						m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Statistics"))
		{
			ImGui::TreePush("ReSTIR PG Statistics tree");

			ImGui::Text("VRAM Usage: %.3fMB", restir_pg_render_pass->get_VRAM_usage());

			float load_factor = restir_pg_render_pass->get_hash_grid_load_factor();
			ImGui::Text("Hash grid load factor: %.3f%%", load_factor * 100.0f);

			ImGui::TreePop(); // ReSTIR PG Statistics Tree
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::TreePop(); // ReSTIR PG Tree
	}
}

void ImGuiSettingsWindow::draw_ltc_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("LTC Settings"))
	{
		ImGui::TreePush("LTC settings tree");

		if (ImGui::SliderFloat("Maximum roughness", &render_data.bsdfs_data.ltcs_data.specular_ltc_maximum_roughness, 0.0f, 1.0f))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker("If the material lobe has a roughness higher than this value, then this lobe "
										"won't be included in the LTC sampling.\n\n"
										""
										"For example, for a diffuse / specular material with a specular roughness of 1.0f "
										"and a max roughness of 0.5f, the specular lobe will never be sampled by LTCs, only "
										"the diffuse lobe. This is to help with LTC sampling overhead on rough lobes that wouldn't really "
										"benefit from it.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_ReGIR_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
	std::shared_ptr<ReGIRRenderPass> regir_render_pass				= std::dynamic_pointer_cast<ReGIRRenderPass>(
		m_renderer->get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_render_pass(ReGIRRenderPass::REGIR_RENDER_PASS_NAME));

	ImGui::BeginDisabled(!regir_render_pass);
	if (ImGui::CollapsingHeader("ReGIR Settings") && regir_render_pass)
	{
		ImGui::TreePush("ReGIR settings tree");

		ImGui::SeparatorText("Grid stats (primary cells | secondary cells)");
		ImGui::Text("# of hash cells occupied: %u | %u", regir_render_pass->get_number_of_cells_alive(true),
					regir_render_pass->get_number_of_cells_alive(false));
		ImGui::Text("Hash cells capacity: %u | %u", regir_render_pass->get_total_number_of_cells_alive(true),
					regir_render_pass->get_total_number_of_cells_alive(false));
		ImGui::Text("Load factor: %.3f%% | %.3f%%", regir_render_pass->get_alive_cells_ratio(true) * 100.0f,
					regir_render_pass->get_alive_cells_ratio(false) * 100.0f);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("VRAM Usage: %.3fMB (avg. %.1fB per cell)", regir_render_pass->get_VRAM_usage_bytes() / 1000000.0f,
					regir_render_pass->get_VRAM_usage_bytes() /
						((float)regir_render_pass->get_total_number_of_cells_alive(true) + regir_render_pass->get_total_number_of_cells_alive(false)));
		ImGui::Text("VRAM Usage breakdown: ");
		std::vector<char> tooltip_buffer(2048);
		snprintf(tooltip_buffer.data(), 2048,
				 "Breakdown:\n"
				 "\t- Primary hit reservoirs: %.3fMB\n"
				 "\t\t- Base reservoirs: %.3fMB\n"
				 "\t\t- Spatial reuse reservoirs: %.3fMB\n"
				 "\t\t- Correlation reduction: %.3fMB\n"
				 "\t\t- Cell world data: %.3fMB\n"
				 "\t\t- Async compute: %.3fMB\n"
				 "\t\t- RIS pre-integration: %.3fMB\n"
				 "\t- Primary hit light distributions: %.3fMB\n\n"

				 "\t- Secondary hit reservoirs: %.3fMB\n"
				 "\t\t- Base reservoirs: %.3fMB\n"
				 "\t\t- Spatial reuse reservoirs: %.3fMB\n"
				 "\t\t- Cell world data: %.3fMB\n"
				 "\t\t- Async compute: %.3fMB\n"
				 "\t\t- RIS pre-integration: %.3fMB\n"
				 "\t- Secondary hit light distributions: %.3fMB",
				 regir_render_pass->get_reservoirs_VRAM_usage_bytes(true) / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_initial_grid_buffers(true).get_byte_size() / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_spatial_grid_buffers(true).get_byte_size() / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_correlation_reduction_buffer().get_byte_size() / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_hash_cell_data_soa(true).get_byte_size() / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_async_compute_staging_buffer(true).get_byte_size() / 1000000.0f,
				 (regir_render_pass->get_hash_grid_storage().get_non_canonical_factors(true).get_byte_size() +
				  regir_render_pass->get_hash_grid_storage().get_canonical_factors(true).get_byte_size()) /
					 1000000.0f,
				 regir_render_pass->get_light_distibutions_VRAM_usage_bytes(true) / 1000000.0f,

				 regir_render_pass->get_reservoirs_VRAM_usage_bytes(false) / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_initial_grid_buffers(false).get_byte_size() / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_spatial_grid_buffers(false).get_byte_size() / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_hash_cell_data_soa(false).get_byte_size() / 1000000.0f,
				 regir_render_pass->get_hash_grid_storage().get_async_compute_staging_buffer(false).get_byte_size() / 1000000.0f,
				 (regir_render_pass->get_hash_grid_storage().get_non_canonical_factors(false).get_byte_size() +
				  regir_render_pass->get_hash_grid_storage().get_canonical_factors(false).get_byte_size()) /
					 1000000.0f,
				 regir_render_pass->get_light_distibutions_VRAM_usage_bytes(false) / 1000000.0f);
		ImGuiRenderer::show_help_marker(tooltip_buffer.data());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ReGIRSettings& regir_settings = m_renderer->get_render_settings().regir_settings;

		if (ImGui::CollapsingHeader("Cell light distributions"))
		{
			ImGui::TreePush("Use per cell llight distributions");

			if (ImGui::Checkbox("Use cell light distributions", &regir_settings.use_per_cell_light_distributions))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_USE_PER_CELL_LIGHT_DISTRIBUTIONS,
													   regir_settings.use_per_cell_light_distributions ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("If true, the contribution of each emissive mesh of the scene will be precomputed "
											"at each cell of the hash grid to build a sampling distribution based on the contribution "
											"of the emissive meshes.\n\n"
											""
											"Those per-cell sampling distribution will then be used during the grid fill to provide higher "
											"quality initial light samples");

			static bool only_use_cell_light_distributions =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_SAMPLE_ONLY_LIGHT_DISTRIBUTIONS);
			if (ImGui::Checkbox("Only use cell light distributions", &only_use_cell_light_distributions))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_SAMPLE_ONLY_LIGHT_DISTRIBUTIONS,
													   only_use_cell_light_distributions ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("If true, ReGIR will not be used to shade points at path tracing time.Only the light distributions precomputed "
											"ahead of time will be used to compute NEE.");
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			if (regir_settings.use_per_cell_light_distributions)
			{
				const char* items[] = { "- Uniform sampling", "- Power sampling", "- Light tree ATS", "- SG light tree" };
				static int light_distributions_defensive_sampling_technique =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTIONS_CANONICAL_SAMPLING_TECHNIQUE);
				if (ImGui::Combo("Defensive sampling technique", &light_distributions_defensive_sampling_technique, items, IM_ARRAYSIZE(items)))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTIONS_CANONICAL_SAMPLING_TECHNIQUE,
														   light_distributions_defensive_sampling_technique);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				static bool use_representative_normal = ReGIR_GridFillCellDistributionsUseRepresentativeNormal;
				if (ImGui::Checkbox("Use representative normal", &use_representative_normal))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTIONS_USE_REPRESENTATIVE_NORMAL,
														   use_representative_normal ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Whether or not to use a repsentative normal when computing the contribution of an emissive "
												"mesh to the grid cell.This can help quickly reject backfacing lights and should "
												"probably be left enabled");
				static bool unbiased_nee_plus_plus = ReGIR_GridFillCellDistributionsUnbiasedNEEPlusPlus;
				if (ImGui::Checkbox("Unbiased NEE++", &unbiased_nee_plus_plus))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTION_UNBIASED_NEE_PLUS_PLUS,
														   unbiased_nee_plus_plus ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker(
					"If this is TRUE, NEE++ visibility estimation will be used in the grid fill target "
					"function for non - canonical reservoirs if grid cell light distributions are enabled.\n\n"
					""
					"If this is false, NEE++ won't be used in the target function with makes the grid fill "
					"quite a bit faster because fetching NEE++ for each non - canonical reservoir is a bit expensive.\n\n"
					""
					"With ReGIR spatial reuse enabled (and only if it is enabled) however, this is going to be biased but the bias is "
					"actually is very small so this is a worthy optimization imo.");
				static bool integrate_mesh = ReGIR_GridFillCellDistributionsIntegrateMesh;
				if (ImGui::Checkbox("Integrate mesh contribution", &integrate_mesh))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTION_INTEGRATE_MESH,
														   integrate_mesh ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("When computing the contribution of meshes to the grid cell point:\n\n"
												""
												"If this option is true, random points will be chosen on the emissive mesh and the "
												"contribution to the grid cell point of each of these points on the emissive mesh "
												"will be integrated to compute an estimate of the overall contribution of the "
												"emissive mesh to the grid cell.\n"
												"The number of random points drawn is equal to \"Integrate mesh sample count\".\n\n"
												""
												"If this option is false, the overall contribution of the mesh is going to be computed "
												"in one go using an approximate representative point for the whole as well as an average reprensetative "
												"normal. This is less precise than integrating over the mesh but way faster.");
				ImGui::BeginDisabled(!integrate_mesh);
				static int integrate_mesh_sample_count = ReGIR_GridFillCellDistributionsIntegrateMeshSampleCount;
				ImGui::SliderInt("Integrate mesh sample count", &integrate_mesh_sample_count, 1, 64);
				ImGuiRenderer::show_help_marker("How many random points to integrate the contribution of an emissive mesh over "
												"if \"Integrate mesh contribution\" is true");
				if (integrate_mesh_sample_count !=
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTION_INTEGRATE_MESH_SAMPLE_COUNT))
				{
					ImGui::TreePush("Integrate mesh sample count tree regir");

					if (ImGui::Button("Apply"))
					{
						global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTION_INTEGRATE_MESH_SAMPLE_COUNT,
															   integrate_mesh_sample_count);

						m_renderer->recompile_kernels();
						m_render_window->set_render_dirty(true);
					}

					ImGui::TreePop();
				}
				ImGui::EndDisabled();

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				static int cache_cells_list_distribution_canonical_samples_count = ReGIR_GridFillCellDistributionsCanonicalSampleCount;
				ImGui::SliderInt("Canonical samples count", &cache_cells_list_distribution_canonical_samples_count, 0, 16);
				ImGuiRenderer::show_help_marker("How many canonical samples(simple power sampling) to draw and combine with cell-light-distribution "
												"samples to guarantee unbiasedness.\n\n"
												""
												"1 guarantees unbiasedness. More than 1 reduces variance more effectively if the coverage of the "
												"cell-light-distribution is poor");
				if (cache_cells_list_distribution_canonical_samples_count !=
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTIONS_CANONICAL_SAMPLE_COUNT))
				{
					ImGui::TreePush("Canonical sample count cell light distribs regir");

					if (ImGui::Button("Apply"))
					{
						global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_CELL_DISTRIBUTIONS_CANONICAL_SAMPLE_COUNT,
															   cache_cells_list_distribution_canonical_samples_count);

						m_renderer->recompile_kernels();
						m_render_window->set_render_dirty(true);
					}

					ImGui::TreePop();
				}

				if (ImGui::SliderInt("Light distribution max. size", &regir_settings.light_distribution_maximum_size, 1,
									 hippt::min(65535u, render_data.buffers.emissive_meshes_data.alias_table_count), "%d", ImGuiSliderFlags_AlwaysClamp))
					m_render_window->set_render_dirty(true);

				if (ImGui::SliderFloat("Light distribution target radiance", &regir_render_pass->get_light_distribution_target_incoming_energy(), 0.0f, 100.0f,
									   "%.3f", ImGuiSliderFlags_AlwaysClamp))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("Percentage of the total incoming energy that we should keep in each light distribution of "
												"each cell at *most* (roughly)\n\n"
												""
												"The light distribution will only contain as many emissive meshes as necessary such that the "
												"distribution covers covers that percentage of the total incoming energy to the grid cell.\n\n"
												""
												"This is \"rounded up\" so if 40% of the total incoming radiance is required by this parameter but "
												"we have to choose between (for example):\n\n"
												""
												" - 5 meshes in the distribution = 38% of the energy covered\n"
												" - 6 meshes in the distribution = 51% of the energy covered\n\n"
												""
												"Then the light distribution will cover 6 meshes.");
				ImGui::TreePush("VRAM saving ReGIR");
				ImGui::Text("VRAM saving 1st hits: %f%%", regir_render_pass->get_light_distributions_compaction_VRAM_savings(true));
				ImGui::Text("VRAM saving 2nd hits: %f%%", regir_render_pass->get_light_distributions_compaction_VRAM_savings(false));
				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				ImGui::TreePop(); // VRAM saving tree
			} // regir_settings.use_per_cell_light_distributions

			ImGui::TreePop(); // Per cell light distributions tree
		}

		if (ImGui::CollapsingHeader("Grid fill pass"))
		{
			ImGui::TreePush("ReGIR grid build tree");

			ImGui::SeparatorText("Primary hits grid cells");
			if (ImGui::SliderInt("Light samples per reservoir", &regir_settings.grid_fill_settings_primary_hits.light_sample_count_per_cell_reservoir, 0, 32))
				m_render_window->set_render_dirty(true);
			if (ImGui::SliderInt("Non-canonical reservoirs per grid cell",
								 regir_settings.grid_fill_settings_primary_hits.get_non_canonical_reservoir_count_per_cell_ptr(), 1, 64))
			{
				int& non_cano_reservoir_count = *regir_settings.grid_fill_settings_primary_hits.get_non_canonical_reservoir_count_per_cell_ptr();

				// Minimum of 1
				non_cano_reservoir_count = hippt::max(1, non_cano_reservoir_count);

				m_render_window->set_render_dirty(true);
			}
			if (ImGui::SliderInt("Canonical reservoirs per grid cell",
								 regir_settings.grid_fill_settings_primary_hits.get_canonical_reservoir_count_per_cell_ptr(), 1, 16))
			{
				int& cano_reservoir_count = *regir_settings.grid_fill_settings_primary_hits.get_canonical_reservoir_count_per_cell_ptr();

				// Minimum of 1
				cano_reservoir_count = hippt::max(1, cano_reservoir_count);

				m_render_window->set_render_dirty(true);
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			ImGui::SeparatorText("Secondary hits grid cells");
			if (ImGui::SliderInt("Light samples per reservoir##secondary",
								 &regir_settings.grid_fill_settings_secondary_hits.light_sample_count_per_cell_reservoir, 0, 64))
				m_render_window->set_render_dirty(true);
			if (ImGui::SliderInt("Non-canonical reservoirs per grid cell##secondary",
								 regir_settings.grid_fill_settings_secondary_hits.get_non_canonical_reservoir_count_per_cell_ptr(), 1, 64))
				m_render_window->set_render_dirty(true);
			if (ImGui::SliderInt("Canonical reservoirs per grid cell##secondary",
								 regir_settings.grid_fill_settings_secondary_hits.get_canonical_reservoir_count_per_cell_ptr(), 1, 16))
				m_render_window->set_render_dirty(true);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Common to primary and secondary grid cells");
			static bool visibility_grid_fill_target_function = ReGIR_GridFillTargetFunctionVisibility;
			if (ImGui::Checkbox("Use visibility in target function", &visibility_grid_fill_target_function))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_VISIBILITY,
													   visibility_grid_fill_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker(
				"Whether or not to use a visibility term in the target function used to resample the reservoirs of the grid cells.\n\n"
				""
				"Probably too expensive to be efficient.");

			static bool nee_plus_plus_visibility_grid_fill_target_function = ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation;
			if (ImGui::Checkbox("Use NEE++ visibility in target function", &nee_plus_plus_visibility_grid_fill_target_function))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_NEE_PLUS_PLUS_VISIBILITY_ESTIMATION,
													   nee_plus_plus_visibility_grid_fill_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to estimate the visibility probability of samples with NEE++ during the grid fill.");
			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_FALSE &&
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_NEE_PLUS_PLUS_VISIBILITY_ESTIMATION) ==
					KERNEL_OPTION_TRUE)
			{
				ImGuiRenderer::add_warning("NEE++ needs to be enabled to use it in ReGIR");

				ImGui::TreePush("Use NEE++ ReGIR Tree");
				use_nee_plus_plus_checkbox("Enable NEE++");
				ImGuiRenderer::show_help_marker("Shortcut for enabling for enabling NEE++");
				ImGui::TreePop();
			}

			static bool cosine_term_grid_fill_target_function = ReGIR_GridFillTargetFunctionCosineTerm;
			if (ImGui::Checkbox("Use cosine term in target function", &cosine_term_grid_fill_target_function))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM,
													   cosine_term_grid_fill_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker(
				"Whether or not to use a the cosine term between the direction to the light sample and the "
				"representative normal of the grid cell in the target function used to resample the reservoirs of the grid cells.\n\n"
				""
				"This has no effect is representative points are not being used.");

			static bool cosine_term_light_source_grid_fill_target_function = ReGIR_GridFillTargetFunctionCosineTermLightSource;
			if (ImGui::Checkbox("Use cosine term light source in target function", &cosine_term_light_source_grid_fill_target_function))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_TARGET_FUNCTION_COSINE_TERM_LIGHT_SOURCE,
													   cosine_term_light_source_grid_fill_target_function ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Takes the cosine term at the light source (i.e. the cosine term of the geometry term) "
											"into account when evaluating the target function during grid fill");

			static bool bsdf_grid_fill_target_function_first_hits = ReGIR_GridFillPrimaryHitsTargetFunctionBSDF;
			if (ImGui::Checkbox("Use BSDF in target function (1st hits)", &bsdf_grid_fill_target_function_first_hits))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_PRIMARY_HITS_TARGET_FUNCTION_BSDF,
													   bsdf_grid_fill_target_function_first_hits ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to include the BSDF in the target function used for the resampling of the initial candidates\n"
											"for the grid fill.\n\n"
											""
											"Helps a lot on glossy surfaces.\n\n"
											""
											"This option applies to primary hits only and should generally be set to true for better sampling.");

			static bool bsdf_grid_fill_target_function_secondary_hits = ReGIR_GridFillSecondaryHitsTargetFunctionBSDF;
			if (ImGui::Checkbox("Use BSDF in target function (2nd hits)", &bsdf_grid_fill_target_function_secondary_hits))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_SECONDARY_HITS_TARGET_FUNCTION_BSDF,
													   bsdf_grid_fill_target_function_secondary_hits ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Same as the option for the primary hits but only applies to secondary hits.\n\n"
											""
											"This option should be set to false in general as we cannot guess in advance what the view direction is going\n"
											"to be at secondary hits(since they can come from anywhere when the rays bounce around the scene) and thus we\n"
											"cannot properly evaluate the BRDF for sampling lights.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (ImGui::SliderInt("Frame skip (1st hits)", &regir_settings.frame_skip_primary_hit_grid, 0, 8))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("How many frames to skip before running the grid fill and spatial reuse passes again.\n\n"
											""
											"A value of 1 for example means that the grid fill and spatial reuse will be ran at frame 0 "
											"but not at frame 1. And ran at frame 2 but not at frame 3. ...\n\n"
											""
											"This amortizes the overhead of ReGIR grid fill / spatial reuse by using the fact that each cell "
											"contains many reservoirs so the same cell can be used multiple times before all reservoirs have been used "
											"and new samples are necessary.\n\n"
											""
											"The frame skips can be different for filling the primary or secondary hits grid.");
			if (ImGui::SliderInt("Frame skip (2nd hits)", &regir_settings.frame_skip_secondary_hit_grid, 0, 8))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("How many frames to skip before running the grid fill and spatial reuse passes again.\n\n"
											""
											"A value of 1 for example means that the grid fill and spatial reuse will be ran at frame 0 "
											"but not at frame 1. And ran at frame 2 but not at frame 3. ...\n\n"
											""
											"This amortizes the overhead of ReGIR grid fill / spatial reuse by using the fact that each cell "
											"contains many reservoirs so the same cell can be used multiple times before all reservoirs have been used "
											"and new samples are necessary.\n\n"
											""
											"The frame skips can be different for filling the primary or secondary hits grid.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Sampling strategies");

			const char* items_base_strategy[]				   = { "- Uniform sampling", "- Power sampling", "- Light tree ATS (Conty & Kulla 2018)",
																   "- SG light tree (Tokuyoshi et al. 2024)" };
			const char* tooltips_base_strategy_non_canonical[] = {
				"All lights are sampled uniformly",

				"Lights are sampled proportionally to their power",

				"Lights are sampled using a light hierarchy with orientation bounds as proposed in the paper of Conty & Kulla, 2018.",

				"Lights are sampled using a light hierarchy of spherical gaussian lights as proposed in the paper of Tokuyoshi et al., 2024.",
			};
			const char* tooltips_base_strategy_canonical[] = {
				"All lights are sampled uniformly",

				"Lights are sampled proportionally to their power",

				"Lights are sampled using a light hierarchy **without** orientation bounds as proposed in the paper of Conty & Kulla, 2018.\n"
				"Canonical candidates do not use the orientation bounds to remain conservative.",

				"Lights are sampled using a light hierarchy of spherical gaussian lights as proposed in the paper of Tokuyoshi et al., 2024.\n"
				"Orientation is not considered to remain conservative",
			};

			bool base_light_sampling_strategy_disabled =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_USE_PER_CELL_LIGHT_DISTRIBUTIONS) == KERNEL_OPTION_TRUE;
			if (base_light_sampling_strategy_disabled)
				ImGuiRenderer::add_warning("Cell light distributions are being used as the main non-canonical sampling strategy");
			ImGui::BeginDisabled(base_light_sampling_strategy_disabled);
			if (ImGuiRenderer::ComboWithTooltips(
					"Non canonical",
					global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY_NON_CANONICAL),
					items_base_strategy, IM_ARRAYSIZE(items_base_strategy), tooltips_base_strategy_non_canonical))
			{
				// Will recompute the alias table if necessary
				m_renderer->recompute_emissives_sampling_data_structure();

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGui::EndDisabled();

			static int canonical_candidates_strategy = 0;
			if (ImGuiRenderer::ComboWithTooltips(
					"Canonical",
					global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::REGIR_GRID_FILL_LIGHT_SAMPLING_BASE_STRATEGY_CANONICAL),
					items_base_strategy, IM_ARRAYSIZE(items_base_strategy), tooltips_base_strategy_canonical))
			{
				// Will recompute the alias table if necessary
				m_renderer->recompute_emissives_sampling_data_structure();

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		if (ImGui::CollapsingHeader("Spatial reuse"))
		{
			ImGui::TreePush("ReGIR spatial reuse tree");

			if (ImGui::Checkbox("Do spatial reuse", &regir_settings.spatial_reuse.do_spatial_reuse))
				m_render_window->set_render_dirty(true);
			if (ImGui::Checkbox("Do coalesced spatial reuse", &regir_settings.spatial_reuse.do_coalesced_spatial_reuse))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("If true, the same random seed will be used by all grid cells during the spatial reuse for a given frame\n."
											"This has the effect of coalescing neighbors memory accesses which improves performance");
			if (ImGui::SliderInt("Spatial reuse pass count", &regir_settings.spatial_reuse.spatial_reuse_pass_count, 1, 4))
				m_render_window->set_render_dirty(true);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (ImGui::SliderInt("Neighbor reuse count", &regir_settings.spatial_reuse.spatial_neighbor_count, 0, 16))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("How many cells around the center cell to reuse from.");

			if (ImGui::SliderInt("Reuse per neighbor count", &regir_settings.spatial_reuse.reuse_per_neighbor_count, 1, 16))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("How many reservoirs to reuse per neighbor cell.");

			if (ImGui::SliderInt("Retries per neighbor", &regir_settings.spatial_reuse.retries_per_neighbor, 1, 16))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker(" When picking a random cell in the neighborhood for reuse, if that "
											"cell is out of the grid or if that cell is not alive etc..., we're "
											"going to retry another cell this many times.\n"
											"This improves the chances that we're actually going to have a good "
											"neighbor to reuse from --> more reuse --> less variance.");

			if (ImGui::SliderInt("Reuse radius", &regir_settings.spatial_reuse.spatial_reuse_radius, 1, 3))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("Radius in cell in which to reuse around the center cell.\n"
											"A radius of 1 means that we're going to reuse in the 3x3 cube around the center cell, givins us 26 neighbors");

			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		if (ImGui::CollapsingHeader("Shading"))
		{
			ImGui::TreePush("ReGIR shading tree");

			if (ImGui::SliderInt("Neighbors resampled", &regir_settings.shading_settings.number_of_neighbors, 1, 8))
				m_render_window->set_render_dirty(true);
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			static bool use_vis_shading_resampling = ReGIR_ShadingResamplingTargetFunctionVisibility;
			if (ImGui::Checkbox("Use visibility in target function", &use_vis_shading_resampling))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_TARGET_FUNCTION_VISIBILITY,
													   use_vis_shading_resampling ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to use a shadow ray in the target function when "
											"shading a point at path tracing time. This reduces visibility noise.");

			ImGui::BeginDisabled(use_vis_shading_resampling);
			static bool use_nee_plus_plus_vis_shading_resampling = ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility;
			if (ImGui::Checkbox("Use NEE++ visibility estimation in target function", &use_nee_plus_plus_vis_shading_resampling))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_TARGET_FUNCTION_NEE_PLUS_PLUS_VISIBILITY,
													   use_nee_plus_plus_vis_shading_resampling ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to use NEE++ to estimate the visibility probability of the reservoir being resampled during "
											"shading such that reservoirs that are likely to be occluded will have a lower resampling probability");
			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_FALSE &&
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_TARGET_FUNCTION_NEE_PLUS_PLUS_VISIBILITY) ==
					KERNEL_OPTION_TRUE)
			{
				ImGuiRenderer::add_warning("NEE++ needs to be enabled to use it in ReGIR");

				ImGui::TreePush("Use NEE++ ReGIR Tree");
				use_nee_plus_plus_checkbox("Enable NEE++");
				ImGuiRenderer::show_help_marker("Shortcut for enabling for enabling NEE++");
				ImGui::TreePop();
			}
			ImGui::EndDisabled();

			static bool include_canonical = ReGIR_ShadingResamplingIncludeCanonicalCandidates;
			if (ImGui::Checkbox("Include canonical candidates", &include_canonical))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_INCLUDE_CANONICAL_CANDIDATES,
													   include_canonical ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to include canonical candidates at all during the shading.\n\n"
											""
											"Disabling this is biased but useful basically only for debug purposes.");
			if (include_canonical)
			{
				ImGui::TreePush("Do canonical candidatres tree ReGIR");

				static bool regir_shading_light_tree_canonical_candidates =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_CANONICAL_CANDIDATES_LIGHT_TREE_ATS);
				if (ImGui::Checkbox("Use light tree", &regir_shading_light_tree_canonical_candidates))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_CANONICAL_CANDIDATES_LIGHT_TREE_ATS,
														   regir_shading_light_tree_canonical_candidates ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("If true, uses the ATS light tree for sampling canonical candidates during shading resampling to avoid"
												"bias instead of using the grid fill canonical candidates.");

				ImGui::TreePop();
			}

			static bool do_resampling_bsdf_mis = ReGIR_ShadingResamplingDoBSDFMIS;
			if (ImGui::Checkbox("Do BSDF MIS during resampling", &do_resampling_bsdf_mis))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_DO_BSDF_MIS,
													   do_resampling_bsdf_mis ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to incorporate BSDF samples with MIS during shading resampling.");
			if (do_resampling_bsdf_mis)
			{
				ImGui::TreePush("BSDF MIS reGIR simplified ray");

				static bool do_resampling_bsdf_mis_simplified_ray = ReGIR_ShadingResamplingDoBSDFMISSimplifiedRay;
				if (ImGui::Checkbox("Simplified ray", &do_resampling_bsdf_mis_simplified_ray))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESAMPLING_DO_BSDF_MIS_SIMPLIFIED_RAY,
														   do_resampling_bsdf_mis_simplified_ray ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("If this is true, BSDF sample rays will be traced in a BVH that contains only the lights of the scene, "
												"not the rest of the geometry.This can increase variance but make the traces way way faster to the point "
												"where BSDF MIS rays are almost free.");

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Jittering");
			if (ImGui::Checkbox("Do cell jittering (1st hits)", &regir_settings.shading_settings.do_cell_jittering_first_hits))
				m_render_window->set_render_dirty(true);
			if (ImGui::Checkbox("Do cell jittering (2nd hits)", &regir_settings.shading_settings.do_cell_jittering_secondary_hits))
				m_render_window->set_render_dirty(true);
			static bool jitter_canonical = ReGIR_ShadingResamplingJitterCanonicalCandidates;
			if (ImGui::Checkbox("Jitter canonical candidates", &jitter_canonical))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESMAPLING_JITTER_CANONICAL_CANDIDATES,
													   jitter_canonical ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to jitter canonical candidates during the shading resampling.\n"
											"This reduces grid artifacts but increases variance.");
			ImGui::BeginDisabled(!regir_settings.shading_settings.do_cell_jittering_first_hits &&
								 !regir_settings.shading_settings.do_cell_jittering_secondary_hits &&
								 !global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESMAPLING_JITTER_CANONICAL_CANDIDATES));
			if (ImGui::SliderFloat("Jitter radius", &regir_settings.shading_settings.jittering_radius, 0.5f, 2.0f))
				m_render_window->set_render_dirty(true);
			ImGui::BeginDisabled(!global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_RESMAPLING_JITTER_CANONICAL_CANDIDATES));
			if (ImGui::SliderFloat("Jitter radius can. candidates", &regir_settings.shading_settings.jittering_radius_canonical_candidates, 0.0f, 1.0f))
				m_render_window->set_render_dirty(true);
			ImGui::EndDisabled();

			static int jitter_retries = ReGIR_ShadingJitterRetries;
			ImGui::SliderInt("Jitter retries", &jitter_retries, 1, 16);
			ImGuiRenderer::show_help_marker("If using jittering, how many retries to perform to find a good neighbor at shading time ?\n\n"
											""
											"This is because with jittering, our jittered position may end up outside of the grid "
											"or in an empty cell, in which case we want to retry with a differently jittered position "
											"to try and find a good neighbor");
			if (jitter_retries != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_JITTER_RETRIES))
			{
				ImGui::TreePush("Apply jitter retries regir");

				if (ImGui::Button("Apply"))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_SHADING_JITTER_RETRIES, jitter_retries);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}
			static bool jitter_in_tangent_plane = ReGIR_JitterInTangentPlane;
			if (ImGui::Checkbox("Jitter in tangent plane", &jitter_in_tangent_plane))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_JITTER_IN_TANGENT_PLANE,
													   jitter_in_tangent_plane ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("If true, shading point jittering will only jitter the point in the tangent plane of the surface.\n\n"
											""
											"This helps reducing bad jittering(jittering which moves the shading point outside of the scene's surface) "
											"and reduces variance becaues we're getting more useful neighbors out of the jitters instead of having to rely "
											"on jittering retries to find a valid neighbor.");
			ImGui::EndDisabled();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Correlation reduction");
			if (ImGui::Checkbox("Correlation reduction", &regir_settings.correlation_reduction.do_correlation_reduction))
				m_render_window->set_render_dirty(true);
			if (ImGui::SliderInt("Correlation reduction factor", &regir_settings.correlation_reduction.correlation_reduction_factor, 1, 8))
				m_render_window->set_render_dirty(true);

			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		if (ImGui::CollapsingHeader("Hash grid"))
		{
			ImGui::TreePush("ReGIR hash grid tree");

			ImGui::SeparatorText("Grid size");
			static bool constant_grid_cell_size = ReGIR_HashGridConstantGridCellSize;
			if (ImGui::Checkbox("Constant grid cell size", &constant_grid_cell_size))
			{
				regir_settings.hash_grid.m_grid_cell_min_size = constant_grid_cell_size ? 0.75f : 0.2f;

				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_CONSTANT_GRID_CELL_SIZE,
													   constant_grid_cell_size ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to use constant grid cell size for the hash grid.\n\n"
											""
											"If this is false, the grid cell size will increase(cells gets bigger) the further away "
											"from the camera.This can help with performance and the number of resident cells "
											"in the hash grid but it tends to hurt quality because of the reduced grid cell resolution.");

			ImGui::BeginDisabled(constant_grid_cell_size);
			if (ImGui::SliderFloat("Grid cell target projected size", &regir_settings.hash_grid.m_grid_cell_target_projected_size, 5, 25))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("The target screen-space size (in pixels) that a grid cell should occupy on the screen.\n"
											"This has the effect of making the grid cells larger in the distance so that the projected size stays "
											"approximately constant.");
			ImGui::EndDisabled();

			std::string grid_cell_size_text = constant_grid_cell_size ? "Grid cell size" : "Grid cell minimum size";
			if (ImGui::SliderFloat(grid_cell_size_text.c_str(), &regir_settings.hash_grid.m_grid_cell_min_size, 0.1, 0.5))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("The minimum size of a grid cell in world space units");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Hashing");
			static bool fuzzy_grid_cells = ReGIR_HashGridHashFuzzyGridCells;
			if (ImGui::Checkbox("Fuzzy grid cells", &fuzzy_grid_cells))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_HASH_FUZZY_GRID_CELLS,
													   fuzzy_grid_cells ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("If true, cell borders will be a bit fuzzy to help with grid cell artifacts.");
			if (ImGui::SliderFloat("Fuzzy grid cell strength", &regir_settings.hash_grid.fuzzy_grid_cells_strength, 0.0f, 1.0f))
				m_render_window->set_render_dirty(true);

			static bool fuzzy_normals = ReGIR_HashGridHashFuzzyNormals;
			if (ImGui::Checkbox("Fuzzy normals", &fuzzy_normals))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_HASH_FUZZY_NORMALS,
													   fuzzy_normals ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("If true, the normals used in the hash function of the hash grid will be jittered a little bit "
											"to help hide grid artifacts caused by the discretization of normals.");
			if (ImGui::SliderFloat("Fuzzy normals strength", &regir_settings.hash_grid.fuzzy_normals_strength, 0.0f, 1.0f))
				m_render_window->set_render_dirty(true);

			static bool include_normals_in_hash = ReGIR_HashGridHashSurfaceNormal;
			if (ImGui::Checkbox("Use surface normal in hash", &include_normals_in_hash))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_HASH_SURFACE_NORMAL,
													   include_normals_in_hash ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker("Whether or not to use the surface normal in the hash function of the hash grid. "
											"Increases quality but also significantly increases memory usage");
			if (include_normals_in_hash)
			{
				ImGui::TreePush("ReGIR surface normal discretization tree");

				static int normal_discretization_precision = ReGIR_HashGridHashSurfaceNormalResolutionPrimaryHits;
				ImGui::SliderInt("Precision 1st hits", &normal_discretization_precision, 2, 4);
				ImGuiRenderer::show_help_marker("Higher values mean more precision for the discretization but also more computational and VRAM usage for "
												"filling the grid as well as a potentially decreased spatial reuse efficiency.");

				if (normal_discretization_precision !=
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_HASH_SURFACE_NORMAL_RESOLUTION_PRIMARY_HITS))
				{
					ImGui::TreePush("Apply button ReGIR normal discretization");

					if (ImGui::Button("Apply"))
					{
						global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_HASH_SURFACE_NORMAL_RESOLUTION_PRIMARY_HITS,
															   normal_discretization_precision);

						m_renderer->recompile_kernels();
						m_render_window->set_render_dirty(true);
					}

					ImGui::TreePop();
				}

				ImGui::TreePop();
			}

			if (include_normals_in_hash)
			{
				ImGui::TreePush("ReGIR surface normal discretization 2nd hits tree");

				static int normal_discretization_precision = ReGIR_HashGridHashSurfaceNormalResolutionSecondaryHits;
				ImGui::SliderInt("Precision 2nd hits", &normal_discretization_precision, 2, 4);
				ImGuiRenderer::show_help_marker("Higher values mean more precision for the discretization but also more computational and VRAM usage for "
												"filling the grid as well as a potentially decreased spatial reuse efficiency.");

				if (normal_discretization_precision !=
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_HASH_SURFACE_NORMAL_RESOLUTION_SECONDARY_HITS))
				{
					ImGui::TreePush("Apply button ReGIR normal discretization");

					if (ImGui::Button("Apply"))
					{
						global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_HASH_SURFACE_NORMAL_RESOLUTION_SECONDARY_HITS,
															   normal_discretization_precision);

						m_renderer->recompile_kernels();
						m_render_window->set_render_dirty(true);
					}

					ImGui::TreePop();
				}

				ImGui::TreePop();
			}

			static bool adaptive_roughness_grid_precision = ReGIR_HashGridAdaptiveRoughnessGridPrecision;
			if (ImGui::Checkbox("Adaptive roughness grid precision", &adaptive_roughness_grid_precision))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_ADAPTIVE_ROUGHNESS_GRID_PRECISION,
													   adaptive_roughness_grid_precision ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Collision resolution");
			const char* items[]	   = { "- Linear probing", "- Rehashing" };
			const char* tooltips[] = {
				"If a collision is found, look up the next index in the hash\n"
				"table and see if that location is empty. If not empty, continue\n"
				"looking at the next location up to the maximum number of steps' times.",

				"If a collision is found, hash the current cell index to get the\n"
				"new candidate location. Continue doing so until an empty location\n"
				"is found or the maximum number of steps is exceeded.",
			};
			if (ImGuiRenderer::ComboWithTooltips(
					"Mode", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_COLLISION_RESOLUTION_MODE), items,
					IM_ARRAYSIZE(items), tooltips))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			static int linear_probing_steps = ReGIR_HashGridCollisionResolutionMaxSteps;
			ImGui::SliderInt("Max. steps", &linear_probing_steps, 1, 32);
			if (linear_probing_steps != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_COLLISION_RESOLUTION_MAX_STEPS))
			{
				ImGui::TreePush("ReGIR linear probing steps apply button");
				if (ImGui::Button("Apply"))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::REGIR_HASH_GRID_COLLISION_RESOLUTION_MAX_STEPS, linear_probing_steps);

					m_render_window->set_render_dirty(true);
					m_renderer->recompile_kernels();
				}
				ImGui::TreePop();
			}

			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		if (ImGui::CollapsingHeader("Performance"))
		{
			ImGui::TreePush("ReGIR Performance tree");

			if (ImGui::Checkbox("Do asynchronous compute", &regir_settings.do_asynchronous_compute))
				m_render_window->set_render_dirty(true);

			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		if (ImGui::CollapsingHeader("Debug"))
		{
			ImGui::TreePush("ReGIR Settings debug tree");

			int regir_debug_mode = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::REGIR_DEBUG_MODE);
			const char* items[]	 = { "- No debug", "- Grid cells", "- Cell representative points", "- Cell representative normals", "- Sampling fallback" };
			if (ImGui::Combo("Debug mode", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::REGIR_DEBUG_MODE), items,
							 IM_ARRAYSIZE(items)))
			{
				if (regir_debug_mode == REGIR_DEBUG_MODE_REPRESENTATIVE_POINTS)
					// Auto settings this to arbitrary 0.1f to help with visualization
					regir_settings.debug_view_scale_factor = 0.1f;

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			else if (regir_debug_mode == REGIR_DEBUG_MODE_REPRESENTATIVE_POINTS)
			{
				if (ImGui::SliderFloat("Distance to point", &regir_settings.debug_view_scale_factor, 0.0f, 1.0f))
					m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
	ImGui::EndDisabled(); // ImGui::BeginDisabled(!regir_render_pass);

	bool light_tree_used_by_regir = m_renderer->get_light_tree_ats_sampling_data_structure().is_needed(
		m_renderer->get_emissive_mesh_count(), m_renderer->get_active_render_graph().get_compiler_options());
	if (light_tree_used_by_regir)
		draw_light_tree_ATS_settings_panel();
}

void ImGuiSettingsWindow::draw_light_tree_ATS_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	LightTreeATSBuilderOptions& build_options						= m_renderer->get_light_tree_ats_build_options();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Light tree ATS settings"))
	{
		ImGui::TreePush("Light tree ATS settings tree");

		ImGui::Text("VRAM Usage: %.3fMB", m_renderer->get_light_tree_ats_sampling_data_structure().get_VRAM_usage_bytes() / 1000000.0f);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::SeparatorText("Build");

		std::vector<const char*> build_mode_items = { "- Midpoint", "- Binned + cost function" };
		if (ImGui::Combo("Build split function", &build_options.build_split_method, build_mode_items.data(), build_mode_items.size()))
		{
			m_renderer->recompute_emissives_sampling_data_structure();

			m_render_window->set_render_dirty(true);
		}
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		switch (build_options.build_split_method)
		{
		case LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED:
		{
			ImGui::TreePush("Bin count tree");

			static int current_bin_count = build_options.bin_count;
			ImGui::SliderInt("Bin count", &current_bin_count, 2, 96);

			if (current_bin_count != build_options.bin_count)
			{
				ImGui::TreePush("Apply button light tree bin count");

				if (ImGui::Button("Apply"))
				{
					current_bin_count		= hippt::clamp(2, 2000000000, current_bin_count);
					build_options.bin_count = current_bin_count;

					m_renderer->recompute_emissives_sampling_data_structure();

					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			std::vector<const char*> cost_function_items = { "- SAH", "- SAOH" };
			if (ImGui::Combo("Cost function", &build_options.cost_function, cost_function_items.data(), cost_function_items.size()))
			{
				m_renderer->recompute_emissives_sampling_data_structure();

				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();

			break;
		}
		}

		static int previous_triangles_per_leaf = build_options.max_triangles_per_leaf;
		ImGui::SliderInt("Max triangles per leaf", &previous_triangles_per_leaf, 1, 32);
		if (previous_triangles_per_leaf != build_options.max_triangles_per_leaf)
		{
			ImGui::TreePush("Apply button triangles per leaf light tree");

			if (ImGui::Button("Apply"))
			{
				previous_triangles_per_leaf			 = hippt::clamp(1, 2000000000, previous_triangles_per_leaf);
				build_options.max_triangles_per_leaf = previous_triangles_per_leaf;

				m_renderer->recompute_emissives_sampling_data_structure();

				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}

		static bool stop_splitting_if_cost_not_worth_it = build_options.stop_splitting_if_cost_not_worth_it;
		ImGui::Checkbox("Stop splitting if cost not worth it", &stop_splitting_if_cost_not_worth_it);
		if (stop_splitting_if_cost_not_worth_it != build_options.stop_splitting_if_cost_not_worth_it)
		{
			ImGui::TreePush("Apply button stop splitting if cost not worth it");

			if (ImGui::Button("Apply"))
			{
				build_options.stop_splitting_if_cost_not_worth_it = stop_splitting_if_cost_not_worth_it;

				m_renderer->recompute_emissives_sampling_data_structure();
				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Sampling");

		static bool do_splitting = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_DO_SPLITTING);
		if (ImGui::Checkbox("Do adaptive splitting", &do_splitting))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_DO_SPLITTING,
												   do_splitting ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		if (do_splitting)
		{
			ImGui::TreePush("Split variance threshold ATS tree");

			if (ImGui::SliderFloat("Split threshold", &render_data.light_tree_ats.settings.light_tree_ats_splitting_variance, 0.0f, 1.0f, "%.3f",
								   ImGuiSliderFlags_AlwaysClamp))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker("User defined split threshold proposed in the paper of Conty & Kulla 2018."
											" The higher this threshold, the more nodes will be split. This parameter is quite scene dependent unfortunately.");

			static int splitting_max_light_samples_count =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES);
			ImGui::SliderInt("Max light samples", &splitting_max_light_samples_count, 1, 16);
			ImGuiRenderer::show_help_marker("If splitting is enabled, how many light samples, at most, per shading point is allowed.\n"
											"Higher values result in higher quality but at a higher performance cost.");

			if (splitting_max_light_samples_count !=
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES))
			{
				ImGui::TreePush("Apply button tree splitting max light sample count");

				if (ImGui::Button("Apply"))
				{
					splitting_max_light_samples_count = hippt::clamp(1, 2000000000, splitting_max_light_samples_count);

					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_SPLITTING_MAX_LIGHT_SAMPLES,
														   splitting_max_light_samples_count);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Importance function");
		static bool importance_function_use_orientation =
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_IMPORTANCE_FUNCTION_USE_ORIENTATION);
		if (ImGui::Checkbox("Use orientation", &importance_function_use_orientation))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_IMPORTANCE_FUNCTION_USE_ORIENTATION,
												   importance_function_use_orientation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_light_tree_SG_settings_panel()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	LightTreeATSBuilderOptions& build_options						= m_renderer->get_light_tree_sg_build_options();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Light tree SG settings"))
	{
		ImGui::TreePush("Light tree SG settings tree");

		ImGui::Text("VRAM Usage: %.3fMB", m_renderer->get_light_tree_sg_sampling_data_structure().get_VRAM_usage_bytes() / 1000000.0f);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::SeparatorText("Build");
		switch (build_options.build_split_method)
		{
		case LIGHT_TREE_BUILD_OPTION_SPLIT_BINNED:
		{
			static int current_bin_count = build_options.bin_count;
			ImGui::SliderInt("Bin count", &current_bin_count, 2, 96);

			if (current_bin_count != build_options.bin_count)
			{
				ImGui::TreePush("Apply button light tree bin count");

				if (ImGui::Button("Apply"))
				{
					current_bin_count		= hippt::clamp(2, 2000000000, current_bin_count);
					build_options.bin_count = current_bin_count;

					m_renderer->recompute_emissives_sampling_data_structure();

					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			break;
		}
		}

		static int previous_triangles_per_leaf = build_options.max_triangles_per_leaf;
		ImGui::SliderInt("Max triangles per leaf", &previous_triangles_per_leaf, 1, 32);
		if (previous_triangles_per_leaf != build_options.max_triangles_per_leaf)
		{
			ImGui::TreePush("Apply button triangles per leaf light tree");

			if (ImGui::Button("Apply"))
			{
				previous_triangles_per_leaf			 = hippt::clamp(1, 2000000000, previous_triangles_per_leaf);
				build_options.max_triangles_per_leaf = previous_triangles_per_leaf;

				m_renderer->recompute_emissives_sampling_data_structure();

				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}

		static bool stop_splitting_if_cost_not_worth_it = build_options.stop_splitting_if_cost_not_worth_it;
		ImGui::Checkbox("Stop splitting if cost not worth it", &stop_splitting_if_cost_not_worth_it);
		if (stop_splitting_if_cost_not_worth_it != build_options.stop_splitting_if_cost_not_worth_it)
		{
			ImGui::TreePush("Apply button stop splitting if cost not worth it");

			if (ImGui::Button("Apply"))
			{
				build_options.stop_splitting_if_cost_not_worth_it = stop_splitting_if_cost_not_worth_it;

				m_renderer->recompute_emissives_sampling_data_structure();
				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Sampling");

		static bool do_splitting = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_DO_SPLITTING);
		ImGui::BeginDisabled(do_splitting);

		static bool use_tree_cut = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_TREE_CUT);
		if (ImGui::Checkbox("Use tree cut sampling", &use_tree_cut))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_TREE_CUT,
												   use_tree_cut ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("When enabled without adaptive splitting, samples are selected from the precomputed SG tree cut using weighted "
										"reservoir sampling. Changes require kernel recompilation.");

		static int current_tree_cut_size = m_renderer->get_light_tree_sg_sampling_data_structure().get_tree_cut_size();
		ImGui::InputInt("Tree cut size", &current_tree_cut_size, 1, 16);
		ImGuiRenderer::show_help_marker(
			"Number of nodes in the SG light-tree frontier, expanded breadth first and stored for sampling. Changes require rebuilding the light-tree data.");
		if (current_tree_cut_size != m_renderer->get_light_tree_sg_sampling_data_structure().get_tree_cut_size())
		{
			ImGui::TreePush("Apply button tree cut size");

			if (ImGui::Button("Apply"))
			{
				m_renderer->get_light_tree_sg_sampling_data_structure().set_tree_cut_size(current_tree_cut_size);

				m_renderer->recompute_emissives_sampling_data_structure();
				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}
		ImGui::EndDisabled();

		static int current_spatial_lobe_count = m_renderer->get_light_tree_sg_sampling_data_structure().get_spatial_lobe_count();
		ImGui::SliderInt("Spatial lobes per node", &current_spatial_lobe_count, 1, LIGHT_TREE_SG_MAX_SPATIAL_LOBES);
		if (current_spatial_lobe_count != m_renderer->get_light_tree_sg_sampling_data_structure().get_spatial_lobe_count())
		{
			ImGui::TreePush("Apply button spatial lobe count");

			if (ImGui::Button("Apply"))
			{
				current_spatial_lobe_count = hippt::clamp(1, LIGHT_TREE_SG_MAX_SPATIAL_LOBES, current_spatial_lobe_count);
				m_renderer->get_light_tree_sg_sampling_data_structure().set_spatial_lobe_count(current_spatial_lobe_count);

				m_renderer->recompute_emissives_sampling_data_structure();
				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}

		if (ImGui::Checkbox("Do adaptive splitting", &do_splitting))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_DO_SPLITTING,
												   do_splitting ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		if (do_splitting)
		{
			ImGui::TreePush("SG light tree adaptive splitting tree");

			static bool use_new_splitting_model = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_NEW_SPLITTING_MODEL);
			if (ImGui::Checkbox("Use new splitting model", &use_new_splitting_model))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_NEW_SPLITTING_MODEL,
													   use_new_splitting_model ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			if (!use_new_splitting_model)
			{
				ImGui::TreePush("Split variance threshold SG tree");

				if (ImGui::SliderFloat("Split threshold", &render_data.light_tree_sg.settings.light_tree_sg_splitting_variance, 0.0f, 1.0f, "%.3f",
									   ImGuiSliderFlags_AlwaysClamp))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker(
					"User defined split threshold proposed in the paper of Conty & Kulla 2018."
					" The higher this threshold, the more nodes will be split. This parameter is quite scene dependent unfortunately.");

				ImGui::TreePop();
			}
			else
			{
				ImGui::TreePush("SG tree new splitting model tree");

				bool split_first_candidate =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_NEW_SPLITTING_MODEL_ALWAYS_SPLIT_FIRST_CANDIDATE);
				if (ImGui::Checkbox("Always split first candidate", &split_first_candidate))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_NEW_SPLITTING_MODEL_ALWAYS_SPLIT_FIRST_CANDIDATE,
														   split_first_candidate ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			static int splitting_max_light_samples_count =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES);
			ImGui::SliderInt("Max light samples", &splitting_max_light_samples_count, 1, 16);
			ImGuiRenderer::show_help_marker("If splitting is enabled, how many light samples, at most, per shading point is allowed.\n"
											"Higher values result in higher quality but at a higher performance cost.");

			if (splitting_max_light_samples_count !=
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES))
			{
				ImGui::TreePush("Apply button tree splitting max light sample count");

				if (ImGui::Button("Apply"))
				{
					splitting_max_light_samples_count = hippt::clamp(1, 2000000000, splitting_max_light_samples_count);

					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_SPLITTING_MAX_LIGHT_SAMPLES,
														   splitting_max_light_samples_count);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			ImGui::TreePop();
		}

		static bool importance_function_do_specular =
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_SG_DO_SPECULAR_IMPORTANCE);
		if (ImGui::Checkbox("Do specular", &importance_function_do_specular))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_ATS_SG_DO_SPECULAR_IMPORTANCE,
												   importance_function_do_specular ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Illumination aware distributions");
		static bool use_illumination_aware_distributions =
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_ILLUMINATION_AWARE_DISTRIBUTIONS);
		if (ImGui::Checkbox("Use illumination aware distributions", &use_illumination_aware_distributions))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::LIGHT_TREE_SG_USE_ILLUMINATION_AWARE_DISTRIBUTIONS,
												   use_illumination_aware_distributions ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		if (use_illumination_aware_distributions)
		{
			std::shared_ptr<IlluminationAwareKDTreeRenderPass> illumination_aware_kd_tree_render_pass =
				m_renderer->get_illumination_aware_kd_tree_render_pass();

			std::size_t vram_usage_bytes = illumination_aware_kd_tree_render_pass ? illumination_aware_kd_tree_render_pass->get_vram_usage_bytes() : 0;
			std::size_t node_capacity = illumination_aware_kd_tree_render_pass ? illumination_aware_kd_tree_render_pass->get_current_node_buffer_capacity() : 0;
			std::size_t occupied_nodes = illumination_aware_kd_tree_render_pass ? illumination_aware_kd_tree_render_pass->get_current_node_count() : 0;
			std::size_t guiding_node_count =
				illumination_aware_kd_tree_render_pass ? illumination_aware_kd_tree_render_pass->get_current_guiding_node_count() : 0;

			ImGui::Text("VRAM Usage: %.3fMB", vram_usage_bytes / 1000000.0f);
			ImGui::Text("  Occupied nodes: %zu / %zu (%.2f%%)", occupied_nodes, node_capacity,
						node_capacity > 0 ? (occupied_nodes * 100.0 / node_capacity) : 0.0);
			ImGui::Text("  Guiding nodes count: %zu", guiding_node_count);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (illumination_aware_kd_tree_render_pass)
			{
				static int training_sample_buffer_capacity = illumination_aware_kd_tree_render_pass->get_training_sample_buffer_capacity();
				ImGui::InputInt("Training sample buffer capacity", &training_sample_buffer_capacity);

				if (training_sample_buffer_capacity != illumination_aware_kd_tree_render_pass->get_training_sample_buffer_capacity())
				{
					ImGui::TreePush("Apply button illumination-aware KD-tree training sample buffer capacity");

					if (ImGui::Button("Apply"))
					{
						illumination_aware_kd_tree_render_pass->get_training_sample_buffer_capacity() = training_sample_buffer_capacity;
						illumination_aware_kd_tree_render_pass->mark_buffers_need_reallocation();

						m_render_window->set_render_dirty(true);
					}

					ImGui::TreePop();
				}
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			static int maximum_lookahead_depth =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT);
			ImGui::SliderInt("Maximum lookahead depth", &maximum_lookahead_depth, 0, 10);
			if (maximum_lookahead_depth !=
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT))
			{
				ImGui::TreePush("Illumination-aware KD-tree maximum lookahead depth apply button");

				if (ImGui::Button("Apply##Illumination-aware KD-tree maximum lookahead depth"))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_MAXIMUM_LOOKAHEAD_LEVEL_COUNT,
														   maximum_lookahead_depth);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			if (illumination_aware_kd_tree_render_pass)
			{
				if (ImGui::SliderInt("Split iterations per SPP", &illumination_aware_kd_tree_render_pass->get_split_iterations_per_SPP(), 1, 8))
					m_render_window->set_render_dirty(true);

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				ImGui::Text("Splitting mode");
				bool splitting_mode_changed = false;
				splitting_mode_changed |= ImGui::RadioButton("Sample count only", ((int*)&illumination_aware_kd_tree_render_pass->get_subdivision_mode()), 0);
				splitting_mode_changed |= ImGui::RadioButton("Mean radiance only", ((int*)&illumination_aware_kd_tree_render_pass->get_subdivision_mode()), 1);
				splitting_mode_changed |= ImGui::RadioButton("Mean direction only", ((int*)&illumination_aware_kd_tree_render_pass->get_subdivision_mode()), 2);
				splitting_mode_changed |= ImGui::RadioButton("Full model", ((int*)&illumination_aware_kd_tree_render_pass->get_subdivision_mode()), 3);
				if (splitting_mode_changed)
					m_render_window->set_render_dirty(true);
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			const char* debug_view_items[] = { "- No debug", "- KD tree leaves solid", "- KD tree leaves outlines",
											   "- KD tree leaves outlines and lookaheads" };
			if (ImGui::Combo("Debug view",
							 global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::ILLUMINATION_AWARE_KD_TREE_DEBUG_MODE),
							 debug_view_items, IM_ARRAYSIZE(debug_view_items)))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			if (illumination_aware_kd_tree_render_pass)
				ImGui::Checkbox("Freeze tree", &illumination_aware_kd_tree_render_pass->get_frozen_tree());
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

template <int ReSTIRVariant>
void ImGuiSettingsWindow::draw_ReSTIR_neighbor_heuristics_panel()
{
	HIPRTRenderSettings& render_settings  = m_renderer->get_render_settings();
	ReSTIRCommonSettings& common_settings = [&render_settings]
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return std::ref(render_settings.restir_di_settings);
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return std::ref(render_settings.restir_gi_settings);
		else
			return std::ref(render_settings.restir_pt_settings);
	}();

	using GIOrPTSettingsType = std::conditional_t<ReSTIRVariant == ReSTIR_VARIANT_GI, ReSTIRGISettings, ReSTIRPTSettings>;
	GIOrPTSettingsType* gi_or_pt_settings;
	if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
		gi_or_pt_settings = &render_settings.restir_gi_settings;
	else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
		gi_or_pt_settings = &render_settings.restir_pt_settings;
	else
		gi_or_pt_settings = nullptr;

	static bool use_heuristics_at_all				= true;
	static bool use_normal_heuristic_backup			= common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic;
	static bool use_plane_distance_heuristic_backup = common_settings.neighbor_similarity_settings.use_plane_distance_heuristic;
	static bool use_roughness_heuristic_backup		= common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic;

	// For ReSTIR GI only
	static bool use_neighbor_sample_point_roughness_heuristic_backup = gi_or_pt_settings->use_neighbor_sample_point_roughness_heuristic;
	static bool use_jacobian_heuristic_backup						 = gi_or_pt_settings->use_jacobian_rejection_heuristic;

	if (ImGui::Checkbox("Use Heuristics for neighbor rejection", &use_heuristics_at_all))
	{
		if (!use_heuristics_at_all)
		{
			// Saving the usage of the heuristics for later restoration
			use_normal_heuristic_backup			= common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic;
			use_plane_distance_heuristic_backup = common_settings.neighbor_similarity_settings.use_plane_distance_heuristic;
			use_roughness_heuristic_backup		= common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic;

			common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic	= false;
			common_settings.neighbor_similarity_settings.use_plane_distance_heuristic		= false;
			common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = false;

			if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI || ReSTIRVariant == ReSTIR_VARIANT_PT)
			{
				use_jacobian_heuristic_backup						 = gi_or_pt_settings->use_jacobian_rejection_heuristic;
				use_neighbor_sample_point_roughness_heuristic_backup = gi_or_pt_settings->use_neighbor_sample_point_roughness_heuristic;

				gi_or_pt_settings->use_jacobian_rejection_heuristic				 = false;
				gi_or_pt_settings->use_neighbor_sample_point_roughness_heuristic = false;
			}
		}
		else
		{
			// Restoring heuristics usage to their backup values
			common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic	= use_normal_heuristic_backup;
			common_settings.neighbor_similarity_settings.use_plane_distance_heuristic		= use_plane_distance_heuristic_backup;
			common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic = use_roughness_heuristic_backup;

			if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI || ReSTIRVariant == ReSTIR_VARIANT_PT)
			{
				gi_or_pt_settings->use_jacobian_rejection_heuristic				 = use_jacobian_heuristic_backup;
				gi_or_pt_settings->use_neighbor_sample_point_roughness_heuristic = use_neighbor_sample_point_roughness_heuristic_backup;
			}
		}

		m_render_window->set_render_dirty(true);
	}
	ImGuiRenderer::show_help_marker("Using heuristics to reject neighbor that are too dissimilar (in "
									"terms of normal orientation/roughnes/... to the pixel doing the resampling "
									"can help reduce variance. It also reduces bias but never removes it "
									"completely, it just makes it less obvious.");

	if (use_heuristics_at_all)
	{
		ImGui::TreePush("ReSTIR Heursitics Tree");
		ImGui::Dummy(ImVec2(0.0f, 10.0f));

		{
			if (ImGui::Checkbox("Use normal similarity heuristic", &common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic))
				m_render_window->set_render_dirty(true);

			ImGui::TreePush("Normal similarity heuristic tree");
			if (ImGui::Checkbox("Use geometric normals", &common_settings.neighbor_similarity_settings.reject_using_geometric_normals))
				m_render_window->set_render_dirty(true);

			if (common_settings.neighbor_similarity_settings.use_normal_similarity_heuristic)
			{
				if (ImGui::SliderFloat("Angle threshold", &common_settings.neighbor_similarity_settings.normal_similarity_angle_degrees, 0.1f, 90.0f,
									   "%.3f deg", ImGuiSliderFlags_AlwaysClamp))
				{
					common_settings.neighbor_similarity_settings.normal_similarity_angle_precomp =
						std::cos(common_settings.neighbor_similarity_settings.normal_similarity_angle_degrees * M_PI / 180.0f);

					m_render_window->set_render_dirty(true);
				}
			}
			ImGui::TreePop();
		}
		ImGui::Dummy(ImVec2(0.0f, 10.0f));

		{
			if (ImGui::Checkbox("Use plane distance heuristic", &common_settings.neighbor_similarity_settings.use_plane_distance_heuristic))
				m_render_window->set_render_dirty(true);

			ImGui::TreePush("Plane distance heuristic tree");
			if (common_settings.neighbor_similarity_settings.use_plane_distance_heuristic)
				if (ImGui::SliderFloat("Distance threshold", &common_settings.neighbor_similarity_settings.plane_distance_threshold, 0.0f, 1.0f))
					m_render_window->set_render_dirty(true);
			ImGui::TreePop();
		}
		ImGui::Dummy(ImVec2(0.0f, 10.0f));

		{
			if (ImGui::Checkbox("Use roughness heuristic", &common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic))
				m_render_window->set_render_dirty(true);

			ImGui::TreePush("Roughness heuristic tree");
			if (common_settings.neighbor_similarity_settings.use_roughness_similarity_heuristic)
				if (ImGui::SliderFloat("Roughness threshold", &common_settings.neighbor_similarity_settings.roughness_similarity_threshold, 0.0f, 1.0f, "%.3f",
									   ImGuiSliderFlags_AlwaysClamp))
					m_render_window->set_render_dirty(true);
			ImGui::TreePop();
		}

		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI || ReSTIRVariant == ReSTIR_VARIANT_PT)
		{
			ImGui::Dummy(ImVec2(0.0f, 10.0f));
			if (ImGui::Checkbox("Use jacobian heuristic", &gi_or_pt_settings->use_jacobian_rejection_heuristic))
				m_render_window->set_render_dirty(true);

			ImGui::TreePush("Jacobian heuristic tree");
			if (gi_or_pt_settings->use_jacobian_rejection_heuristic)
			{
				if (ImGui::SliderFloat("Jacobian threshold", gi_or_pt_settings->get_jacobian_heuristic_threshold_pointer(), 5.0f, 100.0f))
				{
					gi_or_pt_settings->set_jacobian_heuristic_threshold(hippt::max(1.001f, gi_or_pt_settings->get_jacobian_heuristic_threshold()));
					m_render_window->set_render_dirty(true);
				}
			}
			ImGui::TreePop();
			ImGui::Dummy(ImVec2(0.0f, 10.0f));

			if (ImGui::Checkbox("Use sample point roughness heuristic", &gi_or_pt_settings->use_neighbor_sample_point_roughness_heuristic))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker(
				"If the roughness of the neighbor's sample point is lower than this threshold, the neighbor "
				"won't be reused\n"
				"If the neighbor's sample point's roughness is higher than the threshold, it can be reused.\n"
				"This is pretty much necessary to avoid \"bias\" (although this isn't stricly bias, more like extremely "
				"high variance) when the primary hit (visible point) is on a rough surface and the secondary hit (sample point) is on a "
				"specular surface: a rough primary hit bouncing into a window / mirror for example.");

			ImGui::TreePush("Sample point roughness heuristic tree");
			if (gi_or_pt_settings->use_neighbor_sample_point_roughness_heuristic)
				if (ImGui::SliderFloat("Min. neighbor roughness", &gi_or_pt_settings->neighbor_sample_point_roughness_threshold, 0.0f, 1.0f))
					m_render_window->set_render_dirty(true);
			ImGui::TreePop();
		}
		// ReSTIR DI Heursitics Tree
		ImGui::TreePop();
	}
}

template <int ReSTIRVariant>
void ImGuiSettingsWindow::draw_ReSTIR_temporal_reuse_panel(std::function<void(void)> draw_before_panel)
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();
	ReSTIRCommonTemporalPassSettings& restir_common_temporal_settings =
		ReSTIRVariant == ReSTIR_VARIANT_DI	 ? m_renderer->get_render_settings().restir_di_settings.common_temporal_pass
		: ReSTIRVariant == ReSTIR_VARIANT_GI ? m_renderer->get_render_settings().restir_gi_settings.common_temporal_pass
											 : m_renderer->get_render_settings().restir_pt_settings.common_temporal_pass;

	if (ImGui::CollapsingHeader("Temporal Reuse Pass"))
	{
		ImGui::PushID(&restir_common_temporal_settings);
		ImGui::TreePush("ReSTIR - Temporal Reuse Pass Tree");
		{
			draw_before_panel();

			if (restir_common_temporal_settings.do_temporal_reuse_pass)
			{
				// Same line as "Do Temporal Reuse"
				ImGui::SameLine();
				if (ImGui::Button("Reset Temporal Reservoirs"))
				{
					if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
						m_renderer->get_ReSTIR_DI_render_pass()->request_temporal_bufffers_clear();
					else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
						m_renderer->get_ReSTIR_GI_render_pass()->request_temporal_bufffers_clear();
					else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
						m_renderer->get_ReSTIR_PT_render_pass()->request_temporal_bufffers_clear();

					m_render_window->set_render_dirty(true);
				}

				bool last_frame_g_buffer_needed = true;
				last_frame_g_buffer_needed &= !render_settings.accumulate;
				last_frame_g_buffer_needed &= restir_common_temporal_settings.do_temporal_reuse_pass;

				if (ImGui::SliderInt("Max temporal neighbor search count", &restir_common_temporal_settings.max_neighbor_search_count, 0, 16))
				{
					// Clamping
					restir_common_temporal_settings.max_neighbor_search_count = std::max(0, restir_common_temporal_settings.max_neighbor_search_count);

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("Temporal neighbor search radius", &restir_common_temporal_settings.neighbor_search_radius, 0, 16))
				{
					// Clamping
					restir_common_temporal_settings.neighbor_search_radius = std::max(0, restir_common_temporal_settings.neighbor_search_radius);

					m_render_window->set_render_dirty(true);
				}

				ImGuiRenderer::show_help_marker("If true, the back-projected position of the current pixel (temporal neighbor position) will be shuffled"
												" to add temporal variations.");

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				int& m_cap = ReSTIRVariant == ReSTIR_VARIANT_DI	  ? m_renderer->get_render_settings().restir_di_settings.m_cap
							 : ReSTIRVariant == ReSTIR_VARIANT_GI ? m_renderer->get_render_settings().restir_gi_settings.m_cap
																  : m_renderer->get_render_settings().restir_pt_settings.m_cap;
				if (ImGui::SliderInt("M-cap", &m_cap, 0, 255, "%d", ImGuiSliderFlags_AlwaysClamp))
				{
					m_cap = std::max(0, m_cap);
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

template <int ReSTIRVariant>
void ImGuiSettingsWindow::draw_ReSTIR_spatial_reuse_panel(std::function<void(void)> draw_before_panel)
{
	HIPRTRenderSettings& render_settings			 = m_renderer->get_render_settings();
	ReSTIRCommonSpatialPassSettings& restir_settings = [&render_settings]()
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return std::ref(render_settings.restir_di_settings.common_spatial_pass);
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
			return std::ref(render_settings.restir_gi_settings.common_spatial_pass);
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
			return std::ref(render_settings.restir_pt_settings.common_spatial_pass);
	}();

	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Spatial Reuse Pass"))
	{
		ImGui::PushID(&restir_settings);
		ImGui::TreePush("ReSTIR - Spatial Reuse Pass Tree");
		{
			draw_before_panel();

			if (restir_settings.do_spatial_reuse_pass)
			{
				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				bool use_spatial_target_function_visibility;
				if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
					use_spatial_target_function_visibility =
						global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY);
				else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
					use_spatial_target_function_visibility =
						global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY);
				else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
					use_spatial_target_function_visibility =
						global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_SPATIAL_TARGET_FUNCTION_VISIBILITY);
				if (ImGui::Checkbox("Use visibility in target function", &use_spatial_target_function_visibility))
				{
					global_kernel_options->set_macro_value(
						ReSTIRVariant == ReSTIR_VARIANT_DI	 ? GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY
						: ReSTIRVariant == ReSTIR_VARIANT_GI ? GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_TARGET_FUNCTION_VISIBILITY
															 : GPUKernelCompilerOptions::RESTIR_PT_SPATIAL_TARGET_FUNCTION_VISIBILITY,
						use_spatial_target_function_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();

					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Whether or not to use the visibility term in the target function used for "
												"resampling spatial neighbors.");

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				if (ImGui::SliderInt("Spatial reuse pass count", &restir_settings.number_of_passes, 1, 8))
				{
					// Clamping
					restir_settings.number_of_passes = std::max(1, restir_settings.number_of_passes);

					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("Neighbor reuse count", &restir_settings.reuse_neighbor_count, 1, 16))
					m_render_window->set_render_dirty(true);

				bool using_spmis = compute_ReSTIR_PT_using_SPMIS_boolean<ReSTIRVariant>();

				ImGui::BeginDisabled(using_spmis);
				std::string spatial_reuse_radius_text = restir_settings.use_adaptive_directional_spatial_reuse ? "Max reuse radius (px)" : "Reuse radius (px)";
				if (ImGui::SliderInt(spatial_reuse_radius_text.c_str(), &restir_settings.reuse_radius, 0, 64))
				{
					restir_settings.auto_reuse_radius = false;

					if (!restir_settings.debug_neighbor_location)
						// Clamping if not debugging (we do allow negative values when debugging)
						restir_settings.reuse_radius = std::max(0, restir_settings.reuse_radius);

					m_render_window->set_render_dirty(true);
				}
				if (using_spmis)
					ImGuiRenderer::add_tooltip("Disabled because using SPMIS, use the radius settings in \"SPMIS Settings\"");
				ImGui::SameLine();
				if (ImGui::Checkbox("Auto", &restir_settings.auto_reuse_radius))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("Automatically determines the spatial reuse radius (or maximum spatial reuse radius if using "
												"\"adaptive-directional spatial reuse\") to use based on the render resolution.");

				if (ImGui::CollapsingHeader("Directional spatial reuse"))
				{
					ImGui::TreePush("Directional spatial reuse tree");

					if (!render_settings.accumulate)
					{
						ImGuiRenderer::add_warning("Disabled because not accumulating");
						ImGui::Dummy(ImVec2(0.0f, 20.0f));
					}

					ImGui::BeginDisabled(!render_settings.accumulate);

					if (ImGui::Checkbox("Use adaptive-directional spatial reuse", &restir_settings.use_adaptive_directional_spatial_reuse))
						m_render_window->set_render_dirty(true);
					ImGuiRenderer::show_help_marker("Precomputes the best per-pixel spatial reuse radius to use as "
													"well as the sectors in the spatial reuse disk (split in 32 sectors) that should be used for reuse.\n\n"
													""
													"This increases the spatial reuse \"hit rate\" (i.e. the number of neighbors that are not rejected by "
													"G-Buffer heuristics) and thus increases convergence speed.\n\n"
													""
													"Has no effect if not accumulating.");

					if (restir_settings.use_adaptive_directional_spatial_reuse)
					{
						if (ImGui::SliderInt("Minimum reuse radius (px)", &restir_settings.minimum_per_pixel_reuse_radius, 0, restir_settings.reuse_radius))
							m_render_window->set_render_dirty(true);
						ImGuiRenderer::show_help_marker("The minimum radius that will be used per pixel when the optimal per-pixel spatial reuse "
														"radius is computed by \"adaptive-directional spatial reuse\"");
					}

					ImGui::EndDisabled();

					ImGui::Dummy(ImVec2(0.0f, 20.0f));
					ImGui::TreePop();
				}
				if (using_spmis)
					ImGuiRenderer::add_tooltip("Not compatible with SPMIS");
				ImGui::EndDisabled();

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
			}
		}

		ImGui::TreePop();
		ImGui::PopID();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

template <int ReSTIRVariant>
void ImGuiSettingsWindow::draw_ReSTIR_bias_correction_panel()
{
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_data().render_settings;

	ReSTIRCommonSettings& common_settings = [&render_settings]()
	{
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return std::ref(render_settings.restir_di_settings);
		else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
			return std::ref(render_settings.restir_gi_settings);
		else
			return std::ref(render_settings.restir_pt_settings);
	}();

	if (ImGui::CollapsingHeader("MIS Weights"))
	{
		ImGui::PushID(&common_settings);
		ImGui::TreePush("MIS Weights tree ReSTIR");

		std::vector<const char*> mis_weights_types_items = { "- 1/M (Biased)",
															 "- 1/Z",
															 "- MIS-like",
															 "- Generalized balance heuristic",
															 "- Pairwise MIS",
															 "- Pairwise MIS defensive",
															 "- Pairwise symmetric ratio",
															 "- Pairwise asymmetric ratio",
															 "- Stochastic pairwise MIS",
															 "- Stochastic pairwise MIS defensive" };

		std::vector<const char*> tooltips = {
			"Very simple biased weights as described in the 2020 ReSTIR DI paper(Eq. 6).\n"
			"Those weights are biased because they do not account for cases where "
			"we resample a sample that couldn't have been produced by some neighbors.\n"
			"The bias shows up as darkening, mostly at object boundaries. In GRIS vocabulary, "
			"this type of weights can be seen as confidence weights alone c_i / sum(c_j).",

			"Simple unbiased weights as described in the 2020 ReSTIR paper (Eq. 16 and Section 4.3).\n"
			"Those weights are unbiased but can have * *extremely * *bad variance when a neighbor being resampled "
			"has a very low target function(when the neighbor is a glossy surface for example).\n"
			"See Fig. 7 of the 2020 paper.",

			"Unbiased weights as proposed by Eq. 22 of the paper.Way better than 1 / Z in terms of variance "
			"and still unbiased.",

			"Unbiased MIS weights that use the generalized balance heuristic. Very good variance reduction but O(N ^ 2) complexity, "
			"N being the number of neighbors resampled.\n"
			"Eq. 36 of the 2022 Generalized Resampled Importance Sampling paper.",

			"Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.\n"
			"Section 7.1.3 of \"A Gentle Introduction to ReSTIR\", 2023",

			"Similar variance reduction to the generalized balance heuristic and only O(N) computational cost.\n"
			"Section 7.1.3 of \"A Gentle Introduction to ReSTIR\", 2023, defensive approach to reduce correlations at the cost of a bit higher variance",

			"A bit more variance than pairwise MIS but way more robust to temporal correlations.\n\n"
			""
			"Implementation of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, Pan et al., 2024]",

			"A bit more variance than pairwise MIS but way more robust to temporal correlations.\n\n"
			""
			"Implementation of [Enhancing Spatiotemporal Resampling with a Novel MIS Weight, Pan et al., 2024]"
		};

		if (ReSTIRVariant == ReSTIR_VARIANT_PT)
		{
			tooltips.push_back(
				"Implementation of [Stochastic Pairwise MIS for Unbiased Large - Kernel Reuse in Real - Time, Hedstrom et al. 2026] where neighbors are "
				"importance sampled based on the luminance of their samples");

			tooltips.push_back(
				"Implementation of [Stochastic Pairwise MIS for Unbiased Large - Kernel Reuse in Real - Time, Hedstrom et al. 2026] where neighbors are "
				"importance sampled based on the luminance of their samples, defensive approach to reduce correlations at the cost of a bit higher variance");
		}
		else
		{
			tooltips.push_back("Not implemented for ReSTIR DI and ReSTIR GI");
			tooltips.push_back("Not implemented for ReSTIR DI and ReSTIR GI");
		}

		std::vector<unsigned char> disabled_items = {
			false, // "- 1/M (Biased)",
			false, // "- 1/Z",
			false, // "- MIS-like",
			false, //"- Generalized balance heuristic",
			false, // "- Pairwise MIS",
			false, // "- Pairwise MIS defensive",
			false, // "- Pairwise symmetric ratio",
			false, // "- Pairwise asymmetric ratio",
		};

		if (ReSTIRVariant == ReSTIR_VARIANT_PT)
		{
			// "- Stochastic pairwise MIS"
			disabled_items.push_back(false);
			// "- Stochastic pairwise MIS defensive"
			disabled_items.push_back(false);
		}
		else
		{
			// Disabled for ReSTIR DI and ReSTIR GI

			// "- Stochastic pairwise MIS"
			disabled_items.push_back(true);
			// "- Stochastic pairwise MIS defensive"
			disabled_items.push_back(true);
		}

		int* mis_weights_type_option_pointer =
			global_kernel_options->get_raw_pointer_to_macro_value(ReSTIRVariant == ReSTIR_VARIANT_DI   ? GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE
																  : ReSTIRVariant == ReSTIR_VARIANT_GI ? GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_TYPE
																									   : GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE);
		if (ImGuiRenderer::ComboWithTooltips("MIS Weights", mis_weights_type_option_pointer, mis_weights_types_items.data(), mis_weights_types_items.size(),
											 tooltips.data(), disabled_items.data()))
		{
			m_renderer->recompile_kernels();

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("What weights to use to resample reservoirs");

		bool disable_confidence_weights =
			*mis_weights_type_option_pointer == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M || *mis_weights_type_option_pointer == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z;

		if (*mis_weights_type_option_pointer == RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO ||
			*mis_weights_type_option_pointer == RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO ||
			*mis_weights_type_option_pointer == RESTIR_MIS_WEIGHTS_TYPE_SYMMETRIC_RATIO ||
			*mis_weights_type_option_pointer == RESTIR_MIS_WEIGHTS_TYPE_ASYMMETRIC_RATIO)
		{
			if (ImGui::SliderFloat("Beta exponent", &common_settings.symmetric_ratio_mis_weights_beta_exponent, 1.0f, 5.0f))
				m_render_window->set_render_dirty(true);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}

		// No visibility for 1/M weights
		if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI || ReSTIRVariant == ReSTIR_VARIANT_GI)
		{
			bool bias_correction_visibility_disabled = *mis_weights_type_option_pointer == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M;
			bool mis_weights_use_visibility;
			if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
				mis_weights_use_visibility = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY);
			else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
				mis_weights_use_visibility = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_USE_VISIBILITY);
			ImGui::BeginDisabled(bias_correction_visibility_disabled);
			if (ImGui::Checkbox("Use visibility in MIS weights", &mis_weights_use_visibility))
			{
				int* bias_correction_use_visibility_option_pointer = global_kernel_options->get_raw_pointer_to_macro_value(
					ReSTIRVariant == ReSTIR_VARIANT_DI ? GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY
													   : GPUKernelCompilerOptions::RESTIR_GI_MIS_WEIGHTS_USE_VISIBILITY);
				*bias_correction_use_visibility_option_pointer = mis_weights_use_visibility ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE;

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
			if (bias_correction_visibility_disabled)
				ImGuiRenderer::show_help_marker("Visibility in MIS weights cannot be used with 1/M weights.");
			ImGui::EndDisabled();
		}

		ImGui::TreePop();
		ImGui::PopID();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

template <int ReSTIRVariant>
bool ImGuiSettingsWindow::compute_ReSTIR_PT_using_SPMIS_boolean()
{
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	bool using_spmis = false;
	if constexpr (ReSTIRVariant == ReSTIR_VARIANT_DI)
		// SPMIS not implemented for ReSTIR DI
		using_spmis = false;
	else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_GI)
		// SPMIS not implemented for ReSTIR GI
		using_spmis = false;
	else if constexpr (ReSTIRVariant == ReSTIR_VARIANT_PT)
		using_spmis =
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_PT_MIS_WEIGHTS_TYPE) ==
				RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE;

	return using_spmis;
}

void ImGuiSettingsWindow::draw_ReSTIR_PT_SPMIS_settings_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	bool using_spmis = compute_ReSTIR_PT_using_SPMIS_boolean<ReSTIR_VARIANT_PT>();

	bool disable_spmis_settings = !using_spmis;
	ImGui::BeginDisabled(disable_spmis_settings);
	if (ImGui::CollapsingHeader("SPMIS Settings") && !disable_spmis_settings)
	{
		ImGui::TreePush("SPMIS Settings tree");

		ReSTIRPTSPMISSettings& spmis_settings = render_settings.restir_pt_settings.spmis_settings;

		ImGui::SeparatorText("Neighbor cell search");

		if (ImGui::SliderFloat("Initial search radius", &spmis_settings.initial_search_radius, 0.0f, 32.0f))
			m_render_window->set_render_dirty(true);
		if (ImGui::SliderFloat("Search radius growth factor", &spmis_settings.neighboring_cell_search_radius_increment, 1.0f, 4.0f))
			m_render_window->set_render_dirty(true);
		if (ImGui::SliderInt("Max search iterations", &spmis_settings.neighboring_cell_max_search_iterations, 0, 16))
			m_render_window->set_render_dirty(true);

		ImGui::BeginDisabled(spmis_settings.compatibility_guided_cell_selection.do_compatibility_guided_selection);
		if (ImGui::SliderFloat("Distance scaling", &spmis_settings.distance_scaling, 0.0f, 8.0f))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker(
			"When searching for a neighboring cell to reuse from, cells further away are downweighted by 1.0f / distance_to_center_pixel to "
			"improve variance (since we will then be reusing from closer pixels). However, directly weighting by the inverse distance isn't enough "
			"so we're further scaling by a controllable factor. The lower this factor, the more closer cells are preferred. 0.0f turns off "
			"distance scaling.");
		ImGui::EndDisabled();
		if (ImGui::Checkbox("Variance aware reuse radius", &spmis_settings.variance_aware_reuse_radius))
			m_render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("Quick settings");
		if (ImGui::Button("Good input"))
		{
			spmis_settings.tile_size								= 32;
			spmis_settings.initial_search_radius					= 10.0f;
			spmis_settings.neighboring_cell_search_radius_increment = 1.25f;
			spmis_settings.neighboring_cell_max_search_iterations	= 8;
			spmis_settings.distance_scaling							= 8.0f;

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::add_tooltip(
			"Preset of settings that works well for scenes that are not too difficult, where the output of the initial candidates pass is not too "
			"sparse. Using this preset with input that is too sparse may result in correlations.");

		ImGui::SameLine();
		if (ImGui::Button("In-between"))
		{
			spmis_settings.tile_size								= 32;
			spmis_settings.initial_search_radius					= 10.0f;
			spmis_settings.neighboring_cell_search_radius_increment = 1.25f;
			spmis_settings.neighboring_cell_max_search_iterations	= 10;
			// Distance scaling off
			spmis_settings.distance_scaling = 16.0f;

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::add_tooltip("In-between good input and sparse input.");

		ImGui::SameLine();
		if (ImGui::Button("Sparse input"))
		{
			spmis_settings.tile_size								= 32;
			spmis_settings.initial_search_radius					= 20.0f;
			spmis_settings.neighboring_cell_search_radius_increment = 1.25f;
			spmis_settings.neighboring_cell_max_search_iterations	= 12;
			// Distance scaling off
			spmis_settings.distance_scaling = 0.0f;

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::add_tooltip(
			"Preset of settings that works well for scenes that are difficult to render, where the output of the initial candidates pass is sparse "
			"(only few pixels have contributing paths). Using this preset with input that is not sparse may result in increased variance and "
			"better convergence could be achieved with the \"Good input\" preset.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::Checkbox("Compatibility-guided selection", &spmis_settings.compatibility_guided_cell_selection.do_compatibility_guided_selection))
			m_render_window->set_render_dirty(true);
		if (spmis_settings.compatibility_guided_cell_selection.do_compatibility_guided_selection)
		{
			ImGui::TreePush("Compatibility guided selection settings tree");

			if (ImGui::SliderFloat("Solid angle omega", &spmis_settings.compatibility_guided_cell_selection.solid_angle_omega, 0.0f, 0.2f))
				m_render_window->set_render_dirty(true);

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Non-canonical sampling");
		if (ImGui::Checkbox("Non-canonical confidence scaling", &spmis_settings.do_non_canonical_confidence_adjustement))
			m_render_window->set_render_dirty(true);

		ImGui::BeginDisabled(spmis_settings.ris_neighbor_cdf);
		if (ImGui::SliderInt("RIS Steps", &spmis_settings.ris_neighbor_count, 1, 64))
			m_render_window->set_render_dirty(true);
		ImGui::EndDisabled();
		if (ImGui::Checkbox("CDF sampling of the cell", &spmis_settings.ris_neighbor_cdf))
			m_render_window->set_render_dirty(true);
		if (ImGui::SliderInt("CDF RIS count", &spmis_settings.ris_neighbor_cdf_count, 1, 8))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker(
			"How many pixels to sample from the reuse cell using the CDF built over the cell. Each of these neighbors is going to be RISed with a target "
			"function approximately equal to the target function of the center pixel, this improves quality at the cost of some performance.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Canonical sampling");
		if (ImGui::SliderInt("Canonical estimation count", &spmis_settings.canonical_weight_estimation_count, 1, 4))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::add_tooltip("How many samples to take to estimate the MIS weight of the canonical sample. Section 4.2 of the paper.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Hash grid settings");
		if (ImGui::SliderInt("Screen space cell size", &spmis_settings.tile_size, 1, 32, "%d", ImGuiSliderFlags_AlwaysClamp))
			m_render_window->set_render_dirty(true);
		if (ImGui::SliderInt("Normal precision", &spmis_settings.hash_normal_precision, 1, 8))
			m_render_window->set_render_dirty(true);
		if (ImGui::SliderFloat("Normal jitter strength", &spmis_settings.hash_normal_jitter_strength, 0.0f, 0.5f))
			m_render_window->set_render_dirty(true);

		ImGui::TreePop();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
	if (!using_spmis)
		ImGuiRenderer::add_tooltip("Disabled because not using SPMIS");
	ImGui::EndDisabled();
}

void ImGuiSettingsWindow::draw_ReSTIR_PT_initial_candidates_panel()
{
	if (ImGui::CollapsingHeader("Initial candidates"))
	{
		ImGui::TreePush("ReSTIR PT - Initial candidates tree");

		if (ImGui::SliderInt("Initial path trees count", &m_renderer->get_render_settings().restir_pt_settings.initial_candidates.initial_path_trees_count, 1,
							 8))
			m_render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_ReSTIR_PT_light_sampling_panel()
{
	if (ImGui::CollapsingHeader("ReSTIR PT light sampling settings"))
	{
		ImGui::TreePush("ReSTIR PT - Light sampling tree");

		if (ImGui::SliderInt("NEE RIS Light sample count",
							 &m_renderer->get_render_settings().restir_pt_settings.initial_candidates.nee_ris_number_of_light_candidates, 0, 8))
			m_render_window->set_render_dirty(true);

		if (ImGui::SliderInt("NEE RIS Envmap sample count",
							 &m_renderer->get_render_settings().restir_pt_settings.initial_candidates.nee_ris_number_of_envmap_candidates, 0, 4))
			m_render_window->set_render_dirty(true);

		if (ImGui::SliderInt("NEE RIS BSDF sample count",
							 &m_renderer->get_render_settings().restir_pt_settings.initial_candidates.nee_ris_number_of_bsdf_candidates, 0, 4))
			m_render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_next_event_estimation_plus_plus_panel()
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options	 = m_renderer->get_global_compiler_options();
	std::shared_ptr<NEEPlusPlusRenderPass> nee_plus_plus_render_pass = std::dynamic_pointer_cast<NEEPlusPlusRenderPass>(
		m_renderer->get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_render_pass(NEEPlusPlusRenderPass::NEE_PLUS_PLUS_RENDER_PASS_NAME));

	ImGui::BeginDisabled(!nee_plus_plus_render_pass);
	if (ImGui::CollapsingHeader("Next Event Estimation++") && nee_plus_plus_render_pass)
	{
		ImGui::TreePush("Use NEE++ Tree");

		use_nee_plus_plus_checkbox();
		ImGuiRenderer::show_help_marker("Whether or not to use NEE++ [Guo et al., 2020] features at all.");

		if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS) == KERNEL_OPTION_TRUE)
		{
			ImGui::TreePush("NEE++ Settings Tree");

			ImGui::Text("VRAM Usage: %.3fMB", nee_plus_plus_render_pass->get_vram_usage_bytes() / 1000000.0f);
			static bool display_load_factor = false;
			if (display_load_factor)
			{
				nee_plus_plus_render_pass->get_nee_plus_plus_storage().update_cell_alive_count();
				ImGui::Text("Load factor: %.3f%%", nee_plus_plus_render_pass->get_load_factor() * 100.0f);
			}
			else
				ImGui::Text("Load factor: ---");
			ImGui::SameLine();
			ImGui::Checkbox("Display load factor", &display_load_factor);

			if (ImGui::InputFloat("Max VRAM usage (MB)", &nee_plus_plus_render_pass->get_max_vram_usage()))
				m_render_window->set_render_dirty(true);
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			{
				if (ImGui::SliderFloat("Grid cell target projected size", &render_data.nee_plus_plus.m_grid_cell_target_projected_size, 5, 25))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("The target screen-space size (in pixels) that a grid cell should occupy on the screen.\n"
												"This has the effect of making the grid cells larger in the distance so that the projected size stays "
												"approximately constant.");

				if (ImGui::SliderFloat("Grid cell minimum size", &render_data.nee_plus_plus.m_grid_cell_min_size, 0.005, 0.5))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("The minimum size of a grid cell in world space units");

				ImGui::SliderInt("Update max samples", &render_data.nee_plus_plus.m_stop_update_samples, 1, 96);
				ImGuiRenderer::show_help_marker("After this many samples, the update of the visibility will automatically "
												"stop to save some performance because accumulating forever isn't necessary for visibility caching precision.");
				ImGui::Dummy(ImVec2(0.0f, 20.0f));

				bool use_nee_plus_plus_rr = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE);
				if (ImGui::Checkbox("Use NEE++ Russian Roulette", &use_nee_plus_plus_rr))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE,
														   use_nee_plus_plus_rr ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Implementation of NEE++, [Guo et al., 2020].\n"
												"If checked, the voxel-to-voxel visibility estimate of NEE++ will be used to "
												"stochastically determine whether or not attempt at all to trace a shadow at "
												"a light during next-event-estimation.");
				if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS_RUSSIAN_ROULETTE) == KERNEL_OPTION_TRUE)
					ImGui::Text("Shadow rays traced: %.3f%%", m_renderer->get_nee_plus_plus_storage().get_shadow_rays_actually_traced_from_GPU() /
																  (float)m_renderer->get_nee_plus_plus_storage().get_total_shadow_rays_queries_from_GPU() *
																  100.0f);

				if (use_nee_plus_plus_rr)
				{
					ImGui::TreePush("NEE++ RR options tree");

					if (ImGui::Checkbox("Use NEE++ RR for emissives", &render_data.nee_plus_plus.m_enable_nee_plus_plus_RR_for_emissives))
						m_render_window->set_render_dirty(true);

					if (ImGui::Checkbox("Use NEE++ RR for envmap", &render_data.nee_plus_plus.m_enable_nee_plus_plus_RR_for_envmap))
						m_render_window->set_render_dirty(true);

					{
						unsigned int traced = 0;
						unsigned int total	= 0;

						ImGui::SameLine();
						std::string button_text = render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics ? "Stop" : "Resume";
						if (ImGui::Button(button_text.c_str()))
							render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics =
								!render_data.nee_plus_plus.do_update_shadow_rays_traced_statistics;

						ImGui::Dummy(ImVec2(0.0f, 20.0f));
					}

					ImGui::TreePop();
				}

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
			}

			{
				if (ImGui::SliderFloat("Confidence threshold", &render_data.nee_plus_plus.m_confidence_threshold, 0.0f, 1.0f))
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

				if (ImGui::SliderFloat("Minimum unoccluded proba", &render_data.nee_plus_plus.m_minimum_unoccluded_proba, 0.0f, 0.1f))
					m_render_window->set_render_dirty(true);

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
			}

			if (ImGui::CollapsingHeader("Grid prepopulation"))
			{
				ImGui::TreePush("NEE++ Grid prepopulation tree");

				const char* items_lss[]	   = { "- Uniform sampling", "- Power sampling", "- Light tree ATS (Conty & Kulla 2018)",
											   "- SG light tree (Tokuyoshi et al. 2024)" };
				const char* tooltips_lss[] = {
					"All lights are sampled uniformly",

					"Lights are sampled proportionally to their power",

					"Lights are sampled using a light hierarchy with orientation bounds as proposed in the paper of Conty & Kulla, 2018.",

					"Lights are sampled using a light hierarchy of spherical gaussian lights as proposed in the paper of Tokuyoshi et al., 2024.",
				};

				if (ImGuiRenderer::ComboWithTooltips("Light sampling strategy",
													 global_kernel_options->get_raw_pointer_to_macro_value(
														 GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_GRID_PREPOPULATE_LIGHT_SAMPLING_STRATEGY),
													 items_lss, IM_ARRAYSIZE(items_lss), tooltips_lss))
				{
					// Will recompute the alias table if necessary
					m_renderer->recompute_emissives_sampling_data_structure();

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				if (ImGui::SliderInt("Light sample count", &render_data.nee_plus_plus.grid_prepopulate_sample_count, 1, 128))
					m_render_window->set_render_dirty(true);

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				ImGui::TreePop();
			}

			if (ImGui::CollapsingHeader("Debug"))
			{
				ImGui::TreePush("NEE++ debug tree");

				int nee_plus_plus_debug_mode = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::NEE_PLUS_PLUS_DEBUG_MODE);
				const char* items[]			 = { "- No debug", "- Grid cells" };
				if (ImGui::Combo("Debug mode", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::NEE_PLUS_PLUS_DEBUG_MODE), items,
								 IM_ARRAYSIZE(items)))
				{
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::Dummy(ImVec2(0.0f, 20.0f));
				bool display_shadow_rays =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED);
				if (ImGui::Checkbox("Display shadow rays discarded", &display_shadow_rays))
				{
					m_renderer->get_global_compiler_options()->set_macro_value(
						GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED,
						display_shadow_rays ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
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
						m_renderer->get_global_compiler_options()->set_macro_value(
							GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_PLUS_PLUS_DISPLAY_SHADOW_RAYS_DISCARDED_BOUNCE, shadow_ray_bounce_to_display);
						m_renderer->recompile_kernels();
					}

					ImGui::TreePop();
				}

				ImGui::TreePop();
			}

			ImGui::TreePop();
		}

		ImGui::TreePop();
	}

	ImGui::EndDisabled(); // ImGui::BeginDisabled(!nee_plus_plus_render_pass);
}

bool ImGuiSettingsWindow::use_nee_plus_plus_checkbox(const std::string& text)
{
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	bool use_nee_plus_plus = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS);
	if (ImGui::Checkbox(text.c_str(), &use_nee_plus_plus))
	{
		global_kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_USE_NEE_PLUS_PLUS,
											   use_nee_plus_plus ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

		m_renderer->recompile_kernels();
		m_render_window->set_render_dirty(true);

		return true;
	}

	return false;
}

void ImGuiSettingsWindow::draw_principled_bsdf_energy_conservation()
{
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	if (ImGui::CollapsingHeader("Principled BSDF energy conservation"))
	{
		ImGui::TreePush("BSDF energy conservation settings tree");

		bool do_energy_conservation = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION);
		if (ImGui::Checkbox("Do energy conservation", &do_energy_conservation))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION,
												   do_energy_conservation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Global toggle to completely enable/disable any forms "
										"of energy compensation in all the materials using the Principled BSDF");

		if (do_energy_conservation)
		{
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			const char* energy_compensation_mode[]			= { "- LUTs Turquin 2019", "- Random walk Cui 2023" };
			const char* tooltips_energy_compensation_mode[] = {
				"Uses precomputed LUTs following the method of [Practical multiple scattering compensation for microfacet models, Turquin, 2019] to compensate "
				"energy loss due to multiple scattering in microfacet BRDFs. Fast but approximate",

				"Uses an implementation of the method proposed in[Multiple-bounce Smith Microfacet BRDFs using the Invariance Principle, Cui et al., 2023] to "
				"compute the true multiple scattering path integral within the microsurface.More expensive than LUTs but more accurate and physically based."
			};

			ImGui::SeparatorText("Energy compensation method");
			if (ImGuiRenderer::ComboWithTooltips(
					"Method", global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENERGY_COMPENSATION_MODE),
					energy_compensation_mode, IM_ARRAYSIZE(energy_compensation_mode), tooltips_energy_compensation_mode))
			{
				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			if (global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENERGY_COMPENSATION_MODE) ==
				ENERGY_COMPENSATION_MODE_INVARIANCE_CUI)
			{
				ImGui::TreePush("Invariance cui settings tree");

				static int max_microsurface_bounces =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_BOUNCES);
				ImGui::SliderInt("Max microsurface bounces", &max_microsurface_bounces, 1, 15);
				if (max_microsurface_bounces !=
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_BOUNCES))
				{
					ImGui::TreePush("Max microsurface bounces apply button");

					if (ImGui::Button("Apply##max microsurface bounces"))
					{
						max_microsurface_bounces = hippt::clamp(1, 15, max_microsurface_bounces);

						global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_BOUNCES,
															   max_microsurface_bounces);

						m_renderer->recompile_kernels();
						m_render_window->set_render_dirty(true);
					}

					ImGui::TreePop();
				}

				static bool variable_bounce =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_VARIABLE_BOUNCES);
				if (ImGui::Checkbox("Variable number of bounces", &variable_bounce))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_MAX_MICROSURFACE_VARIABLE_BOUNCES,
														   variable_bounce ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGuiRenderer::show_help_marker("If true, the number of bounces in the microsurface will be variable and depend on the roughness of the "
												"material to help performance. Higher roughnesses materials will have more microsurface bounces.\n\n"
												""
												"Current roughnes to max bounces mapping is:\n"
												"\troughness >= 0.7 --> 6 bounces\n"
												"\troughness >= 0.5 --> 5 bounces\n"
												"\troughness >= 0.4 --> 4 bounces\n"
												"\troughness  < 0.4 --> 3 bounces");

				static bool do_russian_roulette =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_DO_RUSSIAN_ROULETTE);
				if (ImGui::Checkbox("Do russian roulette", &do_russian_roulette))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MULTIPLE_SCATTERING_CUI_DO_RUSSIAN_ROULETTE,
														   do_russian_roulette ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("If true, russian roulette will be applied to the random walk in the microsurface for the invariance Cui et "
												"al.method.This can help reduce the cost of that method with high number of maximum bounces but it is going to "
												"have higher variance.");

				if (do_russian_roulette)
				{
					ImGui::TreePush("Invariance cui RR settings tree");

					if (ImGui::SliderInt("RR min bounce", &render_data.bsdfs_data.multiple_scattering_cui_2023_min_bounce_russian_roulette, 1, 15))
						m_render_window->set_render_dirty(true);
					ImGuiRenderer::show_help_marker("If using the invariance Cui et al. method GGX for multiple scattering in the microsurface energy "
													"compensation, this is after how many bounces in the microsurface we start applying russian roulette to "
													"potentially end the random walk. Higher values for this parameter means that more bounces will be "
													"computed without russian roulette --> lower variance but higher cost.\n\n"
													""
													"A value of 1 means that russian roulette will be applied starting at the first bounce in the "
													"microsurface, which means that russian roulette will always be applied.");

					ImGui::TreePop();
				}

				if (ImGui::SliderFloat("Firefly clamping threshold", &render_data.bsdfs_data.multiple_scattering_cui_2023_firefly_clamping_threshold, 0.0f,
									   100.0f))
					m_render_window->set_render_dirty(true);
				ImGuiRenderer::show_help_marker("When using the invariance Cui et al. method for multiple scattering in the microsurface, this is a clamping "
												"threshold for the "
												"contribution of multiple scattering in the microsurface. The random walk weight seems a bit unstable at mid "
												"roughnesses 0.3 - 0.5 and can produce some "
												"fireflies, this is a threshold to clamp those fireflies and the contribution will be clamped to at most that "
												"value.\n\n"
												""
												"10.0f reduces fireflies considerably without producing noticeable bias in the final results.\n\n"
												""
												"0.0f completely disables clamping.");

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			ImGui::SeparatorText("Lobes compensation");
			{
				bool do_glass_energy_compensation =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION);
				if (ImGui::Checkbox("Do glass lobe energy compensation", &do_glass_energy_compensation))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION,
														   do_glass_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Global toggle on whether or not objects in the scene that use "
												"the Principled BSDF should do energy compensation for the glass layer."
												""
												"Implementation of [Practical multiple scattering compensation for microfacet models, Turquin, 2019].");
			}

			{
				bool do_clearcoat_energy_compensation =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION);
				if (ImGui::Checkbox("Do clearcoat lobe energy compensation", &do_clearcoat_energy_compensation))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION,
														   do_clearcoat_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Global toggle on whether or not objects in the scene that use "
												"the Principled BSDF should do energy compensation for the clearcoat layer.\n\n"
												""
												"Energy compensation on the clearcoat layer is an approximation but works very well in common cases.");
			}

			{
				bool do_specular_energy_compensation =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION);
				if (ImGui::Checkbox("Do specular/diffuse lobe energy compensation", &do_specular_energy_compensation))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION,
														   do_specular_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}
				ImGuiRenderer::show_help_marker("Global toggle on whether or not objects in the scene that use "
												"the Principled BSDF should do energy compensation for the glossy (specular/diffuse) layer."
												""
												"Implementation of [Practical multiple scattering compensation for microfacet models, Turquin, 2019].");
			}

			{
				bool do_metallic_energy_compensation =
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION);
				if (ImGui::Checkbox("Do metallic lobe energy compensation", &do_metallic_energy_compensation))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION,
														   do_metallic_energy_compensation ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
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

					bool multiple_scattering_fresnel_disabled =
						global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_ENERGY_COMPENSATION_MODE) !=
						ENERGY_COMPENSATION_MODE_LUTS_TURQUIN;
					ImGui::BeginDisabled(multiple_scattering_fresnel_disabled);
					bool use_multiple_scattering_fresnel =
						global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION);
					if (ImGui::Checkbox("Do GGX Multiple scattering fresnel", &use_multiple_scattering_fresnel))
					{
						global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_FRESNEL_ENERGY_COMPENSATION,
															   use_multiple_scattering_fresnel ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
						m_renderer->recompile_kernels();
						m_render_window->set_render_dirty(true);
					}
					ImGui::EndDisabled();
					if (multiple_scattering_fresnel_disabled)
						ImGuiRenderer::show_help_marker("Fresnel reflectance at each bounce in the microsurface is always enabled for multiple scattering "
														"methods other than LUTs.");
					else
						ImGuiRenderer::show_help_marker(
							"Implementation of [Practical multiple scattering compensation for microfacet models, Turquin, 2019]"
							" for GGX energy compensation. The multiple scattering fresnel term takes into account the Fresnel "
							"reflection/transmission effect when the rays bounce multiple times on the microsurface. This is responsible "
							"for the increase in saturation of the color of conductors due to multiple scattering in-between the "
							"microfacets.");

					ImGui::TreePop();
				}
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::Text("Energy compensation roughness threshold");
			if (ImGui::SliderFloat("", &render_data.bsdfs_data.energy_compensation_roughness_threshold, 0.0f, 1.0f))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker(
				"Below this roughness, energy compensation will not be applied.\n\n"
				""
				"Generally speaking, the darkening of the material due to missing energy compensation is barely visible below 0.15f "
				"roughness.\n\n"
				""
				"0.0f disables the threshold and energy compensation will always be applied.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (ImGui::Checkbox("Use hardware texture interpolation", &render_data.bsdfs_data.use_hardware_tex_interpolation))
			{
				m_renderer->load_GGX_glass_energy_compensation_textures(render_data.bsdfs_data.use_hardware_tex_interpolation ? hipFilterModeLinear
																															  : hipFilterModePoint);
				m_render_window->set_render_dirty(true);
			}
			ImGuiRenderer::show_help_marker(
				"Using the hardware for texture interpolation is faster but less precise than doing manual interpolation in the shader.");
		}

		ImGui::TreePop();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

void ImGuiSettingsWindow::display_ReSTIR_DI_bias_status(std::shared_ptr<GPUKernelCompilerOptions> kernel_options)
{
	ImGui::Text("Status: ");
	ImGui::SameLine();

	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	std::vector<std::string> bias_reasons;
	std::vector<std::string> hover_explanations;
	if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_M)
	{
		bias_reasons.push_back("- 1/M biased weights");
		hover_explanations.push_back("1/M weights do not take the number of neighbors that "
									 "could have produced the resampled sample into account.This leads to darkening "
									 "bias because we're not weighting our picked sample as if it could have been "
									 "produced by M neighbors whereas less neighbors than that could have actually produced it.");
	}

	if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE) == KERNEL_OPTION_TRUE &&
		kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY) == KERNEL_OPTION_FALSE)
	{
		bias_reasons.push_back("- Visibility reuse without visibility in MIS weights");
		hover_explanations.push_back("When using the visibility reuse pass at the end of the "
									 "initial candidates sampling pass, light samples that are occluded are discarded.\n"
									 "Temporal & spatial reuse pass will then only resample on unoccluded samples.\n"
									 "If not accounting for visibility when counting valid neighbors, we may determine "
									 "that a neighbor could have produced the picked sample when actually, it couldn't "
									 "because from the neighbor's point of view, the sample could have been occluded "
									 "(visibility reuse pass).\n"
									 "This overestimates the number of valid neighbors and results in darkening.\n\n");
	}

	if ((kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE ||
		 (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE &&
		  render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)) &&
		kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY) == KERNEL_OPTION_FALSE)
	{
		std::string prefix;
		if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE)
			prefix = " - Initial ";
		else if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_TRUE &&
				 render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass)
			prefix = " - Spatial ";

		bias_reasons.push_back(prefix + "target function visibility without\n"
										"    visibility in MIS weights");
		hover_explanations.push_back("When using the visibility term in the target function used to "
									 "produce initial candidate samples (or temporally/spatially resample), all remaining samples are unoccluded.\n"
									 "Temporal & spatial reuse passes will then only resample on unoccluded samples.\n"
									 "If not accounting for visibility when counting valid neighbors (visibility in MIS weights), we may determine "
									 "that a neighbor could have produced the picked sample when actually, it couldn't "
									 "because from the neighbor's point of view, the sample could have been occluded "
									 "(visibility term in target function).\n"
									 "This overestimates the number of valid neighbors and results in darkening.\n\n");
	}

	if (kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_VISIBILITY_REUSE) == KERNEL_OPTION_FALSE &&
		kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_INITIAL_TARGET_FUNCTION_VISIBILITY) == KERNEL_OPTION_FALSE &&
		kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_USE_VISIBILITY) == KERNEL_OPTION_TRUE &&
		(kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_1_OVER_Z ||
		 kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS ||
		 kernel_options->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_MIS_WEIGHTS_TYPE) == RESTIR_MIS_WEIGHTS_TYPE_PAIRWISE_MIS_DEFENSIVE))
	{
		bias_reasons.push_back("- Visibility in MIS weights without\n"
							   "visibility reuse (or initial candidates visibility)");
		hover_explanations.push_back("When taking visibility into account in the counting of "
									 "valid neighbors (visibility in MIS weights), we're going to assume that if the picked sample (from resampling "
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
		m_render_window->get_display_view_system()->queue_display_view_change(m_application_settings->enable_denoising ? DisplayViewType::DENOISED_BLEND
																													   : DisplayViewType::DEFAULT);
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
	if (!m_application_settings->denoise_when_rendering_done)
	{
		ImGui::TreePush("Denoise conditions tree");

		ImGui::Checkbox("Denoise only on viewport refresh", &m_application_settings->denoise_only_on_viewport_refresh);

		ImGui::TreePop();
	}
	ImGui::SliderInt("Denoiser sample skip", &m_application_settings->denoiser_sample_skip, 1, 128);
	if (ImGui::SliderFloat("Denoiser blend", &display_settings.denoiser_blend, 0.0f, 1.0f))
		m_render_window->set_force_viewport_refresh(true);
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
		changed |= ImGui::SliderFloat("Exposure", &display_settings.tone_mapping_exposure, 0.0f, 5.0f);
		if (changed)
			m_render_window->set_force_viewport_refresh(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
	std::shared_ptr<GMoNRenderPass> gmon_render_pass				= std::dynamic_pointer_cast<GMoNRenderPass>(
		m_renderer->get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_render_pass(GMoNRenderPass::GMON_RENDER_PASS_NAME));
	GMoNGPUData& gmon_data = gmon_render_pass->get_gmon_data();

	if (!render_data.render_settings.accumulate)
	{
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGuiRenderer::add_warning("GMoN cannot be used without enabling accumulation.");
	}

	ImGui::BeginDisabled(!render_data.render_settings.accumulate || !gmon_render_pass);
	if (ImGui::CollapsingHeader("GMoN") && gmon_render_pass)
	{
		ImGui::TreePush("GMoN tree post processing");

		if (ImGui::Checkbox("Use GMoN", &gmon_data.use_gmon))
			toggle_gmon();

		ImGuiRenderer::show_help_marker(
			"Use GMoN for fireflies elimination.\n"
			"The algorithm computes the median of means of the pixels as an estimator "
			"that is more robust than the simple mean usually used to average samples.\n"
			"The algorithm is unbiased as long as enough samples are accumulated. If not "
			"enough samples are accumulated, the firefly elimination tends to be a bit too "
			"strong and the image will probably end up darker than expected, especially on high-variance scenes.\n\n"
			""
			"Implementation following [Firefly removal in Monte Carlo rendering with adaptive Median of meaNs, Buisine et al., 2021]");

		if (gmon_data.use_gmon)
		{
			ImGui::Text("VRAM Usage: %.3fMB", gmon_render_pass->get_VRAM_usage_bytes() / 1000000.0f);

			bool gmon_mode_changed = false;
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::Text("GMoN Mode");
			gmon_mode_changed |= ImGui::RadioButton("Median of Means", ((int*)&render_data.buffers.gmon_estimator.gmon_mode), 0);
			ImGui::SameLine();
			gmon_mode_changed |= ImGui::RadioButton("Binary G-MoN", ((int*)&render_data.buffers.gmon_estimator.gmon_mode), 1);
			ImGui::SameLine();
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
			ImGuiRenderer::show_help_marker(
				"How many sets (M variable in the GMoN paper, [Buisine et al., 2021]) to use.\n\n"
				""
				"A simple way to choose that number is: keep that number as low as possible as long as it removes the fireflies.\n\n"
				""
				"As a general rule: more sets eliminate fireflies the best but more sets require more samples per "
				"pixel to avoid too much darkening, especially on high-variance scene. If your scene is very "
				"easy to render, you probably don't need many sets (less than 15, maybe even less than 11). If your scene has high "
				"variance caustics, you're probably going to need a lot of samples per pixel and so a large "
				"number of sets will be fine anyways.\n\n"
				""
				"Said otherwise: if you're noticing too much darkening, try reducing the number of sets or "
				"try accumulating more samples per pixel.\n\n");
			// If the user modified the number of sets, displaying an "Apply" button
			if (number_of_sets != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT))
			{
				ImGui::TreePush("GMoN Apply number of sets tree");

				if (ImGui::Button("Apply"))
				{
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::GMON_M_SETS_COUNT, number_of_sets);

					m_renderer->recompile_kernels();
					m_render_window->set_render_dirty(true);
				}

				ImGui::TreePop();
			}

			if (ImGui::SliderFloat("GMoN blend factor", &gmon_data.gmon_blend_factor, 0.0f, 1.0f))
			{
				gmon_data.gmon_auto_blend_factor = false;
				m_render_window->set_force_viewport_refresh(true);
			}
			ImGui::SameLine();
			if (ImGui::Checkbox("Auto", &gmon_data.gmon_auto_blend_factor))
				m_render_window->set_force_viewport_refresh(true);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (m_render_window->get_display_view_system()->get_current_display_view_type() != DisplayViewType::GMON_BLEND)
				ImGuiRenderer::add_warning("The display view currently in used isn't \"GMoN blend\" so the output of GMoN cannot be visualized.");
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	ImGui::EndDisabled(); // ImGui::BeginDisabled(!render_data.render_settings.accumulate || !gmon_render_pass);

	std::shared_ptr<SSBNPermutationRenderPass> ssbn_pass =
		std::dynamic_pointer_cast<SSBNPermutationRenderPass>(m_renderer->get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_render_pass(
			SSBNPermutationRenderPass::SSBN_PERMUTATION_RENDER_PASS_NAME));

	ImGui::BeginDisabled(!ssbn_pass);
	if (ImGui::CollapsingHeader("SSBN Permutation") && ssbn_pass)
	{
		ImGui::TreePush("SSBN Permutation tree");

		static bool ssbn_permutation_enabled = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_ENABLED) && ssbn_pass;
		if (ImGui::Checkbox("Enable SSBN permutation", &ssbn_permutation_enabled))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_ENABLED,
												   ssbn_permutation_enabled ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_render_window->set_render_dirty(true);
			m_renderer->recompile_kernels();
		}
		ImGuiRenderer::show_help_marker("Implementation of [Distributing Monte Carlo Errors as a Blue Noise in Screen Space by Permuting Pixel Seeds Between "
										"Frames, Heitz & Belcour, 2019]");

		if (ImGui::Checkbox("Accumulate blue noise at 1SPP", &render_data.ssbn_settings.accumulate_blue_noise_1spp))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker("This is used to force the random seeds to not be reset when we reset the render. Useful to accumulate blue noise "
										"quality with SSBN permutations, otherwise SSBN permutation always needs more than 1SPP to kick in. This breaks "
										"determinism though as all 1SPP frame will be different!");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		if (ssbn_permutation_enabled)
		{
			ImGui::SeparatorText("Sorting pass");

			static int block_size	= global_kernel_options->get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE);
			bool block_size_changed = false;
			ImGui::Text("Sorting block size");
			block_size_changed |= ImGui::RadioButton("8##block_size", &block_size, 8);
			ImGui::SameLine();
			block_size_changed |= ImGui::RadioButton("16##block_size", &block_size, 16);
			ImGui::SameLine();
			block_size_changed |= ImGui::RadioButton("32##block_size", &block_size, 32);

			if (block_size_changed)
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE, block_size);

				m_render_window->set_render_dirty(true);
				m_renderer->recompile_kernels();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			bool blue_noise_texture_size_changed = false;
			ImGui::Text("Blue noise tile size");
			blue_noise_texture_size_changed |= ImGui::RadioButton("16", &ssbn_pass->get_blue_noise_texture_width(), 16);
			ImGui::SameLine();
			blue_noise_texture_size_changed |= ImGui::RadioButton("32", &ssbn_pass->get_blue_noise_texture_width(), 32);
			ImGui::SameLine();
			blue_noise_texture_size_changed |= ImGui::RadioButton("64", &ssbn_pass->get_blue_noise_texture_width(), 64);
			ImGui::SameLine();
			blue_noise_texture_size_changed |= ImGui::RadioButton("128", &ssbn_pass->get_blue_noise_texture_width(), 128);
			ImGui::SameLine();
			blue_noise_texture_size_changed |= ImGui::RadioButton("256", &ssbn_pass->get_blue_noise_texture_width(), 256);
			ImGui::SameLine();
			blue_noise_texture_size_changed |= ImGui::RadioButton("512", &ssbn_pass->get_blue_noise_texture_width(), 512);
			ImGui::SameLine();
			if (blue_noise_texture_size_changed |= ImGui::RadioButton("4096x2048", &ssbn_pass->get_blue_noise_texture_width(), 4096))
				// Special case for 4096 x 2048 blue noise texture
				ssbn_pass->get_blue_noise_texture_height() = 2048;
			if (blue_noise_texture_size_changed && ssbn_pass->get_blue_noise_texture_width() != 4096)
				// Other cases are square textures
				ssbn_pass->get_blue_noise_texture_height() = ssbn_pass->get_blue_noise_texture_width();

			if (blue_noise_texture_size_changed)
			{
				ssbn_pass->get_max_retargeting_radius() = SSBNPermutationRenderPass::DEFAULT_MAX_RETARGETING_RADIUS;

				unsigned int permutation_block_size_clamping = hippt::min(
					global_kernel_options->get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE), ssbn_pass->get_blue_noise_texture_width());
				if (permutation_block_size_clamping != global_kernel_options->get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE))
				{
					block_size = permutation_block_size_clamping;
					global_kernel_options->set_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_BLOCK_SIZE, permutation_block_size_clamping);

					m_renderer->recompile_kernels();
				}

				m_renderer->reload_ssbn_permutation_blue_noise_texture(ssbn_pass->get_blue_noise_texture_width(), ssbn_pass->get_blue_noise_texture_height());
				m_render_window->set_render_dirty(true);
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			if (ImGui::Checkbox("Use screen space hash grid", &render_data.ssbn_settings.use_screen_space_hash_grid))
				m_render_window->set_render_dirty(true);
			if (render_data.ssbn_settings.use_screen_space_hash_grid)
			{
				ImGui::TreePush("Screen space hash grid tree");

				if (ImGui::Checkbox("Use surface normal", &render_data.ssbn_settings.use_surface_normal))
					m_render_window->set_render_dirty(true);

				ImGui::TreePop();
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Retargeting pass");
			if (ImGui::Checkbox("Do retargeting", &ssbn_pass->get_do_retargeting()))
				m_render_window->set_render_dirty(true);

			bool max_retargeting_radius_changed = false;
			ImGui::BeginDisabled(!ssbn_pass->get_do_retargeting());
			ImGui::Text("Max retargeting radius (in pixels)");

			std::vector<int> radii = { 2, 3, 4, 5, 6, 7, 15, 32 };
			for (int i = 0; i < radii.size(); i++)
			{
				max_retargeting_radius_changed |=
					ImGui::RadioButton((std::to_string(radii[i]) + "##retargeting_max_radius").c_str(), &ssbn_pass->get_max_retargeting_radius(), radii[i]);
				if (i != radii.size() - 1)
					ImGui::SameLine();
			}

			if (max_retargeting_radius_changed)
			{
				ssbn_pass->reload_retargeting_data_only(ssbn_pass->get_max_retargeting_radius());

				m_render_window->set_render_dirty(true);
			}
			ImGui::EndDisabled(); // ImGui::BeginDisabled(!ssbn_pass->get_do_retargeting());

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Refresh seeds pass");
			if (ImGui::SliderInt("Refresh seeds interval", &ssbn_pass->get_refresh_seeds_sample_interval(), 0, 64))
				m_render_window->set_render_dirty(true);
			ImGuiRenderer::show_help_marker(
				"Refreshing the seeds used for rendering once in a while. This is needed to ensure convergence because otherwise, the "
				"sorting/retargeting pass keep shuffling around the same seeds forever. This is fine for a few samples but eventually "
				"pixels will run out of fresh seeds to integrate their pixel value and we'll lose convergence. We thus need to refresh the "
				"seeds eventually. Lower values mean more frequent refreshes and better MSE convergence (but not perceptual convergence!) "
				"but also more noise because each frame rendered with refreshed seeds is basically a white noise frame. Higher values mean "
				"better blue noise but worse convergence / more bias on the whole image. 0 never refreshes seeds.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::SeparatorText("Debug");

			static bool visualize_hash_grid = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_HASH_GRID);
			if (ImGui::Checkbox("Debug hash grid", &visualize_hash_grid))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_HASH_GRID, visualize_hash_grid);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			static bool debug_seeds = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_SEEDS);
			if (ImGui::Checkbox("Debug seeds", &debug_seeds))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::SSBN_PERMUTATION_DEBUG_SEEDS, debug_seeds);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}
		}

		ImGui::TreePop();
	}
	ImGui::EndDisabled(); // ImGui::BeginDisabled(!ssbn_pass);

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::TreePop();
}

void ImGuiSettingsWindow::toggle_gmon()
{
	std::shared_ptr<GMoNRenderPass> gmon_render_pass = std::dynamic_pointer_cast<GMoNRenderPass>(
		m_renderer->get_render_graphs()[GPURendererThread::RENDER_GRAPH_FULL_NAME].get_render_pass(GMoNRenderPass::GMON_RENDER_PASS_NAME));

	bool gmon_now_enabled = gmon_render_pass->get_gmon_data().use_gmon;
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

	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	ImGui::TreePush("Quality settings tree");

	ImGui::SeparatorText("Nested dielectrics");
	{
		ImGui::TreePush("Nested dielectrics tree");

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

	ImGui::SeparatorText("Alpha testing");
	{
		ImGui::TreePush("Alpha testing tree");

		if (ImGui::Checkbox("Do alpha testing", &render_settings.do_alpha_testing))
			m_render_window->set_render_dirty(true);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	ImGui::SeparatorText("Textures");
	{
		ImGui::TreePush("Quality settings textures tree");

		if (ImGui::Checkbox("Do normal mapping", &render_settings.do_normal_mapping))
			m_render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		bool use_material_textures = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES);
		if (ImGui::Checkbox("Use material textures", &use_material_textures))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::USE_MATERIAL_TEXTURES,
												   use_material_textures ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		if (!use_material_textures)
		{
			ImGui::TreePush("Use material texture base color texture override");

			bool use_material_textures_base_color_override =
				global_kernel_options->get_macro_value(GPUKernelCompilerOptions::USE_MATERIAL_BASE_COLOR_TEXTURE_OVERRIDE);
			if (ImGui::Checkbox("Use base color texture anyway", &use_material_textures_base_color_override))
			{
				global_kernel_options->set_macro_value(GPUKernelCompilerOptions::USE_MATERIAL_BASE_COLOR_TEXTURE_OVERRIDE,
													   use_material_textures_base_color_override ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

				m_renderer->recompile_kernels();
				m_render_window->set_render_dirty(true);
			}

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	ImGui::SeparatorText("Triangle sampling");
	{
		ImGui::TreePush("Triangle sampling tree");

		const char* items_triangle_sampling[]	 = { "- Turk 1990", "- Heitz 2019" };
		const char* tooltips_triangle_sampling[] = { "Common way of warping from a square to a triangle using square roots:\n"
													 "V = (1.0f - sqrt(u1))* V1 + sqrt(u1)* (s2* V2 + (1.0f - s2) * V3)",

													 "Implementation of[A Low - Distortion Map Between Triangle and Square, Heitz, 2019]\n"
													 "It is faster than Turk method's and better perserves the stratification of the random "
													 "number samplers" };
		if (ImGuiRenderer::ComboWithTooltips(
				"Triangle point sampling strategy",
				global_kernel_options->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::TRIANGLE_POINT_SAMPLING_UNIFORM_AREA_STRATEGY),
				items_triangle_sampling, IM_ARRAYSIZE(items_triangle_sampling), tooltips_triangle_sampling))
		{
			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	ImGui::SeparatorText("Light clamping");
	{
		ImGui::TreePush("Lighting Settings Performance Tree");

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
		bool allow_backfacing_lights = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS);
		if (ImGui::Checkbox("Allow backfacing lights", &allow_backfacing_lights))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_ALLOW_BACKFACING_LIGHTS,
												   allow_backfacing_lights ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	ImGui::SeparatorText("Microfacet regularization");
	draw_microfacet_model_regularization_tree();

	ImGui::TreePop();
}

void ImGuiSettingsWindow::draw_microfacet_model_regularization_tree()
{
	ImGui::TreePush("Microfacet regularization");

	HIPRTRenderData& render_data									= m_renderer->get_render_data();
	HIPRTRenderSettings& render_settings							= m_renderer->get_render_settings();
	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();

	bool regularize_bsdf = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION);
	if (ImGui::Checkbox("Do microfacet model regularization", &regularize_bsdf))
	{
		global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION,
											   regularize_bsdf ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

		m_renderer->recompile_kernels();
		m_render_window->set_render_dirty(true);
	}
	if (regularize_bsdf && render_data.bsdfs_data.GGX_masking_shadowing == GGXMaskingShadowingFlavor::HeightCorrelated)
	{
		ImGuiRenderer::add_warning("Microfacet model regularization cannot be used with height-correlated masking shadowing");

		ImGui::TreePush("Use height uncorrelated button tree");
		if (ImGui::Button("Switch to height-uncorrelated masking shadowing"))
			render_data.bsdfs_data.GGX_masking_shadowing = GGXMaskingShadowingFlavor::HeightUncorrelated;
		ImGui::TreePop();
	}
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	if (regularize_bsdf)
	{
		bool do_consistent_tau =
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION);
		if (ImGui::Checkbox("Consistent parameterization", &do_consistent_tau))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_MICROFACET_REGULARIZATION_CONSISTENT_PARAMETERIZATION,
												   do_consistent_tau ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("With this feature enabled, tau will refine over time to help sharpen caustics a bit"
										" while keeping variance in check.");
		if (do_consistent_tau)
		{
			ImGui::TreePush("Consistent tau tree");
			ImGui::Text("Current tau: %f", MicrofacetRegularization::consistent_tau(render_data.bsdfs_data.microfacet_regularization.tau_0,
																					render_data.render_settings.sample_number));
			ImGui::TreePop();
		}
		bool do_diffusion_heuristic =
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC);
		if (ImGui::Checkbox("Use diffusion heuristic", &do_diffusion_heuristic))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_MICROFACET_REGULARIZATION_DIFFUSION_HEURISTIC,
												   do_diffusion_heuristic ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

			m_renderer->recompile_kernels();
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Whether or not to take the path's roughness into account when regularizing the BSDFs.\n"
										"This feature is essential to keep highlights sharp on directly visible surfaces.");

		std::string tau_text = std::string("Tau") + (do_consistent_tau ? "_0" : "");
		if (ImGui::SliderFloat(tau_text.c_str(), &render_data.bsdfs_data.microfacet_regularization.tau_0, 10.0f, 1000.0f))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker("Main parameter to control the regularization. The lower this parameter, the stronger the regularization.\n\n"
										"Note that if \"Consistent parameterization\" is enabled, this parameter will be adjusted dynamically (starting from "
										"the given value) based on the number "
										"of samples rendered so far.");
		if (ImGui::SliderFloat("Minimum roughness", &render_data.bsdfs_data.microfacet_regularization.min_roughness, 0.0f, 1.0f))
			m_render_window->set_render_dirty(true);
		ImGuiRenderer::show_help_marker("All materials in the scene will at least have that much roughness.\n\n"
										"Useful when lights are so small that even camera ray jittering causes variance and so roughening the surface helps "
										"BSDF rays hit the light source more often (and light samples too). The main purpose is to help with very sharp glossy "
										"highlights. "
										"Regularization is only applied during NEE so the direct apperance of smooth objects isn't affected.");
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

	std::shared_ptr<GPUKernelCompilerOptions> global_kernel_options = m_renderer->get_global_compiler_options();
	HardwareAccelerationSupport hwi_supported						= m_renderer->device_supports_hardware_acceleration();

	if (ImGui::CollapsingHeader("General Settings"))
	{
		ImGui::TreePush("Perf settings general settings tree");

		if (ImGui::InputFloat("GPU Stall Percentage", &m_application_settings->GPU_stall_percentage))
			m_application_settings->GPU_stall_percentage = std::max(0.0f, std::min(m_application_settings->GPU_stall_percentage, 99.9f));
		ImGuiRenderer::show_help_marker(
			"How much percent of the time the GPU will be forced to be idle (not rendering anything)."
			" This feature is basically only meant for GPUs that get too hot to avoid burning your GPUs during long renders if you have"
			" time to spare.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		draw_russian_roulette_options();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		bool delta_distrib_opti = global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION);
		if (ImGui::Checkbox("BSDF delta distribution optimization", &delta_distrib_opti))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DELTA_DISTRIBUTION_EVALUATION_OPTIMIZATION,
												   delta_distrib_opti ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
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
		if (!delta_distrib_opti && global_kernel_options->get_macro_value(GPUKernelCompilerOptions::PATH_SAMPLING_STRATEGY) == PATH_SAMPLING_RESTIR_GI)
		{
			ImGuiRenderer::add_warning("Due to numerical float imprecisions, errors on specular surfaces (especially glass) "
									   "are expected with ReSTIR GI if not using \"BSDF delta distribution optimization\"."
									   "\nThis will manifest as darkening on perfectly specular surfaces (delta distributions).\n\n"
									   ""
									   "Enable \"BSDF delta distribution optimization\" to get rid of this issue.");
		}

		bool direct_light_delta_distrib_opti =
			global_kernel_options->get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION);
		if (ImGui::Checkbox("NEE delta distribution optimization", &direct_light_delta_distrib_opti))
		{
			global_kernel_options->set_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_SAMPLING_BSDF_DELTA_DISTRIBUTION_OPTIMIZATION,
												   direct_light_delta_distrib_opti ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
			m_renderer->recompile_kernels();

			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("If this is true, then NEE light samples will not even be attempted of delta distribution materials."
										"This is because for delta distribution materials, an arbitrary incident light direction will always produce a "
										"0-contribution outgoing radiance "
										"so doing NEE with light samples on these materials is useless and we can save some computations here.");

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Ray-tracing settings"))
	{
		ImGui::TreePush("Ray-tracing settings tree");

		static bool use_hardware_acceleration = global_kernel_options->has_macro("__USE_HWI__");
		ImGui::BeginDisabled(hwi_supported != HardwareAccelerationSupport::SUPPORTED);
		if (ImGui::Checkbox("Use ray tracing hardware acceleration", &use_hardware_acceleration))
		{
			global_kernel_options->set_macro_value("__USE_HWI__", use_hardware_acceleration);

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

		bool bvh_needs_rebuild			   = false;
		static int build_type_chosen	   = 0;
		std::vector<const char*> bvh_items = { "- SBVH", "- HPLOC", "- LBVH" };
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

			m_renderer->rebuild_whole_scene_bvh(build_flags, do_bvh_compaction);
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Kernel settings"))
	{
		ImGui::TreePush("Shared/global stack Traversal Options Tree");

		// List of exceptions because these kernels do not trace any rays
		static std::vector<std::string> kernel_names;
		static std::map<std::string, std::shared_ptr<GPUKernel>> kernels = m_renderer->get_tracing_kernels();
		if (kernel_names.empty())
			// Filling the kernel names if not already done
			for (const auto& name_to_kernel : kernels)
				kernel_names.push_back(name_to_kernel.first);

		static std::string selected_kernel_name					 = FillGBufferRenderPass::FILL_GBUFFER_KERNEL;
		static std::shared_ptr<GPUKernel> selected_kernel		 = kernels[selected_kernel_name];
		static GPUKernelCompilerOptions* selected_kernel_options = &selected_kernel->get_kernel_options();

		if (ImGui::BeginCombo("Kernel", selected_kernel_name.c_str()))
		{
			for (const std::string& kernel_name : kernel_names)
			{
				const bool is_selected = (selected_kernel_name == kernel_name);
				if (ImGui::Selectable(kernel_name.c_str(), is_selected))
				{
					selected_kernel_name	= kernel_name;
					selected_kernel			= kernels[selected_kernel_name];
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
				use_shared_stack_traversal[selected_kernel_name] =
					selected_kernel_options->get_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL);
			bool& use_shared_stack_traversal_bool = use_shared_stack_traversal[selected_kernel_name];

			if (ImGui::Checkbox("Use shared/global stack BVH traversal", &use_shared_stack_traversal_bool))
			{
				selected_kernel_options->set_macro_value(GPUKernelCompilerOptions::USE_SHARED_STACK_BVH_TRAVERSAL,
														 use_shared_stack_traversal_bool ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);
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
					pending_stack_size_changes[selected_kernel_name] =
						selected_kernel_options->get_macro_value(GPUKernelCompilerOptions::SHARED_STACK_BVH_TRAVERSAL_SIZE);
				int& pending_stack_size = pending_stack_size_changes[selected_kernel_name];

				ImGui::PushItemWidth(8 * ImGui::GetFontSize());
				if (ImGui::InputInt("Shared stack size", &pending_stack_size))
					pending_stack_size = std::max(0, pending_stack_size);
				ImGui::PopItemWidth();

				ImGuiRenderer::show_help_marker("Fast shared memory stack used for the BVH traversal of \"global\" rays (rays that search for a closest hit "
												"with no maximum distance)\n\n"
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
			m_renderer->get_render_data().global_traversal_stack_buffer_size =
				hippt::clamp(0, 128, m_renderer->get_render_data().global_traversal_stack_buffer_size);
			m_render_window->set_render_dirty(true);
		}
		ImGui::PopItemWidth();

		ImGuiRenderer::show_help_marker(
			"Size of the global stack buffer for each thread. Used for complementing the shared memory stack allocated in the kernels."
			"A good value for this parameter is scene-complexity dependent.\n\n"
			"A lower value will use less VRAM but will start introducing artifacts if the value is too low due "
			"to insufficient stack size for the BVH traversal.\n\n"
			"16 seems to be a good value to start with. If lowering this value improves performance, then that "
			"means that the BVH traversal is starting to suffer (the traversal is incomplete --> improved performance) "
			"and rendering artifacts will start to show up.");

		std::string size_string = "Global Stack Buffer VRAM Usage: ";
		size_string += std::to_string(m_renderer->get_render_data().global_traversal_stack_buffer_size * std::ceil(m_renderer->m_render_resolution.x / 8.0f) *
									  8.0f * std::ceil(m_renderer->m_render_resolution.y / 8.0f) * 8.0f * sizeof(int) / 1000000.0f);
		size_string += " MB";
		ImGui::Text("%s", size_string.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Lighting settings"))
	{
		ImGui::TreePush("Lighting settings tree");

		draw_next_event_estimation_plus_plus_panel();

		ImGui::TreePop();
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::TreePop();
}

void ImGuiSettingsWindow::draw_performance_metrics_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (!ImGui::CollapsingHeader("Performance metrics"))
		return;

	ImGui::TreePush("Performance metrics tree");

	ImGui::Text("Device: %s", m_renderer->get_device_properties().name);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (ImGui::InputInt("Max sample count", &m_application_settings->max_sample_count))
		m_application_settings->max_sample_count = std::max(m_application_settings->max_sample_count, 0);
	if (ImGui::InputInt("Samples per frame", &render_settings.samples_per_frame))
	{
		// Clamping to 1
		render_settings.samples_per_frame = std::max(1, render_settings.samples_per_frame);

		// If the user manually changed to number of samples per frame, let's disable auto sample per frame
		// because the user probably doesn't want it
		m_application_settings->auto_sample_per_frame = false;
	}
	ImGui::SameLine();
	ImGui::Checkbox("Auto", &m_application_settings->auto_sample_per_frame);

	bool rolling_window_size_changed = false;
	int rolling_window_size			 = m_render_window_perf_metrics->get_window_size();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::Text("Measures Window Size");
	ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("25", &rolling_window_size, 25);
	ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("100", &rolling_window_size, 100);
	ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("250", &rolling_window_size, 250);
	ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("1000", &rolling_window_size, 1000);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	if (rolling_window_size_changed)
		m_render_window_perf_metrics->resize_window(rolling_window_size);

	RenderGraph& render_graph = m_renderer->get_active_render_graph();
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

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	ImGui::TreePop();
}

void ImGuiSettingsWindow::draw_perf_metric_specific_panel(std::shared_ptr<PerformanceMetricsComputer> perf_metrics,
														  const std::string& perf_metrics_key,
														  const std::string& label)
{
	float variance, min, max;
	variance = perf_metrics->get_variance(perf_metrics_key);
	min		 = perf_metrics->get_min(perf_metrics_key);
	max		 = perf_metrics->get_max(perf_metrics_key);

	static std::unordered_map<std::string, bool> key_to_display_graph;
	if (key_to_display_graph.find(perf_metrics_key) == key_to_display_graph.end())
		key_to_display_graph[perf_metrics_key] = false;

	// Pusing the ID for that perf key metrics so that no ImGui widgets collide
	ImGui::PushID(perf_metrics_key.c_str());

	ImGui::Text("%s: %.3fms (%.1f FPS)", label.c_str(), perf_metrics->get_current_value(perf_metrics_key),
				1000.0f / perf_metrics->get_average(perf_metrics_key));
	if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
	{
		std::string line_1 =
			format_perf_metrics_tooltip_line(label, " (avg):", " (min / max):", " %.3fms (%.1f FPS)", perf_metrics->get_average(perf_metrics_key),
											 1000.0f / perf_metrics->get_average(perf_metrics_key));
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
		scale_min		 = perf_metrics->get_data_index(perf_metrics_key) == 0 ? min : scale_min;
		scale_max		 = perf_metrics->get_data_index(perf_metrics_key) == 0 ? max : scale_max;

		ImGui::PlotHistogram("", PerformanceMetricsComputer::data_getter, perf_metrics->get_data(perf_metrics_key).data(),
							 perf_metrics->get_value_count(perf_metrics_key),
							 /* value offset */ 0, label.c_str(), scale_min, scale_max,
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
std::string ImGuiSettingsWindow::format_perf_metrics_tooltip_line(const std::string& label,
																  const std::string& suffix,
																  const std::string& longest_header_for_padding,
																  const std::string& formatter_after_header,
																  const Args&... args)
{
	// Creating the formatter for automatically left-padding the header of the lines to the longer line (which is "(min / max)")
	std::string header_padding_formatter = "%-" + std::to_string(label.length() + longest_header_for_padding.length()) + "s";
	std::string line_formatter			 = header_padding_formatter + formatter_after_header;
	std::string header					 = label + suffix;

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

		if (ImGui::Button("Hard shaders reload"))
		{
			m_renderer->recompile_kernels(false);
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Forces the recompilation of the shaders without using the shader cache.");
		if (ImGui::Button("Soft shaders reload"))
		{
			m_renderer->recompile_kernels(true);
			m_render_window->set_render_dirty(true);
		}
		ImGuiRenderer::show_help_marker("Recompiles the shaders using the shader cache.");

		if (ImGui::Button("Clear shader cache"))
			std::filesystem::remove_all("shader_cache");
		ImGuiRenderer::show_help_marker("Completely clears the shader cache on the disk.");

		static GPUKernelCompiler::ShaderCacheUsageOverride shader_cache_use_override = g_gpu_kernel_compiler.get_shader_cache_usage_override();
		std::vector<const char*> shader_cache_override_values						 = { "No override", "Do not use shader cache", "Always use shader cache" };
		if (ImGui::Combo("Shader cache use override", (int*)&shader_cache_use_override, shader_cache_override_values.data(),
						 shader_cache_override_values.size()))
			g_gpu_kernel_compiler.set_shader_cache_usage_override(shader_cache_use_override);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::CollapsingHeader("Kernels compilation statistics"))
		{
			ImGui::TreePush("Kernel compilation statistics tree");

			ImGui::Text("Kernel [Registers, Shared Memory, Local Memory]");
			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			// Computing the longest kernel name for aligning everything
			size_t longest_kernel_name = 0;
			for (auto kernel_name_to_kernel : m_renderer->get_all_kernels())
				longest_kernel_name = hippt::max(longest_kernel_name, kernel_name_to_kernel.first.length());
			std::string padding_formatter = "%-" + std::to_string(longest_kernel_name) + "s";

			for (auto kernel_name_to_kernel : m_renderer->get_all_kernels())
			{
				const std::string& kernel_name			= kernel_name_to_kernel.first;
				const std::shared_ptr<GPUKernel> kernel = kernel_name_to_kernel.second;

				if (kernel->has_been_compiled())
				{
					int nb_reg	  = kernel->get_kernel_attribute(ORO_FUNC_ATTRIBUTE_NUM_REGS);
					int nb_shared = kernel->get_kernel_attribute(ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
					int nb_local  = kernel->get_kernel_attribute(ORO_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);

					ImGui::PushID(kernel_name.c_str());
					if (ImGui::Button("C"))
					{
						std::string commandline_string;

						commandline_string = "hipcc -x hip ";

						std::vector<std::string> options = kernel->get_kernel_options().get_all_macros_as_std_vector_string();
						for (std::string& option : options)
							commandline_string += option + " ";

						std::vector<std::string> include_directories = GPUKernel::COMMON_ADDITIONAL_KERNEL_INCLUDE_DIRS;
						for (std::string& include_dir : include_directories)
							commandline_string += "-I" + include_dir + " ";

						// For debugging info and assembly-source code line correspondences
						commandline_string += "-std=c++17 -gline-tables-only --save-temps ";
						// For hardware ray tracing instructions
						commandline_string += "--offload-arch=gfx1100 ";
						// To avoid some warnings caused by kernel compilation options causing constant
						// compile-time operands like 1 || 1 or 0 && 1, stuff like that
						commandline_string += "-Wno-constant-logical-operand -Wno-tautological-compare ";
						// Source file that hipcc compiles
						commandline_string += "../src/llvm-compile-kernel.h";

						// For outputting the disassembly + source line correspondances to a .txt and opening it with notepad++
						commandline_string += " && llvm-objdump --no-show-raw-insn -S llvm-compile-kernel-hip-amdgcn-amd-amdhsa-gfx1100.out > assembly.txt && "
											  "notepad++.exe assembly.txt &";

						ImGui::SetClipboardText(commandline_string.c_str());
					}
					ImGuiRenderer::add_tooltip("Copies the hipcc compilation command to the clipboard.");
					ImGui::PopID();

					ImGui::SameLine();
					std::string text = padding_formatter + " [%d, %d, %d]";
					ImGui::Text(text.c_str(), kernel_name.c_str(), nb_reg, nb_shared, nb_local);
				}
				else
				{
					std::string text = padding_formatter + " [Not compiled]";
					ImGui::Text(text.c_str(), kernel_name.c_str());
				}
			}

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiSettingsWindow::draw_debug_panel()
{
	if (!ImGui::CollapsingHeader("Debug"))
		return;

	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	ImGui::TreePush("Debug tree");

	if (ImGui::CollapsingHeader("Debug/WIP options"))
	{
		ImGui::TreePush("Debug options tree");

		ImGui::PushItemWidth(24 * ImGui::GetFontSize());

		if (ImGui::SliderInt("ReGIR Pre integration iterations", &render_settings.DEBUG_REGIR_PRE_INTEGRATION_ITERATIONS, 1, 64))
			m_render_window->set_render_dirty(true);
		if (ImGui::SliderInt("ReGIR Pre integration sample per res", &render_settings.DEBUG_REGIR_PRE_INTEGRATION_SAMPLE_COUNT_PER_RESERVOIR, 1, 64))
			m_render_window->set_render_dirty(true);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		static int debug_index = 0;
		if (ImGui::InputInt("Debug index", &debug_index))
			debug_index = hippt::clamp(0, 1023, debug_index);
		unsigned long long int sum_count = OrochiBuffer<unsigned long long int>::download_data(
			reinterpret_cast<unsigned long long int*>(render_settings.DEBUG_BUFFER_ULL_1) + debug_index, 1)[0];
		unsigned long long int sums = OrochiBuffer<unsigned long long int>::download_data(
			reinterpret_cast<unsigned long long int*>(render_settings.DEBUG_BUFFER_ULL_2) + debug_index, 1)[0];
		ImGui::Text("Debug sum count / sums / ratio:"
					"\n\t%llu"
					"\n\t%llu"
					"\n\t%f",
					sum_count, sums, sum_count / (double)sums);

		ImGui::TreePop();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}

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

	static bool display_only_sample = DisplayOnlySampleN;
	if (ImGui::Checkbox("Display only sample N", &display_only_sample))
	{
		m_renderer->get_global_compiler_options()->set_macro_value(GPUKernelCompilerOptions::DISPLAY_ONLY_SAMPLE_N,
																   display_only_sample ? KERNEL_OPTION_TRUE : KERNEL_OPTION_FALSE);

		m_render_window->set_render_dirty(true);
		m_renderer->recompile_kernels();
	}
	if (display_only_sample)
	{
		ImGui::SameLine();
		ImGui::PushItemWidth(16 * ImGui::GetFontSize());
		if (ImGui::InputInt("", &m_renderer->get_render_data().render_settings.output_debug_sample_N))
			m_render_window->set_render_dirty(true);

		static bool auto_sample = true;
		ImGui::SameLine();
		ImGui::Checkbox("Auto", &auto_sample);
		if (auto_sample)
		{
			int new_sample_count = m_render_window->get_application_settings()->max_sample_count - 1;

			if (m_renderer->get_render_data().render_settings.output_debug_sample_N != new_sample_count)
				m_render_window->set_render_dirty(true);

			m_renderer->get_render_data().render_settings.output_debug_sample_N = m_render_window->get_application_settings()->max_sample_count - 1;
		}
	}

	ImGui::TreePop();
}
