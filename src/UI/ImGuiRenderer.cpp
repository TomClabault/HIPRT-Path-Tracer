/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ImGuiRenderer.h"
#include "UI/RenderWindow.h"

#include <chrono>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

void ImGuiRenderer::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;

	m_application_settings = render_window->get_application_settings();
	m_renderer = render_window->get_renderer();
	m_denoiser = render_window->get_denoiser();
	m_perf_metrics = m_render_window->get_performance_metrics();
}

void ImGuiRenderer::draw_imgui_interface()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	ImGuiIO& io = ImGui::GetIO();
	ImGui::ShowDemoWindow();

	ImGui::Begin("Settings");

	auto now_time = std::chrono::high_resolution_clock::now();
	if (!m_render_window->is_rendering_done())
	{
		float sample_time = m_renderer->get_sample_time();

		if (!render_settings.render_low_resolution)
			// Not adding the frame time if we're rendering low resolution, not relevant
			m_perf_metrics->add_value(PerformanceMetricsComputer::SAMPLE_TIME_KEY, sample_time);
	}

	ImGui::Text("Render time: %.3fs", m_render_window->get_current_render_time() / 1000.0f);
	ImGui::Text("%d samples | %.2f samples/s @ %dx%d", render_settings.sample_number, m_render_window->get_samples_per_second(), m_renderer->m_render_width, m_renderer->m_render_height);

	ImGui::Separator();

	if (ImGui::Button("Save viewport to PNG"))
		m_render_window->get_screenshoter()->write_to_png();

	ImGui::Separator();

	ImGui::PushItemWidth(233);

	draw_render_settings_panel();
	draw_environment_panel();
	draw_sampling_panel();
	draw_objects_panel();
	draw_denoiser_panel();
	draw_post_process_panel();
	draw_performance_panel();
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

	if (ImGui::Combo("Render Kernel", &m_application_settings->selected_kernel, "Full Path Tracer\0Normals Visualisation\0\0"))
	{
		m_renderer->compile_trace_kernel(m_application_settings->kernel_files[m_application_settings->selected_kernel].c_str(), m_application_settings->kernel_functions[m_application_settings->selected_kernel].c_str());
		m_render_window->set_render_dirty(true);
	}

	const char* items[] = { "- Default", "- Denoiser blend", "- Denoiser - Normals", "- Denoiser - Denoised normals", "- Denoiser - Albedo", "- Denoiser - Denoised albedo", "- Adaptive sampling heatmap" };
	if (ImGui::Combo("Display View", (int*)(&m_application_settings->display_view), items, IM_ARRAYSIZE(items)))
		m_render_window->change_display_view(m_application_settings->display_view);

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

	if (ImGui::InputInt("Max Sample Count", &m_application_settings->max_sample_count))
		m_application_settings->max_sample_count = std::max(m_application_settings->max_sample_count, 0);

	unsigned int converged_count;
	unsigned int total_pixel_count;
	ImGui::BeginDisabled(render_settings.enable_adaptive_sampling);
	if (ImGui::InputFloat("Stop render at noise threshold", &render_settings.stop_noise_threshold))
	{
		bool need_buffers = false;
		need_buffers |= render_settings.enable_adaptive_sampling == 1;
		need_buffers |= render_settings.stop_noise_threshold > 0.0f;

		unsigned int zero_data = 0;
		render_settings.stop_noise_threshold = std::max(0.0f, render_settings.stop_noise_threshold);
		m_renderer->get_stop_noise_threshold_buffer().upload_data(&zero_data);
		m_renderer->toggle_adaptive_sampling_buffers(need_buffers);
	}

	ImGui::TreePush("Tree stop noise threshold");
	{
		converged_count = m_renderer->get_stop_noise_threshold_buffer().download_data()[0] * (!render_settings.enable_adaptive_sampling);
		total_pixel_count = m_renderer->m_render_width * m_renderer->m_render_height;
		ImGui::Text("Pixels converged: %d / %d - %.4f%%", converged_count, total_pixel_count, static_cast<float>(converged_count) / total_pixel_count * 100.0f);
	}
	ImGui::TreePop();
	ImGui::EndDisabled();

	if (ImGui::InputInt("Max bounces", &render_settings.nb_bounces))
	{
		// Clamping to 0 in case the user input a negative number of bounces	
		render_settings.nb_bounces = std::max(render_settings.nb_bounces, 0);
		m_render_window->set_render_dirty(true);
	}
	ImGui::Separator();
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

	ImGui::Separator();
	if (ImGui::CollapsingHeader("Adaptive sampling"))
	{
		ImGui::TreePush("Adaptive sampling tree");

		if (ImGui::Checkbox("Enable adaptive sampling", (bool*)&render_settings.enable_adaptive_sampling))
		{
			bool need_buffers = false;
			need_buffers |= render_settings.enable_adaptive_sampling == 1;
			need_buffers |= render_settings.stop_noise_threshold > 0.0f;

			m_renderer->toggle_adaptive_sampling_buffers(need_buffers);
			m_render_window->set_render_dirty(true);
		}

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

	if (ImGui::CollapsingHeader("Nested dielectrics"))
	{
		ImGui::TreePush("Nested dielectrics tree");

		const char* items[] = { "- Automatic", "- With priorities" };
		if (ImGui::Combo("Nested dielectrics strategy", m_renderer->get_kernel_option_pointer(GPUKernelOptions::INTERIOR_STACK_STRATEGY), items, IM_ARRAYSIZE(items)))
		{
			m_renderer->compile_trace_kernel(m_application_settings->kernel_files[m_application_settings->selected_kernel].c_str(), m_application_settings->kernel_functions[m_application_settings->selected_kernel].c_str());

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

		render_made_piggy |= ImGui::RadioButton("None", ((int*)&m_renderer->get_world_settings().ambient_light_type), 0); ImGui::SameLine();
		render_made_piggy |= ImGui::RadioButton("Use uniform lighting", ((int*)&m_renderer->get_world_settings().ambient_light_type), 1); ImGui::SameLine();
		render_made_piggy |= ImGui::RadioButton("Use envmap lighting", ((int*)&m_renderer->get_world_settings().ambient_light_type), 2);

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

	if (ImGui::CollapsingHeader("Sampling"))
	{
		ImGui::TreePush("Sampling tree");

		const char* items[] = { "- No direct light sampling", "- Uniform one light", "- BSDF Sampling", "- MIS (1 Light + 1 BSDF)", "- RIS BDSF + Light candidates" };
		if (ImGui::Combo("Direct light sampling strategy", m_renderer->get_kernel_option_pointer(GPUKernelOptions::DIRECT_LIGHT_SAMPLING_STRATEGY), items, IM_ARRAYSIZE(items)))
		{
			m_renderer->compile_trace_kernel(m_application_settings->kernel_files[m_application_settings->selected_kernel].c_str(), m_application_settings->kernel_functions[m_application_settings->selected_kernel].c_str());

			m_render_window->set_render_dirty(true);
		}

		// Display additional widgets to control the parameters of the direct light
		// sampling strategy chosen (the number of candidates for RIS for example)
		switch (m_renderer->get_kernel_option_value(GPUKernelOptions::DIRECT_LIGHT_SAMPLING_STRATEGY))
		{
		case LSS_NO_DIRECT_LIGHT_SAMPLING:
			break;

		case LSS_UNIFORM_ONE_LIGHT:
			break;

		case LSS_MIS_LIGHT_BSDF:
			break;

		case LSS_RIS_BSDF_AND_LIGHT:
			static bool use_visiblity_checked = m_renderer->get_kernel_option_value(GPUKernelOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION) == 1;
			if (ImGui::Checkbox("Use visibility in target function", &use_visiblity_checked))
			{
				m_renderer->set_kernel_option(GPUKernelOptions::RIS_USE_VISIBILITY_TARGET_FUNCTION, use_visiblity_checked ? 1 : 0);
				m_renderer->compile_trace_kernel(m_application_settings->kernel_files[m_application_settings->selected_kernel].c_str(), m_application_settings->kernel_functions[m_application_settings->selected_kernel].c_str());

				m_render_window->set_render_dirty(true);
			}

			if (ImGui::SliderInt("RIS # of BSDF candidates", &render_settings.ris_number_of_bsdf_candidates, 0, 32))
			{
				// Clamping to 0
				render_settings.ris_number_of_bsdf_candidates = std::max(0, render_settings.ris_number_of_bsdf_candidates);

				m_render_window->set_render_dirty(true);
			}

			if (ImGui::SliderInt("RIS # of light candidates", &render_settings.ris_number_of_light_candidates, 0, 128))
			{
				// Clamping to 0
				render_settings.ris_number_of_light_candidates = std::max(0, render_settings.ris_number_of_light_candidates);

				m_render_window->set_render_dirty(true);
			}

			break;

		default:
			break;
		}

		ImGui::TreePop();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

void ImGuiRenderer::draw_objects_panel()
{
	if (!ImGui::CollapsingHeader("Objects"))
		return;
	ImGui::TreePush("Objects tree");

	std::vector<RendererMaterial> materials = m_renderer->get_materials();

	int material_modfied_id = -1;
	int material_counter = 0;
	bool some_material_changed = false;

	ImGui::PushItemWidth(384);
	for (RendererMaterial& material : materials)
	{
		// Multiple ImGui widgets cannot have the same label
		// If all our materials use the same "Base color", "Subsurface", ... labels for
		// the slider, there is a chance that the slider will be linked together
		// and that multiple materials will be modified when only touching one slider
		// One solution to that is to avoid using the same label for multiple sliders
		// by naming them "material 1 Base color", "material 2 Base color" for example
		// This is however not very practical so ImGui provides us with the PushID function which
		// essentially differentiate the widgets without having to change the labels 
		ImGui::PushID(material_counter);

		some_material_changed |= ImGui::ColorEdit3("Base color", (float*)&material.base_color);
		some_material_changed |= ImGui::SliderFloat("Subsurface", &material.subsurface, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Specular", &material.specular, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Specular tint strength", &material.specular_tint, 0.0f, 1.0f);
		some_material_changed |= ImGui::ColorEdit3("Specular color", (float*)&material.specular_color);
		some_material_changed |= ImGui::SliderFloat("Roughness", &material.roughness, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Anisotropic", &material.anisotropic, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Anisotropic rotation", &material.anisotropic_rotation, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Sheen", &material.sheen, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Sheen tint strength", &material.sheen_tint, 0.0f, 1.0f);
		some_material_changed |= ImGui::ColorEdit3("Sheen color", (float*)&material.sheen_color);
		some_material_changed |= ImGui::SliderFloat("Clearcoat", &material.clearcoat, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Clearcoat roughness", &material.clearcoat_roughness, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Clearcoat IOR", &material.clearcoat_ior, 0.0f, 5.0f);
		some_material_changed |= ImGui::SliderFloat("IOR", &material.ior, 0.0f, 5.0f);
		ImGui::Separator();
		some_material_changed |= ImGui::SliderFloat("Transmission", &material.specular_transmission, 0.0f, 1.0f);
		some_material_changed |= ImGui::SliderFloat("Absorption distance", &material.absorption_at_distance, 0.0f, 20.0f);
		some_material_changed |= ImGui::ColorEdit3("Absorption color", (float*)&material.absorption_color);
		unsigned short int zero = 0, eight = 8;
		ImGui::BeginDisabled(material.specular_transmission == 0.0f || m_renderer->get_kernel_option_value(GPUKernelOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITES);
		some_material_changed |= ImGui::SliderScalar("Dielectric priority", ImGuiDataType_U16, &material.dielectric_priority, &zero, &eight);
		ImGui::EndDisabled();
		some_material_changed |= ImGui::ColorEdit3("Emission", (float*)&material.emission, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);

		ImGui::PopID();

		ImGui::Separator();

		if (some_material_changed && material_modfied_id == -1)
			material_modfied_id = material_counter;
		material_counter++;
	}
	ImGui::PopItemWidth();

	if (some_material_changed)
	{
		RendererMaterial& material = materials[material_modfied_id];
		material.make_safe();
		material.precompute_properties();

		m_renderer->update_materials(materials);
		m_render_window->set_render_dirty(true);
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
		m_render_window->change_display_view(m_application_settings->enable_denoising ? DisplayView::DENOISED_BLEND : DisplayView::DEFAULT);
	ImGui::BeginDisabled(!m_application_settings->enable_denoising);
	if (ImGui::CollapsingHeader("AOVs"))
	{
		ImGui::TreePush("Denoiser AOVs Tree");
		if (ImGui::Checkbox("Use albedo AOV", &m_application_settings->denoiser_use_albedo))
		{
			m_denoiser->set_use_albedo(m_application_settings->denoiser_use_albedo);
			if (!m_application_settings->denoiser_use_albedo)
			{
				// We're forcing the use of normals AOV off here because it seems like OIDN doesn't support normal
				// AOV without also using albedo AOV (at least I got some oidn::Exception when I tried
				// using the normals without the albedo).
				m_application_settings->denoiser_use_normals = false;
				m_denoiser->set_use_normals(false);
			}

			m_denoiser->finalize();
		}
		ImGui::SameLine();
		if (ImGui::Checkbox("Denoise albedo", &m_application_settings->denoiser_denoise_albedo))
		{
			m_denoiser->set_denoise_albedo(m_application_settings->denoiser_denoise_albedo);
			m_denoiser->finalize();
		}
		ImGui::BeginDisabled(!m_application_settings->denoiser_use_albedo);
		if (ImGui::Checkbox("Use normals AOV", &m_application_settings->denoiser_use_normals))
		{
			m_denoiser->set_use_normals(m_application_settings->denoiser_use_normals);
			m_denoiser->finalize();
		}
		ImGui::SameLine();
		if (ImGui::Checkbox("Denoise normals", &m_application_settings->denoiser_denoise_normals))
		{
			m_denoiser->set_denoise_normals(m_application_settings->denoiser_denoise_normals);
			m_denoiser->finalize();
		}
		ImGui::EndDisabled();
		ImGui::TreePop();
	}
	ImGui::Checkbox("Only Denoise at \"Max Sample Count\"", &m_application_settings->denoise_at_max_samples);
	ImGui::SliderInt("Denoise Sample Skip", &m_application_settings->denoiser_sample_skip, 1, 128);
	ImGui::SliderFloat("Denoiser blend", &m_application_settings->denoiser_blend, 0.0f, 1.0f);
	ImGui::EndDisabled();

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

void ImGuiRenderer::draw_performance_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (!ImGui::CollapsingHeader("Performance"))
		return;

	ImGui::TreePush("Performance tree");

	ImGui::Text("Device: %s", m_renderer->get_device_properties().name);
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (ImGui::Button("Apply benchmark settings"))
	{
		render_settings.freeze_random = true;
		render_settings.enable_adaptive_sampling = false;
		m_application_settings->auto_sample_per_frame = false;

		m_render_window->set_render_dirty(true);
	}
	if (ImGui::Checkbox("Freeze random", (bool*)&render_settings.freeze_random))
		m_render_window->set_render_dirty(true);

	bool rolling_window_size_changed = false;
	int rolling_window_size = m_perf_metrics->get_window_size();
	rolling_window_size_changed |= ImGui::RadioButton("25", &rolling_window_size, 25); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("100", &rolling_window_size, 100); ImGui::SameLine();
	rolling_window_size_changed |= ImGui::RadioButton("1000", &rolling_window_size, 1000);

	if (rolling_window_size_changed)
		m_perf_metrics->resize_window(rolling_window_size);

	float variance, min, max;
	variance = m_perf_metrics->get_variance(PerformanceMetricsComputer::SAMPLE_TIME_KEY);
	min = m_perf_metrics->get_min(PerformanceMetricsComputer::SAMPLE_TIME_KEY);
	max = m_perf_metrics->get_max(PerformanceMetricsComputer::SAMPLE_TIME_KEY);

	static float scale_min = min, scale_max = max;
	scale_min = m_perf_metrics->get_data_index(PerformanceMetricsComputer::SAMPLE_TIME_KEY) == 0 ? min : scale_min;
	scale_max = m_perf_metrics->get_data_index(PerformanceMetricsComputer::SAMPLE_TIME_KEY) == 0 ? max : scale_max;

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::PlotHistogram("",
		PerformanceMetricsComputer::data_getter,
		m_perf_metrics->get_data(PerformanceMetricsComputer::SAMPLE_TIME_KEY).data(),
		m_perf_metrics->get_value_count(PerformanceMetricsComputer::SAMPLE_TIME_KEY),
		/* value offset */0,
		"Sample time",
		scale_min, scale_max,
		/* size */ ImVec2(0, 80));
	static bool auto_rescale = false;
	ImGui::SameLine();
	if (ImGui::Button("Rescale") || auto_rescale)
	{
		scale_min = min;
		scale_max = max;
	}
	ImGui::SameLine();
	ImGui::Checkbox("Auto-rescale", & auto_rescale);

	ImGui::Text("Sample time (avg)      : %.3fms (%.1f FPS)", m_perf_metrics->get_average(PerformanceMetricsComputer::SAMPLE_TIME_KEY), 1000.0f / m_perf_metrics->get_average(PerformanceMetricsComputer::SAMPLE_TIME_KEY));
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

	ImGui::TreePush("Debug tree");
	ImGui::TreePop();
}
