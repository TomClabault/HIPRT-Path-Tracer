/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/Baker/GGXConductorDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GGXGlassDirectionalAlbedoSettings.h"
#include "Renderer/Baker/GGXThinGlassDirectionalAlbedoSettings.h"

#include "UI/ImGui/ImGuiToolsWindow.h"
#include "UI/RenderWindow.h"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

#include "implot.h"

const char* ImGuiToolsWindow::TITLE = "Tools";

void ImGuiToolsWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;

	m_renderer = m_render_window->get_renderer();
}

void ImGuiToolsWindow::draw()
{
	ImGui::Begin(ImGuiToolsWindow::TITLE);

	ImGui::PushItemWidth(16 * ImGui::GetFontSize());

	draw_ggx_energy_compensation_panel();
	draw_image_difference_panel();
	draw_graph_convergence_panel();

	ImGui::PopItemWidth();

	ImGui::End();
}

void ImGuiToolsWindow::draw_ggx_energy_compensation_panel()
{
	if (ImGui::CollapsingHeader("Baking"))
	{
		ImGui::TreePush("Baking tree");

		if (ImGui::CollapsingHeader("GGX Energy compensation"))
		{
			ImGui::TreePush("Baking GGX Energy compensation tree");

			draw_GGX_conductors();
			draw_GGX_fresnel();
			draw_GGX_glass();
			draw_GGX_thin_glass();
			draw_glossy_dielectric();

			static std::vector<float> roughnesses = { 0.0f, 0.25f, 0.5f, 1.0f };
			static std::vector<float> iors = { 1.0f, 1.1f, 1.3f, 1.5f, 2.0f };
			static bool cooking = false;
			static bool next_step_ready = true;
			static int step = -1;
			int nb_steps = roughnesses.size() * iors.size();

			if (ImGui::Button("Start screenshotting"))
			{
				step = -1;
				cooking = true;
			}

			if (cooking)
			{
				if (next_step_ready && step < nb_steps - 1)
				{
					next_step_ready = false;
					step++;

					std::vector<CPUMaterial> materials = m_renderer->get_current_materials();
					materials[0].ior = iors[step % iors.size()];
					materials[0].roughness = roughnesses[step / iors.size()];
					materials[0].make_safe();

					m_renderer->update_all_materials(materials);
					m_render_window->set_render_dirty(true);
				}
				else
				{
					if (m_render_window->is_rendering_done() && m_renderer->get_render_settings().sample_number > m_renderer->get_render_settings().adaptive_sampling_min_samples)
					{
						std::string filename = "Screenshot" + std::to_string(roughnesses[step / iors.size()]) + "x" + std::to_string(iors[step % iors.size()]) + " - " + std::to_string(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_COS_THETA_O) + "x" + std::to_string(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_ROUGHNESS) + "x" + std::to_string(GPUBakerConstants::GGX_THIN_GLASS_DIRECTIONAL_ALBEDO_TEXTURE_SIZE_IOR) + "x" + ".png";
						m_render_window->get_screenshoter()->write_to_png(filename);

						next_step_ready = true;
						if (step == nb_steps - 1)
							cooking = false;
					}
				}
			}

			ImGui::TreePop();
		}

		ImGui::TreePop();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
	}
}

/**
 * Panel for the GGX conductors directional albedo
 */
void ImGuiToolsWindow::draw_GGX_conductors()
{
	if (ImGui::CollapsingHeader("GGX Conductors Directional Albedo"))
	{
		ImGui::TreePush("GGX_E tree");

		static GGXConductorDirectionalAlbedoSettings ggx_dir_albedo_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size - Cos Theta", &ggx_dir_albedo_settings.texture_size_cos_theta);
		ImGui::InputInt("Texture Size - Roughness", &ggx_dir_albedo_settings.texture_size_roughness);
		ImGui::InputInt("Integration Sample Count", &ggx_dir_albedo_settings.integration_sample_count);
		std::vector<const char*> masking_shadowing_items = { "- Smith height-correlated", "- Smith height-uncorrelated" };
		ImGui::Combo("GGX Masking-Shadowing", (int*)&ggx_dir_albedo_settings.masking_shadowing_term, masking_shadowing_items.data(), masking_shadowing_items.size());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = GPUBakerConstants::get_GGX_conductor_directional_albedo_texture_filename(ggx_dir_albedo_settings.masking_shadowing_term,
				ggx_dir_albedo_settings.texture_size_cos_theta, 
				ggx_dir_albedo_settings.texture_size_roughness);

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool bake_started_at_least_once = false;
		ImGui::BeginDisabled(!baker->is_ggx_conductor_directional_albedo_bake_complete() && bake_started_at_least_once);
		if (ImGui::Button("Bake!"))
		{
			bake_started_at_least_once = true;
			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
			baker->bake_ggx_conductor_directional_albedo(ggx_dir_albedo_settings, output_filename);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (!baker->is_ggx_conductor_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking...";
		else if (baker->is_ggx_conductor_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking complete!";

		ImGui::SameLine();
		ImGui::Text("%s", baking_text.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

/**
 * Panel for the GGX fresnel directional albedo
 */
void ImGuiToolsWindow::draw_GGX_fresnel()
{
	if (ImGui::CollapsingHeader("GGX + Fresnel Directional Albedo"))
	{
		ImGui::TreePush("GGX_fresnel tree");

		static GGXFresnelDirectionalAlbedoSettings ggx_fresnel_dir_albedo_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size - Cos Theta", &ggx_fresnel_dir_albedo_settings.texture_size_cos_theta);
		ImGui::InputInt("Texture Size - Roughness", &ggx_fresnel_dir_albedo_settings.texture_size_roughness);
		ImGui::InputInt("Texture Size - IOR", &ggx_fresnel_dir_albedo_settings.texture_size_ior);
		ImGui::InputInt("Integration Sample Count", &ggx_fresnel_dir_albedo_settings.integration_sample_count);
		std::vector<const char*> masking_shadowing_items = { "- Smith height-correlated", "- Smith height-uncorrelated" };
		ImGui::Combo("GGX Masking-Shadowing", (int*)&ggx_fresnel_dir_albedo_settings.masking_shadowing_term, masking_shadowing_items.data(), masking_shadowing_items.size());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = GPUBakerConstants::get_GGX_fresnel_directional_albedo_texture_filename(ggx_fresnel_dir_albedo_settings.masking_shadowing_term,
				ggx_fresnel_dir_albedo_settings.texture_size_cos_theta, 
				ggx_fresnel_dir_albedo_settings.texture_size_roughness, 
				ggx_fresnel_dir_albedo_settings.texture_size_ior);

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool bake_started_at_least_once = false;
		ImGui::BeginDisabled(!baker->is_ggx_fresnel_directional_albedo_bake_complete() && bake_started_at_least_once);
		if (ImGui::Button("Bake!"))
		{
			bake_started_at_least_once = true;
			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
			baker->bake_ggx_fresnel_directional_albedo(ggx_fresnel_dir_albedo_settings, output_filename);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (!baker->is_ggx_fresnel_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking...";
		else if (baker->is_ggx_fresnel_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking complete!";

		ImGui::SameLine();
		ImGui::Text("%s", baking_text.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

/**
 * Panel for the GGX directional albedo over the sphere for
 * glass material energy compensation
 */
void ImGuiToolsWindow::draw_GGX_glass()
{
	if (ImGui::CollapsingHeader("Glass Directional Albedo"))
	{
		ImGui::TreePush("GGX_E_glass tree");

		static GGXGlassDirectionalAlbedoSettings ggx_glass_dir_albedo_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size - Cos Theta", &ggx_glass_dir_albedo_settings.texture_size_cos_theta_o);
		ImGui::InputInt("Texture Size - Roughness", &ggx_glass_dir_albedo_settings.texture_size_roughness);
		ImGui::InputInt("Texture Size - IOR", &ggx_glass_dir_albedo_settings.texture_size_ior);
		ImGui::InputInt("Integration Sample Count", &ggx_glass_dir_albedo_settings.integration_sample_count);
		std::vector<const char*> masking_shadowing_items = { "- Smith height-correlated", "- Smith height-uncorrelated" };
		ImGui::Combo("GGX Masking-Shadowing", (int*)&ggx_glass_dir_albedo_settings.masking_shadowing_term, masking_shadowing_items.data(), masking_shadowing_items.size());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = GPUBakerConstants::get_GGX_glass_directional_albedo_texture_filename(ggx_glass_dir_albedo_settings.masking_shadowing_term,
				ggx_glass_dir_albedo_settings.texture_size_cos_theta_o,
				ggx_glass_dir_albedo_settings.texture_size_roughness,
				ggx_glass_dir_albedo_settings.texture_size_ior);

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool bake_started_at_least_once = false;
		ImGui::BeginDisabled(!baker->is_ggx_glass_directional_albedo_bake_complete() && bake_started_at_least_once);
		if (ImGui::Button("Bake!"))
		{
			bake_started_at_least_once = true;
			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
			baker->bake_ggx_glass_directional_albedo(ggx_glass_dir_albedo_settings, output_filename);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (!baker->is_ggx_glass_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking...";
		else if (baker->is_ggx_glass_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking complete!";

		ImGui::SameLine();
		ImGui::Text("%s", baking_text.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

void ImGuiToolsWindow::draw_GGX_thin_glass()
{
	if (ImGui::CollapsingHeader("Thin Glass Directional Albedo"))
	{
		ImGui::TreePush("GGX_thin_glass tree");

		static GGXThinGlassDirectionalAlbedoSettings ggx_thin_glass_dir_albedo_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size - Cos Theta", &ggx_thin_glass_dir_albedo_settings.texture_size_cos_theta_o);
		ImGui::InputInt("Texture Size - Roughness", &ggx_thin_glass_dir_albedo_settings.texture_size_roughness);
		ImGui::InputInt("Texture Size - IOR", &ggx_thin_glass_dir_albedo_settings.texture_size_ior);
		ImGui::InputInt("Integration Sample Count", &ggx_thin_glass_dir_albedo_settings.integration_sample_count);
		std::vector<const char*> masking_shadowing_items = { "- Smith height-correlated", "- Smith height-uncorrelated" };
		ImGui::Combo("GGX Masking-Shadowing", (int*)&ggx_thin_glass_dir_albedo_settings.masking_shadowing_term, masking_shadowing_items.data(), masking_shadowing_items.size());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = GPUBakerConstants::get_GGX_thin_glass_directional_albedo_texture_filename(ggx_thin_glass_dir_albedo_settings.masking_shadowing_term,
				ggx_thin_glass_dir_albedo_settings.texture_size_cos_theta_o,
				ggx_thin_glass_dir_albedo_settings.texture_size_roughness,
				ggx_thin_glass_dir_albedo_settings.texture_size_ior);

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool bake_started_at_least_once = false;
		ImGui::BeginDisabled(!baker->is_ggx_glass_directional_albedo_bake_complete() && bake_started_at_least_once);
		if (ImGui::Button("Bake!"))
		{
			bake_started_at_least_once = true;
			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
			baker->bake_ggx_thin_glass_directional_albedo(ggx_thin_glass_dir_albedo_settings, output_filename);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (!baker->is_ggx_thin_glass_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking...";
		else if (baker->is_ggx_thin_glass_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking complete!";

		ImGui::SameLine();
		ImGui::Text("%s", baking_text.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

/**
 * Panel for the glossy dielectric directional albedo
 */
void ImGuiToolsWindow::draw_glossy_dielectric()
{
	if (ImGui::CollapsingHeader("Glossy Dielectric Directional Albedo"))
	{
		ImGui::TreePush("Glossy Dielectric tree");

		static GlossyDielectricDirectionalAlbedoSettings glossy_dielectric_albedo_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size - Cos Theta", &glossy_dielectric_albedo_settings.texture_size_cos_theta_o);
		ImGui::InputInt("Texture Size - Roughness", &glossy_dielectric_albedo_settings.texture_size_roughness);
		ImGui::InputInt("Texture Size - IOR", &glossy_dielectric_albedo_settings.texture_size_ior);
		ImGui::InputInt("Integration Sample Count", &glossy_dielectric_albedo_settings.integration_sample_count);
		std::vector<const char*> masking_shadowing_items = { "- Smith height-correlated", "- Smith height-uncorrelated" };
		ImGui::Combo("GGX Masking-Shadowing", (int*)&glossy_dielectric_albedo_settings.masking_shadowing_term, masking_shadowing_items.data(), masking_shadowing_items.size());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = GPUBakerConstants::get_glossy_dielectric_directional_albedo_texture_filename(glossy_dielectric_albedo_settings.masking_shadowing_term,
				glossy_dielectric_albedo_settings.texture_size_cos_theta_o,
				glossy_dielectric_albedo_settings.texture_size_roughness,
				glossy_dielectric_albedo_settings.texture_size_ior);

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool bake_started_at_least_once = false;
		ImGui::BeginDisabled(!baker->is_glossy_dielectric_directional_albedo_bake_complete() && bake_started_at_least_once);
		if (ImGui::Button("Bake!"))
		{
			bake_started_at_least_once = true;
			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
			baker->bake_glossy_dielectric_directional_albedo(glossy_dielectric_albedo_settings, output_filename);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (!baker->is_glossy_dielectric_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking...";
		else if (baker->is_glossy_dielectric_directional_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking complete!";

		ImGui::SameLine();
		ImGui::Text("%s", baking_text.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

#define USING_VIEWPORT_TEXT "Using viewport"
void ImGuiToolsWindow::draw_image_difference_panel()
{
	if (ImGui::CollapsingHeader("Image difference"))
	{
		ImGui::TreePush("Image difference tree");

		const char* filters[] = {"*.png", "*.jpg"};

		static float error_value = 1.0f;
		static std::string status_text = "";
		static std::string reference_image_path = "";
		static std::string subject_image_path = "";

		static Image32Bit reference_image;
		static Image32Bit subject_image;

		ImGui::Text("Step 1 - Select a reference image");
		if (ImGui::Button("Select reference image"))
		{
			reference_image_path = Utils::open_file_dialog(filters, 2);
			reference_image = Image32Bit::read_image(reference_image_path, 3, false);
		}
		if (reference_image_path != "")
		{
			ImGui::TreePush("Reference image text tree");
			
			if (ImGui::Button("C"))
				Utils::copy_image_to_clipboard(reference_image);
			ImGuiRenderer::add_tooltip("Copies the image to the clipboard");
			std::string filename = std::filesystem::path(reference_image_path).filename().string();
			ImGui::SameLine();  ImGui::Text("%s", filename.c_str());

			ImGui::TreePop();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("Step 2 - Select a subject image");
		static std::string subject_image_text = "";
		if (ImGui::Button("Select subject image"))
		{
			subject_image_path = Utils::open_file_dialog(filters, 2);
			subject_image = Image32Bit::read_image(subject_image_path, 3, false);

			subject_image_text = std::filesystem::path(subject_image_path).filename().string();
		}
		ImGui::SameLine();
		if (ImGui::Button("Use viewport"))
		{
			subject_image = Image32Bit(m_render_window->get_screenshoter()->get_image(), 3);
			subject_image_text = USING_VIEWPORT_TEXT;
		}
		if (subject_image_text != "")
		{
			ImGui::TreePush("Subject image text tree");

			if (ImGui::Button("C"))
				Utils::copy_image_to_clipboard(subject_image);
			ImGuiRenderer::add_tooltip("Copies the image to the clipboard");
			ImGui::SameLine();
			ImGui::Text("%s", subject_image_text.c_str());

			ImGui::TreePop();
		}

		bool ready_to_compute = reference_image.width != 0 && subject_image.width != 0;

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("Step 3 - Compute a difference metric");
		ImGui::BeginDisabled(!ready_to_compute);
		if (ImGui::Button("Compute MSE"))
		{
			if (subject_image_text == USING_VIEWPORT_TEXT)
				// Updating the subject image with the viewport
				subject_image = Image32Bit(m_render_window->get_screenshoter()->get_image(), 3);

			if (reference_image.width != subject_image.width ||
				reference_image.height != subject_image.height)
			{
				status_text = "Error: Images must have the same dimensions!";
			}
			else
			{
				error_value = Utils::compute_image_mse(reference_image, subject_image);

				status_text = std::string("MSE: " + std::to_string(error_value));
			}
		}

		if (ImGui::Button("Compute RMSE"))
		{
			if (subject_image_text == USING_VIEWPORT_TEXT)
				// Updating the subject image with the viewport
				subject_image = Image32Bit(m_render_window->get_screenshoter()->get_image(), 3);

			if (reference_image.width != subject_image.width ||
				reference_image.height != subject_image.height)
			{
				status_text = "Error: Images must have the same dimensions!";
			}
			else
			{
				error_value = Utils::compute_image_rmse(reference_image, subject_image);

				status_text = std::string("RMSE: " + std::to_string(error_value));
			}
		}

		static bool output_flip_error_map = false;
		if (ImGui::Button("Compute FLIP"))
		{
			if (subject_image_text == USING_VIEWPORT_TEXT)
				// Updating the subject image with the viewport
				subject_image = Image32Bit(m_render_window->get_screenshoter()->get_image(), 3);

			if (reference_image.width != subject_image.width ||
				reference_image.height != subject_image.height)
			{
				status_text = "Error: Images must have the same dimensions!";
			}
			else
			{
				float* error_map = nullptr;
				error_value = Utils::compute_image_weighted_median_FLIP(reference_image, subject_image, &error_map);

				if (output_flip_error_map)
					// Write the error map to disk
					Utils::copy_image_to_clipboard(Image32Bit(error_map, reference_image.width, reference_image.height, 3));
				free(error_map);

				status_text = std::string("FLIP: " + std::to_string(error_value));
			}
		}
		ImGui::TreePush("Output FLIP error map tree");
		ImGui::Checkbox("Copy error map to clipboard", &output_flip_error_map);
		ImGui::TreePop();
		ImGui::EndDisabled();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (status_text != "")
		{
			if (ImGui::Button("C"))
				ImGui::SetClipboardText(std::to_string(error_value).c_str());

			ImGuiRenderer::show_help_marker("Copies the error value to the clipboard");
			ImGui::SameLine();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
		}
		ImGui::Text("%s", status_text.c_str());

		ImGui::TreePop();
	}
}

void ImGuiToolsWindow::draw_graph_convergence_panel()
{
	HIPRTRenderSettings& render_settings = m_renderer->get_render_settings();

	if (ImGui::CollapsingHeader("Convergence graph"))
	{
		ImGui::TreePush("Convergence graph tree");

		ImGui::Text("Step 1: Choose a reference image");
		ImGui::TreePush("Convergence graph - step 1 tree");

		static Image32Bit ref_image;
		static std::string ref_image_text = "";
		if (ImGui::Button("Select reference image"))
		{
			const char* filters[] = { "*.png", "*.jpg" };
			std::string ref_image_path = Utils::open_file_dialog(filters, 2);
			ref_image = Image32Bit::read_image(ref_image_path, 3, false);
			ref_image_text = std::filesystem::path(ref_image_path).filename().string();
		}
		if (ref_image_text != "")
		{
			ImGui::TreePush("Reference image text tree");
			ImGui::Text("%s", ref_image_text.c_str());
			ImGui::TreePop();
		}

		ImGui::TreePop();



		ImGui::Dummy(ImVec2(0.0f, 20.0f));



		ImGui::Text("Step 2: Data capture settings");
		ImGui::TreePush("Convergence graph - step 2 tree");
		static int capture_interval_type = 0;
		ImGui::SeparatorText("Capture interval type");
		ImGui::RadioButton("Every N seconds", &capture_interval_type, 0);
		ImGui::RadioButton("Every N samples", &capture_interval_type, 1);

		static float capture_interval_value = 1.0f;
		ImGui::InputFloat("Capture interval value", &capture_interval_value);
		if (capture_interval_type == 1)
			// We want an integer number of samples
			capture_interval_value = std::roundf(capture_interval_value);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Capture duration");

		static int number_of_captures = 16;
		std::vector<std::string>& recorded_legends = m_convergence_graph_widget.get_recorded_legends();
		std::vector<std::vector<float>>& recorded_xs_list = m_convergence_graph_widget.get_recorded_xs_list();
		std::vector<std::vector<float>>& recorded_ys_list = m_convergence_graph_widget.get_recorded_ys_list();
		if (ImGui::InputInt("Number of captures", &number_of_captures))
		{
			if (recorded_xs_list.size() > 0)
			{
				if (number_of_captures > recorded_xs_list.at(0).size())
				{
					// Extending all recorded xs and ys
					for (int i = 0; i < recorded_xs_list.size(); i++)
					{
						for (int j = recorded_xs_list.at(i).size(); j < number_of_captures; j++)
						{
							recorded_xs_list.at(i).push_back((float)((j + 1) * capture_interval_value));
							recorded_ys_list.at(i).push_back(0.0f);
						}
					}
				}
				else if (number_of_captures < recorded_xs_list.at(0).size())
				{
					// Reducing all recorded xs and ys
					for (int i = 0; i < recorded_xs_list.size(); i++)
					{
						recorded_xs_list.at(i).resize(number_of_captures);
						recorded_ys_list.at(i).resize(number_of_captures);
					}
				}
			}
		}


		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		static int error_metric_type = 2;
		ImGui::SeparatorText("Error metric");
		ImGui::RadioButton("MSE", &error_metric_type, 0);
		ImGui::RadioButton("RMSE", &error_metric_type, 1);
		ImGui::RadioButton("FLIP", &error_metric_type, 2);

		ImGui::TreePop();



		ImGui::Dummy(ImVec2(0.0f, 20.0f));



		ImGui::Text("Step 3: Configure your render settings...");
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::Text("Step 4: Capture...");
		ImGui::TreePush("Start capture tree");
		// Only used in "real-time" non accumulated mode for sampled-based captures
		static int total_samples_rendered = 0;
		static bool capture_requested = false;
		static bool capture_started = false;
		static int captures_taken = 0;
		static float last_captured_ratio = 0.0f;
		static std::vector<float> current_captured_errors;
		static std::vector<float> current_recorded_xs;
		static std::vector<float> current_recorded_ys;

		if (capture_started)
		{
			ImGui::BeginDisabled(true);
			ImGui::Button("Capturing... ");
			ImGui::EndDisabled();

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.0f, 0.0f, 1.0f));        // Red
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.2f, 0.2f, 1.0f)); // Lighter red when hovered
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.0f, 0.0f, 1.0f));  // Darker red when clicked
			if (ImGui::Button("Stop capturing"))
				capture_started = false;
			ImGui::PopStyleColor(3);

			ImGui::SameLine();
			ImGui::Text("%d / %d", captures_taken, number_of_captures);
		}
		else
		{
			ImGui::BeginDisabled(ref_image.width == 0);

			if (ImGui::Button("Start capture"))
			{
				// Removing auto samples per frame for consistency and to avoid
				// that multiple samples are rendered between two captures (especially when using sample-based captures)
				render_settings.samples_per_frame = 1;
				m_render_window->get_application_settings()->auto_sample_per_frame = false;
				m_render_window->get_application_settings()->max_sample_count = 0;
				m_render_window->get_application_settings()->max_render_time = 0;

				captures_taken = 0;
				last_captured_ratio = 0.0f;
				total_samples_rendered = 0;
				current_captured_errors.clear();
				current_recorded_xs.clear();
				current_recorded_ys.clear();

				for (int i = 0; i < number_of_captures; i++)
					current_recorded_xs.push_back((float)((i + 1) * capture_interval_value));

				m_render_window->set_render_dirty(true);
				// Requesting the capture and the capture will start as soon as the render is effectively reset
				// (because set_render_dirty doesn't immediately reset the render, it merely asks for a reset
				// but the render window will only really reset once the current frame has finished rendering)
				capture_requested = true;
			}

			if (ref_image.width == 0)
				ImGuiRenderer::add_warning("No reference image selected");

			ImGui::EndDisabled();
		}

		if (capture_requested && render_settings.sample_number == 0)
		{
			// Starting the capture only once the render has effectively restarted
			capture_started = true;
			capture_requested = false;
		}

		float current_ratio = 0.0f;
		if (capture_interval_type == 0)
			// Time-based capture
			current_ratio = std::floor(m_render_window->get_current_render_time_ms() / 1000.0f / capture_interval_value);
		else
		{
			// Sample-based capture

			if (render_settings.accumulate)
				current_ratio = std::floor((float)render_settings.sample_number / capture_interval_value);
			else
				current_ratio = std::floor((float)total_samples_rendered++ / capture_interval_value);
		}

		if (capture_started && current_ratio > last_captured_ratio && captures_taken < number_of_captures)
		{
			// Time to capture!
			last_captured_ratio = current_ratio;
			captures_taken++;

			Image32Bit current_image(m_render_window->get_screenshoter()->get_image(), 3);

			float error;
			switch (error_metric_type)
			{
				case 0:
					// MSE
					error = Utils::compute_image_mse(ref_image, current_image);
					break;

				case 1:
					// RMSE
					error = Utils::compute_image_rmse(ref_image, current_image);
					break;

				case 2:
				{
					// FLIP
					float* error_map = nullptr;
					error = Utils::compute_image_weighted_median_FLIP(ref_image, current_image, &error_map);

					free(error_map);

					break;
				}
			default:
				break;
			}

			current_captured_errors.push_back(error);
			current_recorded_ys.push_back(error);

			if (captures_taken == number_of_captures)
				capture_started = false;
		}

		if (!capture_started && captures_taken == number_of_captures)
		{
			ImGui::Text("Capture done! Add it to the graph with a legend name");

			float min_error = 10000.0f;
			float max_error = 0.0f;
			for (int i = 0; i < number_of_captures; i++)
			{
				min_error = std::min(min_error, current_recorded_ys.at(i));
				max_error = std::max(max_error, current_recorded_ys.at(i));
			}

			ImGui::Text("Min / max error: %f / %f", min_error, max_error);
		}

		ImGui::TreePop();



		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Text("Step 5: Add capture data to graph");
		ImGui::TreePush("Add capture data tree");

		static std::string legend = "Data legend";
		ImGui::InputText("Legend", &legend);
		if (ImGui::Button("Add Step 4 captured data"))
		{
			recorded_legends.push_back(legend);
			recorded_xs_list.push_back(current_recorded_xs);
			recorded_ys_list.push_back(current_recorded_ys);
		}
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		ImGui::SeparatorText("Recorded data");
		ImGui::TreePush("Recorded data tree");
		for (int i = 0; i < recorded_legends.size(); i++)
		{
			std::string& leg = recorded_legends.at(i);
			ImGui::Text("- "); ImGui::SameLine();
			ImGui::InputText(std::string("##" + std::to_string(i)).c_str(), &leg); ImGui::SameLine();
			if (ImGui::Button(std::string("Delete##" + std::to_string(i)).c_str()))
			{
				recorded_legends.erase(recorded_legends.begin() + i);
				recorded_xs_list.erase(recorded_xs_list.begin() + i);
				recorded_ys_list.erase(recorded_ys_list.begin() + i);
			}

			ImGui::BeginDisabled(i == 0);
			ImGui::SameLine();
			if (ImGui::ArrowButton(std::string("Up##" + std::to_string(i)).c_str(), ImGuiDir_Up))
			{
				std::swap(recorded_legends.at(i), recorded_legends.at(i - 1));
				std::swap(recorded_xs_list.at(i), recorded_xs_list.at(i - 1));
				std::swap(recorded_ys_list.at(i), recorded_ys_list.at(i - 1));
			}
			ImGui::EndDisabled();

			ImGui::BeginDisabled(i == recorded_legends.size() - 1);
			ImGui::SameLine();
			if (ImGui::ArrowButton(std::string("Down##" + std::to_string(i)).c_str(), ImGuiDir_Down))
			{
				std::swap(recorded_legends.at(i), recorded_legends.at(i + 1));
				std::swap(recorded_xs_list.at(i), recorded_xs_list.at(i + 1));
				std::swap(recorded_ys_list.at(i), recorded_ys_list.at(i + 1));
			}
			ImGui::EndDisabled();
		}

		for (int i = 0; i < recorded_legends.size(); i++)
		{
			float min_error = 100000.0f;
			float max_error = 0.0f;

			for (int j = 0; j < number_of_captures; j++)
			{
				min_error = std::min(min_error, recorded_ys_list.at(i).at(j));
				max_error = std::max(max_error, recorded_ys_list.at(i).at(j));
			}

			ImGui::TreePush(std::string("Min max error tree ##" + std::to_string(i)).c_str());
			ImGui::Text(" Min / max error: %f / %f", min_error, max_error);
			ImGui::TreePop();
		}
		ImGui::TreePop();

		ImGui::TreePop();

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::SeparatorText("Current graph:");

		ImGui::InputFloat("Line weight", &m_convergence_graph_widget.get_line_weight());
		ImGui::SliderInt("Plot width", &m_convergence_graph_widget.get_plot_width(), 1, 1000);
		ImGui::SliderInt("Plot height", &m_convergence_graph_widget.get_plot_height(), 1, 1000);
		ImGui::InputText("Plot title", &m_convergence_graph_widget.get_plot_title());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		if (ImGui::Button("Screenshot graph"))
			m_convergence_graph_widget.request_screenshot(true, true);
		ImGui::SameLine();
		if (ImGui::Button("Copy graph to clipboard"))
			m_convergence_graph_widget.request_screenshot(true, false);

		std::string x_axis_name = (capture_interval_type == 0) ? "Time (s)" : "Samples";
		std::string y_axis_name = (error_metric_type == 0) ? "MSE" : (error_metric_type == 1) ? "RMSE" : "Mean FLIP Error";

		m_convergence_graph_widget.set_x_axis_name(x_axis_name);
		m_convergence_graph_widget.set_y_axis_name(y_axis_name);
		m_convergence_graph_widget.draw();

		ImGui::TreePop();
	}
}

ImGuiConvergenceGraphWidget& ImGuiToolsWindow::get_convergence_graph_widget()
{
	return m_convergence_graph_widget;
}
