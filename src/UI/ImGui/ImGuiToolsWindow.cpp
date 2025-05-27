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

void ImGuiToolsWindow::draw_image_difference_panel()
{
	if (ImGui::CollapsingHeader("Image difference"))
	{
		ImGui::TreePush("Image difference tree");

		ImGui::SeparatorText("RMSE");

		const char* filters[] = {"*.png", "*.jpg"};

		static std::string reference_image = "";
		static std::string subject_image = "";
		if (ImGui::Button("Select reference image"))
			reference_image = Utils::open_file_dialog(filters, 2);
		if (ImGui::Button("Select subject image"))
			subject_image = Utils::open_file_dialog(filters, 2);

		std::cout << reference_image << std::endl;

		ImGui::TreePop();
	}
}
