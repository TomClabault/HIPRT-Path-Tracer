/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/Baker/GGXHemisphericalAlbedoSettings.h"
#include "Renderer/Baker/GGXGlassHemisphericalAlbedoSettings.h"
#include "UI/ImGui/ImGuiBakingWindow.h"
#include "UI/RenderWindow.h"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

const char* ImGuiBakingWindow::TITLE = "Baking";

void ImGuiBakingWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;

	m_renderer = m_render_window->get_renderer();
}

void ImGuiBakingWindow::draw()
{
	ImGui::Begin(ImGuiBakingWindow::TITLE);

	ImGui::PushItemWidth(16 * ImGui::GetFontSize());

	draw_ggx_energy_conservation_panel();

	ImGui::PopItemWidth();

	ImGui::End();
}

void ImGuiBakingWindow::draw_ggx_energy_conservation_panel()
{
	if (ImGui::CollapsingHeader("GGX Energy Conservation"))
	{
		ImGui::TreePush("Baking GGX Energy Conservation tree");

		draw_GGX_E();
		draw_GGX_glass_E();

		/*static std::vector<float> roughnesses = { 1.0f };
		static std::vector<float> iors = { 1.3f, 1.5f, 2.0f, 2.5f };
		static bool cooking = false;
		static bool next_step_ready = true;
		static int step = -1;
		int nb_steps = roughnesses.size() * iors.size();

		if (ImGui::Button("Start screenhsotting"))
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

				std::vector<RendererMaterial> materials = m_renderer->get_materials();
				materials[0].ior = iors[step % iors.size()];
				materials[0].roughness = roughnesses[step / iors.size()];
				materials[0].make_safe();

				m_renderer->update_materials(materials);
				m_render_window->set_render_dirty(true);
			}
			else
			{
				if (m_render_window->is_rendering_done() && m_renderer->get_render_settings().sample_number > m_renderer->get_render_settings().adaptive_sampling_min_samples)
				{
					std::string filename = "Screenshot" + std::to_string(roughnesses[step / iors.size()]) + "x" + std::to_string(iors[step % iors.size()]) + " - " +std::to_string(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_COS_THETA_O) + "x" + std::to_string(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_ROUGHNESS) + "x" + std::to_string(GPUBakerConstants::GGX_GLASS_ESS_TEXTURE_SIZE_IOR) + "x" + ".png";
					m_render_window->get_screenshoter()->write_to_png(filename);

					next_step_ready = true;
					if (step == nb_steps - 1)
						cooking = false;
				}
			}
		}*/

		ImGui::TreePop();
	}
}

/**
 * Panel for the GGX hemispherical albedo
 */
void ImGuiBakingWindow::draw_GGX_E()
{
	if (ImGui::CollapsingHeader("Hemispherical Directional Albedo E(wo)"))
	{
		ImGui::TreePush("GGX_E tree");

		static GGXHemisphericalAlbedoSettings ggx_hemispherical_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size - Cos Theta", &ggx_hemispherical_settings.texture_size_cos_theta);
		ImGui::InputInt("Texture Size - Roughness", &ggx_hemispherical_settings.texture_size_roughness);
		ImGui::InputInt("Integration Sample Count", &ggx_hemispherical_settings.integration_sample_count);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = GPUBakerConstants::get_GGX_Ess_filename(ggx_hemispherical_settings.texture_size_cos_theta, ggx_hemispherical_settings.texture_size_roughness);

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool bake_started_at_least_once = false;
		ImGui::BeginDisabled(!baker->is_ggx_hemispherical_albedo_bake_complete() && bake_started_at_least_once);
		if (ImGui::Button("Bake!"))
		{
			bake_started_at_least_once = true;
			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
			baker->bake_ggx_hemispherical_albedo(ggx_hemispherical_settings, output_filename);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (!baker->is_ggx_hemispherical_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking...";
		else if (baker->is_ggx_hemispherical_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking complete!";

		ImGui::SameLine();
		ImGui::Text(baking_text.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}

/**
 * Panel for the GGX hemispherical albedo over the sphere for
 * glass material energy conservation
 */
void ImGuiBakingWindow::draw_GGX_glass_E()
{
	if (ImGui::CollapsingHeader("Glass Directional Albedo E_glass(wo)"))
	{
		ImGui::TreePush("GGX_E_glass tree");

		static GGXGlassHemisphericalAlbedoSettings ggx_glass_hemispherical_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size - Cos Theta", &ggx_glass_hemispherical_settings.texture_size_cos_theta_o);
		ImGui::InputInt("Texture Size - Roughness", &ggx_glass_hemispherical_settings.texture_size_roughness);
		ImGui::InputInt("Texture Size - IOR", &ggx_glass_hemispherical_settings.texture_size_ior);
		ImGui::InputInt("Integration Sample Count", &ggx_glass_hemispherical_settings.integration_sample_count);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = GPUBakerConstants::get_GGX_glass_Ess_filename(ggx_glass_hemispherical_settings.texture_size_cos_theta_o, ggx_glass_hemispherical_settings.texture_size_roughness, ggx_glass_hemispherical_settings.texture_size_ior);

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool bake_started_at_least_once = false;
		ImGui::BeginDisabled(!baker->is_ggx_glass_hemispherical_albedo_bake_complete() && bake_started_at_least_once);
		if (ImGui::Button("Bake!"))
		{
			bake_started_at_least_once = true;
			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
			baker->bake_ggx_glass_hemispherical_albedo(ggx_glass_hemispherical_settings, output_filename);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (!baker->is_ggx_glass_hemispherical_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking...";
		else if (baker->is_ggx_glass_hemispherical_albedo_bake_complete() && bake_started_at_least_once)
			baking_text = " Baking complete!";

		ImGui::SameLine();
		ImGui::Text(baking_text.c_str());

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}
}