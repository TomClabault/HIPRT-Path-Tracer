/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

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

		ImGui::SeparatorText("Hemispherical Directional Albedo");

		static GGXHemisphericalAlbedoSettings ggx_hemispherical_settings;

		static bool filename_modified = false;
		static std::string output_filename;

		ImGui::InputInt("Texture Size", &ggx_hemispherical_settings.texture_size);
		ImGui::InputInt("Integration Sample Count", &ggx_hemispherical_settings.integration_sample_count);
		if (ImGui::InputText("Output Texture Filename", &output_filename))
			filename_modified = true;

		if (!filename_modified)
			// As long as the user hasn't touched the output filename,
			// we modify it automatically so that's its more convenient
			output_filename = "GGX_Ess_" + std::to_string(ggx_hemispherical_settings.texture_size) + "x" + std::to_string(ggx_hemispherical_settings.texture_size) + ".hdr";

		std::shared_ptr<GPUBaker> baker = m_render_window->get_baker();

		static bool ggx_hemispherical_albedo_baking = false;
		ImGui::BeginDisabled(ggx_hemispherical_albedo_baking);
		if (ImGui::Button("Bake!"))
		{
			ggx_hemispherical_albedo_baking = true;

			// This starts the baking job asynchronously and the texture is
			// automatically written to disk when the baking is done
		 	baker->bake_ggx_hemispherical_albedo(ggx_hemispherical_settings);
		}
		ImGui::EndDisabled();

		static std::string baking_text = "";
		if (ggx_hemispherical_albedo_baking)
		{
			baking_text = " Baking...";
			if (baker->is_ggx_hemispherical_albedo_bake_complete())
			{
				baking_text = " Baking complete!";

				Image32Bit result = baker->get_bake_ggx_hemispherical_albedo_result();
				result.write_image_hdr(output_filename.c_str(), false);

				ggx_hemispherical_albedo_baking = false;
			}
		}

		ImGui::SameLine();
		ImGui::Text(baking_text.c_str());

		ImGui::TreePop();
	}
}
