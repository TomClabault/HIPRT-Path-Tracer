/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ImGui/ImGuiObjectsWindow.h"
#include "UI/RenderWindow.h"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

const char* ImGuiObjectsWindow::TITLE = "Objects";

struct MaterialOverrideState
{
	bool override_base_color = false;
	bool override_roughness = false;
	bool override_anisotropy = false;
	bool override_anisotropy_rotation = false;

	bool override_specular = false;
	bool override_specular_color = false;
	bool override_specular_tint_strength = false;
	bool override_specular_darkening = false;
	
	bool override_metallic = false;
	bool override_F82_reflectivity = false;
	bool override_F90_reflectivity = false;
	bool override_F90_falloff_exponent = false;
	bool override_second_roughness_weight = false;
	bool override_second_roughness = false;

	bool override_sheen_strength = false;
	bool override_sheen_color = false;
	bool override_sheen_roughness = false;

	bool override_coat_strength = false;
	bool override_coat_medium_absorption = false;
	bool override_coat_medium_thickness = false;
	bool override_coat_roughness = false;
	bool override_coat_roughening = false;
	bool override_coat_darkening = false;
	bool override_coat_anisotropy = false;
	bool override_coat_anisotropy_rotation = false;
	bool override_coat_IOR = false;

	bool override_transmission = false;
	bool override_diffuse_transmission = false;
	bool override_IOR = false;
	bool override_absorption_distance = false;
	bool override_absorption_color = false;
	bool override_dispersion_abbe = false;
	bool override_dispersion_scale = false;
	bool override_dielectric_priority = false;
	bool override_thin_material = false;

	bool override_thin_film = false;
	bool override_thin_film_thickness = false;
	bool override_thin_film_ior = false;
	bool override_thin_film_do_ior_override = false;
	bool override_thin_film_base_ior_override = false;
	bool override_thin_film_kappa_3 = false;
	bool override_thin_film_hue_shift = false;

	bool override_emission = false;
	bool override_emission_strength = false;

	bool override_opacity = false;
};

void ImGuiObjectsWindow::set_render_window(RenderWindow* render_window)
{
	m_render_window = render_window;

	m_renderer = m_render_window->get_renderer();
}

void ImGuiObjectsWindow::draw()
{
	ImGui::Begin(ImGuiObjectsWindow::TITLE);

	draw_global_objects_panel();
	draw_objects_panel();

	ImGui::End();
}

template <typename T>
void apply_material_override(bool override_flag, T CPUMaterial::* property, const T& override_value, std::vector<CPUMaterial>& materials_to_override)
{
	if (override_flag)
		for (CPUMaterial& renderer_mat : materials_to_override)
			renderer_mat.*property = override_value;
}

bool draw_material_override_line_common(bool& override_state_bool)
{
	ImGui::TableSetColumnIndex(0);
	float column_width = ImGui::GetColumnWidth();
	float radio_width = ImGui::GetFrameHeight();
	float padding = (column_width - radio_width) / 2.0f;

	static unsigned long long int checkbox_id = 0;
	ImGui::SetCursorPosX(ImGui::GetCursorPosX() + padding);
	ImGui::PushID(reinterpret_cast<long long int>(&override_state_bool));
	bool changed = ImGui::Checkbox("##checkbox_mat_override", &override_state_bool);
	ImGui::PopID();

	return changed;
}

bool draw_material_override_line(const std::string& text, bool& override_state_bool, float& material_override_property, float v_min, float v_max, const char* format = "%.3f")
{
	bool changed = draw_material_override_line_common(override_state_bool);

	ImGui::TableSetColumnIndex(1);
	changed |= ImGui::SliderFloat(text.c_str(), &material_override_property, v_min, v_max, format);

	return changed;
}

bool draw_material_override_line(const std::string& text, bool& override_state_bool, int& material_override_property, int v_min, int v_max)
{
	bool changed = draw_material_override_line_common(override_state_bool);

	ImGui::TableSetColumnIndex(1);
	changed |= ImGui::SliderInt(text.c_str(), &material_override_property, v_min, v_max);

	return changed;
}

bool draw_material_override_line(const std::string& text, bool& override_state_bool, ColorRGB32F& material_override_property)
{
	bool changed = draw_material_override_line_common(override_state_bool);

	ImGui::TableSetColumnIndex(1);
	changed |= ImGui::ColorEdit3(text.c_str(), (float*)&material_override_property);

	return changed;
}

bool draw_material_override_line(const std::string& text, bool& override_state_bool, bool& material_override_property)
{
	bool changed = draw_material_override_line_common(override_state_bool);

	ImGui::TableSetColumnIndex(1);
	changed |= ImGui::Checkbox(text.c_str(), &material_override_property);

	return changed;
}

void ImGuiObjectsWindow::draw_global_objects_panel()
{
	if (!ImGui::CollapsingHeader("Global material overrider"))
		return;

	ImGui::TreePush("Global material overrider tree");

	std::vector<const char*> items = { "- None", "- Lambertian BRDF", "- Oren Nayar BRDF", "- Principled BSDF" };
	if (ImGui::Combo("All Objects BSDF Override", m_renderer->get_global_compiler_options()->get_raw_pointer_to_macro_value(GPUKernelCompilerOptions::BSDF_OVERRIDE), items.data(), items.size()))
	{
		m_renderer->recompile_kernels();

		m_render_window->set_render_dirty(true);
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	std::shared_ptr<GPUKernelCompilerOptions> kernel_options = m_renderer->get_global_compiler_options();
	static CPUMaterial material_override;
	static MaterialOverrideState override_state;

	ImGui::PushItemWidth(16 * ImGui::GetFontSize());


	bool material_override_changed = false;

	static bool override_all = false;
	if (ImGui::Checkbox("Override All", &override_all))
	{
		std::memset(&override_state, override_all, sizeof(MaterialOverrideState));
		material_override_changed = true;
	}
	if (ImGui::CollapsingHeader("Base Layer"))
	{
		ImGui::TreePush("Base layer material tree");

		if (ImGui::BeginTable("Table base layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 6; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Base color", override_state.override_base_color, material_override.base_color);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Roughness", override_state.override_roughness, material_override.roughness, 0.0f, 1.0f);
					break;

				case 3:
					material_override_changed |= draw_material_override_line("Anisotropy", override_state.override_anisotropy, material_override.anisotropy, 0.0f, 1.0f);
					break;

				case 4:
					material_override_changed |= draw_material_override_line("Anisotropy rotation", override_state.override_anisotropy_rotation, material_override.anisotropy_rotation, 0.0f, 1.0f);
					break;

				case 5:
					material_override_changed |= draw_material_override_line("IOR", override_state.override_IOR, material_override.ior, 1.0f, 3.0f);
					if (material_override.ior < 1.0f || material_override.ior > 3.0f && (material_override.do_glass_energy_compensation || material_override.do_specular_energy_compensation))
					{
						ImGui::SameLine();
						ImGuiRenderer::show_help_marker("Energy compensation behavior is undefined for IORs < 1.0f or IORs > 3.0f", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
					}

					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Specular layer"))
	{
		ImGui::TreePush("Specular layer material tree");

		if (ImGui::BeginTable("Table specular layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 5; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Specular", override_state.override_specular, material_override.specular, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Specular color", override_state.override_specular_color, material_override.specular_color);
					break;

				case 3:
					material_override_changed |= draw_material_override_line("Specular tint strength", override_state.override_specular_tint_strength, material_override.specular_tint, 0.0f, 1.0f);
					break;

				case 4:
					material_override_changed |= draw_material_override_line("Specular darkening", override_state.override_specular_darkening, material_override.specular_darkening, 0.0f, 1.0f);
					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Metallic Layer"))
	{
		ImGui::TreePush("Metallic layer material tree");

		if (ImGui::BeginTable("Table metallic layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 8; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Metallic", override_state.override_metallic, material_override.metallic, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("F0 Reflectivity", override_state.override_base_color, material_override.base_color);
					ImGuiRenderer::show_help_marker("Reflectivity color at 0 degree angles: microfacet-normal "
						"and view direction perfectly aligned: looking straigth into "
						"the object.");

					break;

				case 3:
					material_override_changed |= draw_material_override_line("F82 Reflectivity", override_state.override_F82_reflectivity, material_override.metallic_F82);
					ImGuiRenderer::show_help_marker("Reflectivity color at 82 degree angles: microfacet-normal "
						"and view direction almost orthogonal.");

					break;

				case 4:
					material_override_changed |= draw_material_override_line("F90 Reflectivity", override_state.override_F90_reflectivity, material_override.metallic_F90);
					ImGuiRenderer::show_help_marker("Reflectivity color at 90 degree angles: microfacet-normal "
						"and view direction perfectly orthogonal.");

					break;

				case 5:
					material_override_changed |= draw_material_override_line("F90 Falloff exponent", override_state.override_F90_falloff_exponent, material_override.metallic_F90_falloff_exponent, 0.5f, 5.0f);
					ImGuiRenderer::show_help_marker("The \"falloff\" controls how wide the influence of F90 is.\n"
						"\n"
						"The lower the value, the wider F90's effect will be.");

					break;

				case 6:
					ImGui::Dummy(ImVec2(0.0f, 20.0f));

					material_override_changed |= draw_material_override_line("Second roughness weight", override_state.override_second_roughness_weight, material_override.second_roughness_weight, 0.0f, 1.0f);
					ImGuiRenderer::show_help_marker("The principled BSDF can have two metal lobes. They are exactly the "
						"same (F0/F82/F90, Anisotropy, ...) except that they can each have "
						"their own roughness.\n"
						"The first metal lobe's roughness is controlled by the general "
						"roughness of the material and the second metal lobe's roughness "
						"is controlled by 'Second roughness'.\n"
						"The two lobes are then linearly blended together using "
						"'Second roughness weight'. 'Second roughness weight' = 1 means "
						"that the primary roughness of the material is ignored and there "
						"is effectively only the second metallic lobe left.");

					break;

				case 7:
					material_override_changed |= draw_material_override_line("Second roughness", override_state.override_second_roughness, material_override.second_roughness, 0.0f, 1.0f);
					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Sheen Layer"))
	{
		ImGui::TreePush("Sheen layer material tree");

		if (ImGui::BeginTable("Table sheen layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 4; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Sheen strength", override_state.override_sheen_strength, material_override.sheen, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Sheen color", override_state.override_sheen_color, material_override.sheen_color);
					break;

				case 3:
					material_override_changed |= draw_material_override_line("Sheen roughness", override_state.override_sheen_roughness, material_override.sheen_roughness, 0.0f, 1.0f);
					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Coat Layer"))
	{
		ImGui::TreePush("Coat layer material tree");

		if (ImGui::BeginTable("Table coat layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 10; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Coat strength", override_state.override_coat_strength, material_override.coat, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Coat medium absorption", override_state.override_coat_medium_absorption, material_override.coat_medium_absorption);
					break;

				case 3:
					material_override_changed |= draw_material_override_line("Coat medium thickness", override_state.override_coat_medium_thickness, material_override.coat_medium_thickness, 0.0f, 15.0f);
					break;

				case 4:
					material_override_changed |= draw_material_override_line("Coat roughness", override_state.override_coat_roughness, material_override.coat_roughness, 0.0f, 1.0f);
					break;

				case 5:
					material_override_changed |= draw_material_override_line("Coat roughening", override_state.override_coat_roughening, material_override.coat_roughening, 0.0f, 1.0f);
					ImGuiRenderer::show_help_marker("Physical accuracy requires that a rough clearcoat also roughens what's underneath it "
						"i.e. the specular/metallic/transmission layers.\n"
						"The option is however given here to artistically disable "
						"that behavior by using coat roughening = 0.0f.");

					break;

				case 6:
					material_override_changed |= draw_material_override_line("Coat darkening", override_state.override_coat_darkening, material_override.coat_darkening, 0.0f, 1.0f);
					ImGuiRenderer::show_help_marker("Because of the total internal reflection that can happen inside the coat layer (i.e. "
						"light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the clearcoat will appear will increased "
						"saturation.\n\n"
						""
						"This parameter controls the strength of that darkening/increase in saturation.\n"
						"0.0f disables the effect which is non-physically accurate but may be artistically desirable.");
					break;

				case 7:
					material_override_changed |= draw_material_override_line("Coat anisotropy", override_state.override_coat_anisotropy, material_override.coat_anisotropy, 0.0f, 1.0f);
					break;

				case 8:
					material_override_changed |= draw_material_override_line("Coat anisotropy rotation", override_state.override_anisotropy_rotation, material_override.anisotropy_rotation, 0.0f, 1.0f);
					break;

				case 9:
					material_override_changed |= draw_material_override_line("Coat IOR", override_state.override_coat_IOR, material_override.coat_ior, 1.0f, 3.0f);
					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Transmission Layer"))
	{
		ImGui::TreePush("Transmission layer material tree");

		if (ImGui::BeginTable("Table transmission layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 10; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Diffuse transmission", override_state.override_diffuse_transmission, material_override.diffuse_transmission, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Specular transmission", override_state.override_transmission, material_override.specular_transmission, 0.0f, 1.0f);
					break;

				case 3:
					material_override_changed |= draw_material_override_line("IOR", override_state.override_IOR, material_override.ior, 1.0f, 3.0f);
					if (material_override.ior < 1.0f || material_override.ior > 3.0f && (material_override.do_glass_energy_compensation || material_override.do_specular_energy_compensation))
					{
						ImGui::SameLine();
						ImGuiRenderer::show_help_marker("Energy compensation behavior is undefined for IORs < 1.0f or IORs > 3.0f", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
					}

					break;

				case 4:
					material_override_changed |= draw_material_override_line("Absorption distance", override_state.override_absorption_distance, material_override.absorption_at_distance, 0.0f, 20.0f);
					break;

				case 5:
					material_override_changed |= draw_material_override_line("Absorption color", override_state.override_absorption_color, material_override.absorption_color);
					break;

				case 6:
					material_override_changed |= draw_material_override_line("Dispersion Abbe number", override_state.override_dispersion_abbe, material_override.dispersion_abbe_number, 9.0f, 91.0f);
					ImGuiRenderer::show_help_marker("Abbe number for the dispersion of the glass. The lower the number, the stronger the dispersion.");
					break;

				case 7:
					material_override_changed |= draw_material_override_line("Dispersion scale", override_state.override_dispersion_scale, material_override.dispersion_scale, 0.0f, 1.0f);
					break;

				case 8:
					material_override_changed |= draw_material_override_line("Dielectric priority", override_state.override_dielectric_priority, material_override.dielectric_priority, 1, StackPriorityEntry::PRIORITY_MAXIMUM);
					break;

				case 9:
					material_override_changed |= draw_material_override_line("Thin walled", override_state.override_thin_material, material_override.thin_walled);
					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Thin-Film Layer"))
	{
		ImGui::TreePush("Thin-film layer material tree");

		if (ImGui::BeginTable("Table thin-film layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 8; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Thin film", override_state.override_thin_film, material_override.thin_film, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Thin film thickness", override_state.override_thin_film_thickness, material_override.thin_film_thickness, 1.0f, 3.0f, "%.3f nm");
					break;

				case 3:
					material_override_changed |= draw_material_override_line("Thin film IOR", override_state.override_thin_film_ior, material_override.thin_film_ior, 1.0f, 3.0f);
					break;

				case 4:
					material_override_changed |= draw_material_override_line("Thin film hue shift", override_state.override_thin_film_hue_shift, material_override.thin_film_hue_shift_degrees, 0.0f, 360.0f);
					break;

				case 5:
					ImGui::Dummy(ImVec2(0.0f, 20.0f));

					material_override_changed |= draw_material_override_line("Override material IOR", override_state.override_thin_film_do_ior_override, material_override.thin_film_do_ior_override);

					// BeginDisabled for the cases that follow
					ImGui::BeginDisabled(!material_override.thin_film_do_ior_override);
					break;

				case 6:
					material_override_changed |= draw_material_override_line("Eta IOR override", override_state.override_thin_film_base_ior_override, material_override.thin_film_base_ior_override, 1.0f, 3.0f);
					ImGuiRenderer::show_help_marker("Overrides the eta parameter of the IOR of the base material. This is not physically based but allows for better artistic control.");
					break;

				case 7:
					material_override_changed |= draw_material_override_line("Kappa IOR override", override_state.override_thin_film_kappa_3, material_override.thin_film_kappa_3, 0.0f, 5.0f);
					ImGuiRenderer::show_help_marker("Overrides the kappa parameter (extinction coefficient) of the base material. This is not physically based but allows for better artistic control.");

					// BeginDisabled in "case 4:" and we're guaranteed to go through all cases one by one
					ImGui::EndDisabled();

					break;
				}
			}

			ImGui::EndTable();
		}
		
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Emission Properties"))
	{
		ImGui::TreePush("Emission material tree");

		if (ImGui::BeginTable("Table base layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 3; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Emission", override_state.override_emission, material_override.emission);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Emission strength", override_state.override_emission_strength, material_override.emission_strength, 0.0f, 10.0f);
					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Other properties"))
	{
		ImGui::TreePush("Other properties material tree");

		if (ImGui::BeginTable("Table base layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 3; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
					material_override_changed |= draw_material_override_line("Opacity", override_state.override_opacity, material_override.alpha_opacity, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Thin walled", override_state.override_thin_material, material_override.thin_walled);
					break;
				}
			}

			ImGui::EndTable();
		}

		ImGui::TreePop();
	}

	ImGui::PopItemWidth();

	if (material_override_changed)
	{
		std::vector<CPUMaterial> overriden_materials = m_renderer->get_original_materials();
		material_override.make_safe();

		apply_material_override(override_state.override_base_color, &CPUMaterial::base_color, material_override.base_color, overriden_materials);
		apply_material_override(override_state.override_roughness, &CPUMaterial::roughness, material_override.roughness, overriden_materials);
		apply_material_override(override_state.override_anisotropy, &CPUMaterial::anisotropy, material_override.anisotropy, overriden_materials);
		apply_material_override(override_state.override_anisotropy_rotation, &CPUMaterial::anisotropy_rotation, material_override.anisotropy_rotation, overriden_materials);

		apply_material_override(override_state.override_specular, &CPUMaterial::specular, material_override.specular, overriden_materials);
		apply_material_override(override_state.override_specular_color, &CPUMaterial::specular_color, material_override.specular_color, overriden_materials);
		apply_material_override(override_state.override_specular_tint_strength, &CPUMaterial::specular_tint, material_override.specular_tint, overriden_materials);
		apply_material_override(override_state.override_specular_darkening, &CPUMaterial::specular_darkening, material_override.specular_darkening, overriden_materials);

		apply_material_override(override_state.override_metallic, &CPUMaterial::metallic, material_override.metallic, overriden_materials);
		apply_material_override(override_state.override_F82_reflectivity, &CPUMaterial::metallic_F82, material_override.metallic_F82, overriden_materials);
		apply_material_override(override_state.override_F90_reflectivity, &CPUMaterial::metallic_F90, material_override.metallic_F90, overriden_materials);
		apply_material_override(override_state.override_F90_falloff_exponent, &CPUMaterial::metallic_F90_falloff_exponent, material_override.metallic_F90_falloff_exponent, overriden_materials);
		apply_material_override(override_state.override_second_roughness_weight, &CPUMaterial::second_roughness_weight, material_override.second_roughness_weight, overriden_materials);
		apply_material_override(override_state.override_second_roughness, &CPUMaterial::second_roughness, material_override.second_roughness, overriden_materials);

		apply_material_override(override_state.override_sheen_strength, &CPUMaterial::sheen, material_override.sheen, overriden_materials);
		apply_material_override(override_state.override_sheen_color, &CPUMaterial::sheen_color, material_override.sheen_color, overriden_materials);
		apply_material_override(override_state.override_sheen_roughness, &CPUMaterial::sheen_roughness, material_override.sheen_roughness, overriden_materials);

		apply_material_override(override_state.override_coat_strength, &CPUMaterial::coat, material_override.coat, overriden_materials);
		apply_material_override(override_state.override_coat_medium_absorption, &CPUMaterial::coat_medium_absorption, material_override.coat_medium_absorption, overriden_materials);
		apply_material_override(override_state.override_coat_medium_thickness, &CPUMaterial::coat_medium_thickness, material_override.coat_medium_thickness, overriden_materials);
		apply_material_override(override_state.override_coat_roughness, &CPUMaterial::coat_roughness, material_override.coat_roughness, overriden_materials);
		apply_material_override(override_state.override_coat_roughening, &CPUMaterial::coat_roughening, material_override.coat_roughening, overriden_materials);
		apply_material_override(override_state.override_coat_darkening, &CPUMaterial::coat_darkening, material_override.coat_darkening, overriden_materials);
		apply_material_override(override_state.override_coat_anisotropy, &CPUMaterial::coat_anisotropy, material_override.coat_anisotropy, overriden_materials);
		apply_material_override(override_state.override_coat_anisotropy_rotation, &CPUMaterial::coat_anisotropy_rotation, material_override.coat_anisotropy_rotation, overriden_materials);
		apply_material_override(override_state.override_coat_IOR, &CPUMaterial::coat_ior, material_override.coat_ior, overriden_materials);

		apply_material_override(override_state.override_transmission, &CPUMaterial::diffuse_transmission, material_override.diffuse_transmission, overriden_materials);
		apply_material_override(override_state.override_transmission, &CPUMaterial::specular_transmission, material_override.specular_transmission, overriden_materials);
		apply_material_override(override_state.override_IOR, &CPUMaterial::ior, material_override.ior, overriden_materials);
		apply_material_override(override_state.override_absorption_distance, &CPUMaterial::absorption_at_distance, material_override.absorption_at_distance, overriden_materials);
		apply_material_override(override_state.override_absorption_color, &CPUMaterial::absorption_color, material_override.absorption_color, overriden_materials);
		apply_material_override(override_state.override_dispersion_abbe, &CPUMaterial::dispersion_abbe_number, material_override.dispersion_abbe_number, overriden_materials);
		apply_material_override(override_state.override_dispersion_scale, &CPUMaterial::dispersion_scale, material_override.dispersion_scale, overriden_materials);
		apply_material_override(override_state.override_dielectric_priority, &CPUMaterial::dielectric_priority, material_override.dielectric_priority, overriden_materials);
		apply_material_override(override_state.override_thin_material, &CPUMaterial::thin_walled, material_override.thin_walled, overriden_materials);

		apply_material_override(override_state.override_thin_film, &CPUMaterial::thin_film, material_override.thin_film, overriden_materials);
		apply_material_override(override_state.override_thin_film_thickness, &CPUMaterial::thin_film_thickness, material_override.thin_film_thickness, overriden_materials);
		apply_material_override(override_state.override_thin_film_ior, &CPUMaterial::thin_film_ior, material_override.thin_film_ior, overriden_materials);
		apply_material_override(override_state.override_thin_film_do_ior_override, &CPUMaterial::thin_film_do_ior_override, material_override.thin_film_do_ior_override, overriden_materials);
		apply_material_override(override_state.override_thin_film_base_ior_override, &CPUMaterial::thin_film_base_ior_override, material_override.thin_film_base_ior_override, overriden_materials);
		apply_material_override(override_state.override_thin_film_kappa_3, &CPUMaterial::thin_film_kappa_3, material_override.thin_film_kappa_3, overriden_materials);
		apply_material_override(override_state.override_thin_film_hue_shift, &CPUMaterial::thin_film_hue_shift_degrees, material_override.thin_film_hue_shift_degrees, overriden_materials);

		// Special case for the emission since it's a private member
		apply_material_override(override_state.override_emission, &CPUMaterial::emission, material_override.emission, overriden_materials);
		apply_material_override(override_state.override_emission_strength, &CPUMaterial::emission_strength, material_override.emission_strength, overriden_materials);

		apply_material_override(override_state.override_opacity, &CPUMaterial::alpha_opacity, material_override.alpha_opacity, overriden_materials);

		m_renderer->update_all_materials(overriden_materials);
		m_render_window->set_render_dirty(true);
	}

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	ImGui::TreePop();
}

void ImGuiObjectsWindow::draw_objects_panel()
{
	if (!ImGui::CollapsingHeader("Per object settings"))
		return;

	ImGui::TreePush("Objects tree");

	std::vector<CPUMaterial> materials = m_renderer->get_current_materials();
	const std::vector<std::string>& material_names = m_renderer->get_material_names();
	const std::vector<std::string>& mesh_names = m_renderer->get_mesh_names();

	bool material_changed = false;
	static int currently_selected_material_index = 0;

	if (ImGui::CollapsingHeader("All objects"))
	{
		static std::string filter_string = "";
		// This set contains all the ids of materials that should be displayed in the
		// list box. This list is refined based on the search that the user has typed
		// in to filter the materials
		static std::unordered_set<int> filtered_material_indices;

		ImGui::TreePush("All objects tree");

		// This boolean variable is to decide whether or not we need to populate the
		// 'accepted_material_indices' set
		bool first_time = filter_string == "" && filtered_material_indices.size() == 0 && materials.size() > 0;
		if (ImGui::InputText("Search", &filter_string) || first_time)
			filtered_material_indices = filter_displayed_materials(materials.size(), material_names, mesh_names, filter_string);
		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		if (ImGui::BeginListBox("##all_objects", ImVec2(-FLT_MIN, 15 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (int material_index = 0; material_index < materials.size(); material_index++)
			{
				if (filter_string != "")
				{
					// The user has filtered the materials, checking if the current material
					// has been filtered out or not.
					//
					// The material isn't filtered out (it is accepted) if its index can be found
					// in the 'accepted_material_indices' set
					//
					// If not, the material has been filetered out
					if (filtered_material_indices.find(material_index) == filtered_material_indices.end())
						continue;
				}

				const bool is_selected = (currently_selected_material_index == material_index);
				std::string text = mesh_names[material_index] + " (" + material_names[material_index] + ")";
				if (ImGui::Selectable(text.c_str(), is_selected))
					currently_selected_material_index = material_index;

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
		static std::string filter_string = "";
		// This set contains all the ids of materials that should be displayed in the
		// list box. This list is refined based on the search that the user has typed
		// in to filter the materials
		static std::unordered_set<int> filtered_material_indices;

		ImGui::TreePush("Emissive objects tree");

		static float global_emissive_objects_factor = 1.0f;
		if (ImGui::SliderFloat("Global Emissive Objects Factor", &global_emissive_objects_factor, 0.0f, 10.0f))
		{
			for (CPUMaterial& material : materials)
			{
					material.global_emissive_factor = global_emissive_objects_factor;

					material.make_safe();
			}

			m_renderer->update_all_materials(materials);
			m_renderer->recompute_emissives_power_alias_table();
			m_render_window->set_render_dirty(true);
		}

		// This boolean variable is to decide whether or not we need to populate the
		// 'accepted_material_indices' set
		bool first_time = filter_string == "" && filtered_material_indices.size() == 0 && materials.size() > 0;
		if (ImGui::InputText("Search", &filter_string) || first_time)
			filtered_material_indices = filter_displayed_materials(materials.size(), material_names, mesh_names, filter_string);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));

		if (ImGui::BeginListBox("Emissive objects", ImVec2(-FLT_MIN, 7 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (int material_index = 0; material_index < materials.size(); material_index++)
			{
				if (!materials[material_index].is_emissive())
					continue;

				if (filter_string != "")
				{
					// The user has filtered the materials, checking if the current material
					// has been filtered out or not.
					//
					// The material isn't filtered out (it is accepted) if its index can be found
					// in the 'accepted_material_indices' set
					//
					// If not, the material has been filetered out
					if (filtered_material_indices.find(material_index) == filtered_material_indices.end())
						continue;
				}

				const bool is_selected = (currently_selected_material_index == material_index);
				if (ImGui::Selectable(material_names[material_index].c_str(), is_selected))
					currently_selected_material_index = material_index;

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
		CPUMaterial& material = materials[currently_selected_material_index];

		ImGui::PushItemWidth(16 * ImGui::GetFontSize());

		ImGui::Text("- "); ImGui::SameLine();
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.9f, 0.0f, 1.0f));
		ImGui::Text("Selected object"); ImGui::SameLine();
		ImGui::PopStyleColor();
		ImGui::Text(": "); ImGui::SameLine();
		ImGui::Text("%s", material_names[currently_selected_material_index].c_str());

		if (ImGui::CollapsingHeader("Base Layer"))
		{
			ImGui::TreePush("Base layer material tree");

			material_changed |= ImGui::ColorEdit3("Base color", (float*)&material.base_color);
			material_changed |= ImGui::SliderFloat("Roughness", &material.roughness, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Anisotropy", &material.anisotropy, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Anisotropy rotation", &material.anisotropy_rotation, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("IOR", &material.ior, 1.0f, 3.0f);
			if (material.ior < 1.0f || material.ior > 3.0f)
			{
				ImGui::SameLine();
				ImGuiRenderer::show_help_marker("Energy compensation behavior is undefined for IORs < 1.0f or IORs > 3.0f", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}

			ImGui::Dummy(ImVec2(0.0f, 20.0f));

			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Specular layer"))
		{
			ImGui::TreePush("Specular layer material tree");

			material_changed |= ImGui::SliderFloat("Specular", &material.specular, 0.0f, 1.0f);
			material_changed |= ImGui::ColorEdit3("Specular color", (float*)&material.specular_color);
			material_changed |= ImGui::SliderFloat("Specular tint strength", &material.specular_tint, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Specular darkening", &material.specular_darkening, 0.0f, 1.0f);
			ImGuiRenderer::show_help_marker("Same as coat darkening but for total internal reflection inside the specular layer "
				"that sits on top of the diffuse base.");
			if (material.do_specular_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled. This material option will have no effect.\n"
					"Energy compensation can be globally enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			else if (material.do_specular_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_SPECULAR_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled for the glossy layer (specular/diffuse). This material option will have no effect.\n"
					"Energy compensation can be enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			material_changed |= ImGui::Checkbox("Glossy layer energy compensation", &material.do_specular_energy_compensation);
			ImGuiRenderer::show_help_marker("Whether or not to do energy compensation for the glossy layer (specular/diffuse) lobe of this material.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Metallic Layer"))
		{
			ImGui::TreePush("Metallic layer material tree");

			material_changed |= ImGui::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
			material_changed |= ImGui::ColorEdit3("F0 Reflectivity", (float*)&material.base_color);
			ImGuiRenderer::show_help_marker("Reflectivity color at 0 degree angles: microfacet-normal "
				"and view direction perfectly aligned: looking straigth into "
				"the object.");
			material_changed |= ImGui::ColorEdit3("F82 Reflectivity", (float*)&material.metallic_F82);
			ImGuiRenderer::show_help_marker("Reflectivity color at 82 degree angles: microfacet-normal "
				"and view direction almost orthogonal.");
			material_changed |= ImGui::ColorEdit3("F90 Reflectivity", (float*)&material.metallic_F90);
			ImGuiRenderer::show_help_marker("Reflectivity color at 90 degree angles: microfacet-normal "
				"and view direction perfectly orthogonal.");
			material_changed |= ImGui::SliderFloat("F90 Falloff exponent", &material.metallic_F90_falloff_exponent, 0.5f, 5.0f);
			ImGuiRenderer::show_help_marker("The \"falloff\" controls how wide the influence of F90 is.\n"
				"\n"
				"The lower the value, the wider F90's effect will be.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			material_changed |= ImGui::SliderFloat("Second roughness weight", &material.second_roughness_weight, 0.0f, 1.0f);
			ImGuiRenderer::show_help_marker("The principled BSDF can have two metal lobes. They are exactly the "
				"same (F0/F82/F90, Anisotropy, ...) except that they can each have "
				"their own roughness.\n"
				"The first metal lobe's roughness is controlled by the general "
				"roughness of the material and the second metal lobe's roughness "
				"is controlled by 'Second roughness'.\n"
				"The two lobes are then linearly blended together using "
				"'Second roughness weight'. 'Second roughness weight' = 1 means "
				"that the primary roughness of the material is ignored and there "
				"is effectively only the second metallic lobe left.");

			ImGui::BeginDisabled(material.second_roughness_weight == 0.0f);
			material_changed |= ImGui::SliderFloat("Second roughness", &material.second_roughness, 0.0f, 1.0f);
			ImGui::EndDisabled();

			if (material.do_metallic_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled. This material option will have no effect.\n"
					"Energy compensation can be globally enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			else if (material.do_metallic_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_METALLIC_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled for the metallic layer. This material option will have no effect.\n"
					"Energy compensation can be enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			material_changed |= ImGui::Checkbox("Metallic layer energy compensation", &material.do_metallic_energy_compensation);
			ImGuiRenderer::show_help_marker("Whether or not to do energy compensation for the metallic layer of this material.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Sheen Layer"))
		{
			ImGui::TreePush("Sheen layer material tree");

			material_changed |= ImGui::SliderFloat("Sheen strength", &material.sheen, 0.0f, 1.0f);
			material_changed |= ImGui::ColorEdit3("Sheen color", (float*)&material.sheen_color);
			material_changed |= ImGui::SliderFloat("Sheen roughness", &material.sheen_roughness, 0.0f, 1.0f);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Coat Layer"))
		{
			ImGui::TreePush("Coat layer material tree");

			material_changed |= ImGui::SliderFloat("Coat strength", &material.coat, 0.0f, 1.0f);
			material_changed |= ImGui::ColorEdit3("Coat medium absorption", (float*)&material.coat_medium_absorption);
			material_changed |= ImGui::SliderFloat("Coat medium thickness", &material.coat_medium_thickness, 0.0f, 15.0f);
			material_changed |= ImGui::SliderFloat("Coat roughness", &material.coat_roughness, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Coat roughening", &material.coat_roughening, 0.0f, 1.0f);
			ImGuiRenderer::show_help_marker("Physical accuracy requires that a rough clearcoat also roughens what's underneath it "
				"i.e. the specular/metallic/transmission layers.\n"
				"The option is however given here to artistically disable "
				"that behavior by using coat roughening = 0.0f.");
			material_changed |= ImGui::SliderFloat("Coat darkening", &material.coat_darkening, 0.0f, 1.0f);
			ImGuiRenderer::show_help_marker("Because of the total internal reflection that can happen inside the coat layer (i.e. "
				"light bouncing between the coat/BSDF and air/coat interfaces), the BSDF below the clearcoat will appear will increased "
				"saturation.\n\n"
				""
				"This parameter controls the strength of that darkening/increase in saturation.\n"
				"0.0f disables the effect which is non-physically accurate but may be artistically desirable.");
			material_changed |= ImGui::SliderFloat("Coat anisotropy", &material.coat_anisotropy, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Coat anisotropy rotation", &material.coat_anisotropy_rotation, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Coat IOR", &material.coat_ior, 1.0f, 3.0f);
			if (material.do_coat_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled. This material option will have no effect.\n"
					"Energy compensation can be globally enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			else if (material.do_coat_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_CLEARCOAT_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled for the clearcoat layer. This material option will have no effect.\n"
					"Energy compensation can be enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			material_changed |= ImGui::Checkbox("Clearcoat layer energy compensation", &material.do_coat_energy_compensation);
			ImGuiRenderer::show_help_marker("Whether or not to do energy compensation for the clearcoat layer of this material.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Transmission Layer"))
		{
			ImGui::TreePush("Transmission layer material tree");

			material_changed |= ImGui::SliderFloat("Diffuse transmission", &material.diffuse_transmission, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Specular transmission", &material.specular_transmission, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("IOR", &material.ior, 1.0f, 3.0f);
			if (material.ior < 1.0f || material.ior > 3.0f && (material.do_glass_energy_compensation || material.do_specular_energy_compensation))
			{
				ImGui::SameLine();
				ImGuiRenderer::show_help_marker("Energy compensation behavior is undefined for IORs < 1.0f or IORs > 3.0f", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			material_changed |= ImGui::SliderFloat("Absorption distance", &material.absorption_at_distance, 0.0f, 20.0f);
			material_changed |= ImGui::ColorEdit3("Absorption color", (float*)&material.absorption_color);
			material_changed |= ImGui::SliderFloat("Dispersion Abbe number", &material.dispersion_abbe_number, 9.0f, 91.0f);
			ImGuiRenderer::show_help_marker("Abbe number for the dispersion of the glass. The lower the number, the stronger the dispersion.");
			material_changed |= ImGui::SliderFloat("Dispersion scale", &material.dispersion_scale, 0.0f, 1.0f);
			material_changed |= ImGui::SliderInt("Dielectric priority", &material.dielectric_priority, 1, StackPriorityEntry::PRIORITY_MAXIMUM);
			material_changed |= ImGui::Checkbox("Thin walled", &material.thin_walled);
			if (material.do_glass_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled. This material option will have no effect.\n"
					"Energy compensation can be globally enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			else if (material.do_glass_energy_compensation && kernel_options->get_macro_value(GPUKernelCompilerOptions::PRINCIPLED_BSDF_DO_GLASS_ENERGY_COMPENSATION) == KERNEL_OPTION_FALSE)
			{
				ImGui::Text("Warning: ");
				ImGuiRenderer::show_help_marker("Energy compensation is globally disabled for the glass layer. This material option will have no effect.\n"
					"Energy compensation can be enabled in \"Settings\" --> \"Sampling\" --> \"Materials\"", ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
			}
			material_changed |= ImGui::Checkbox("Glass layer energy compensation", &material.do_glass_energy_compensation);
			ImGuiRenderer::show_help_marker("Whether or not to do energy compensation for the glass layer of this material.");

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Thin-Film Layer"))
		{
			ImGui::TreePush("Thin film layer material tree");

			material_changed |= ImGui::SliderFloat("Thin film", &material.thin_film, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Thin film thickness", &material.thin_film_thickness, 0.0f, 2000.0f, "%.3f nm");
			material_changed |= ImGui::SliderFloat("Thin film IOR", &material.thin_film_ior, 1.0f, 3.0f);
			material_changed |= ImGui::SliderFloat("Thin film hue shift", &material.thin_film_hue_shift_degrees, 0.0f, 360.0f);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			material_changed |= ImGui::Checkbox("Override material IOR", &material.thin_film_do_ior_override);
			ImGui::BeginDisabled(!material.thin_film_do_ior_override);
			material_changed |= ImGui::SliderFloat("Eta IOR override", &material.thin_film_base_ior_override, 1.0f, 3.0f);
			ImGuiRenderer::show_help_marker("Overrides the eta parameter of the IOR of the base material. This is not physically based but allows for better artistic control.");
			material_changed |= ImGui::SliderFloat("Kappa IOR override", &material.thin_film_kappa_3, 0.0f, 5.0f);
			ImGuiRenderer::show_help_marker("Overrides the kappa parameter (extinction coefficient) of the base material. This is not physically based but allows for better artistic control.");
			ImGui::EndDisabled();

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Emission Properties"))
		{
			ImGui::TreePush("Emission material tree");

			bool emission_controlled_by_texture = material.emission_texture_index != MaterialConstants::NO_TEXTURE;
			ImGui::BeginDisabled(emission_controlled_by_texture);
			
			bool emission_changed = false;
			// TODO we would need to recompute the alias table for the emissive lights here
			emission_changed |= ImGui::ColorEdit3("Emission", (float*)&material.emission, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
			ImGui::EndDisabled();
			if (emission_controlled_by_texture)
				ImGuiRenderer::show_help_marker("Disabled because the emission of this material is controlled by a texture");

			// TODO we would need to recompute the alias table for the emissive lights here
			emission_changed |= ImGui::SliderFloat("Emission Strength", &material.emission_strength, 0.0f, 10.0f);

			material_changed |= emission_changed;
			if (emission_changed)
				m_renderer->get_NEE_plus_plus_render_pass()->reset(false);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Other properties"))
		{
			ImGui::TreePush("Other properties material tree");

			material_changed |= ImGui::SliderFloat("Opacity", &material.alpha_opacity, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			material_changed |= ImGui::Checkbox("Thin walled", &material.thin_walled);

			ImGui::EndDisabled();

			ImGui::TreePop();
		}

		ImGui::PopItemWidth();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Separator();

		material_changed |= draw_material_presets(material);

		if (material_changed)
		{
			material.make_safe();

			m_renderer->update_one_material(material, currently_selected_material_index);
			m_render_window->set_render_dirty(true);
		}
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));
}

std::unordered_set<int> ImGuiObjectsWindow::filter_displayed_materials(int material_count, const std::vector<std::string>& material_names, const std::vector<std::string>& mesh_names, const std::string& filter_string) const
{
	std::unordered_set<int> accepted_material_indices;
	if (filter_string == "")
	{
		// If no filter, all materials are accepted so we're adding them all to the set
		for (int i = 0; i < material_count; i++)
			accepted_material_indices.insert(i);

		return accepted_material_indices;
	}

	auto case_insensitive_string_find = [](const std::string& haystack, const std::string& needle)
	{
		auto found = std::search(
			haystack.begin(), haystack.end(), 
			needle.begin(), needle.end(), 
			[](unsigned char char1, unsigned char char2) { return std::toupper(char1) == std::toupper(char2); }
		);

		return found != haystack.end();
	};

	// Just pure brute force search...
	// Will improve if this ever becomes a serious bottleneck
	for (int material_index = 0; material_index < material_count; material_index++)
		if (case_insensitive_string_find(material_names[material_index], filter_string) || case_insensitive_string_find(mesh_names[material_index], filter_string))
			accepted_material_indices.insert(material_index);

	return accepted_material_indices;
}

bool ImGuiObjectsWindow::draw_material_presets(CPUMaterial& material)
{
	bool material_changed = false;

	ImGui::Dummy(ImVec2(0.0f, 20.0f));
	if (!ImGui::CollapsingHeader("Material presets"))
		return false;

	ImGui::TreePush("Materials presets tree");
	ImGui::Text("Metals");
	ImGui::Separator();

	// Reference: [Adobe Standard Material, Technical Documentation, Kutz, Hasan, Edmondson]
	const std::vector<std::pair<std::string, std::pair<ColorRGB32F, ColorRGB32F>>> names_to_f0_f82 = {
		{ "Silver", { ColorRGB32F(0.9868f, 0.9830f, 0.9667f), ColorRGB32F(0.9929f, 0.9961f, 1.0000f) } },
		{ "Aluminum", { ColorRGB32F(0.9157f, 0.9226f, 0.9236f), ColorRGB32F(0.9090f, 0.9365f, 0.9596f) } },
		{ "Gold", { ColorRGB32F(1.0000f, 0.7099f, 0.3148f), ColorRGB32F(0.9408f, 0.9636f, 0.9099f) } },
		{ "Chromium", { ColorRGB32F(0.5496f, 0.5561f, 0.5531f), ColorRGB32F(0.7372f, 0.7511f, 0.8170f) } },
		{ "Copper", { ColorRGB32F(1.0000f, 0.6504f, 0.5274f), ColorRGB32F(0.9755f, 0.9349f, 0.9301f) } },
		{ "Iron", { ColorRGB32F(0.8951f, 0.8755f, 0.8154f), ColorRGB32F(0.8551f, 0.8800f, 0.8966f) } },
		{ "Mercury", { ColorRGB32F(0.7815f, 0.7795f, 0.7783f), ColorRGB32F(0.8103f, 0.8532f, 0.9046f) } },
		{ "Magnesium", { ColorRGB32F(0.8918f, 0.8821f, 0.8948f), ColorRGB32F(0.8949f, 0.9147f, 0.9504f) } },
		{ "Nickel", { ColorRGB32F(0.7014f, 0.6382f, 0.5593f), ColorRGB32F(0.8134f, 0.8352f, 0.8725f) } },
		{ "Lead", { ColorRGB32F(0.7363f, 0.7023f, 0.6602f), ColorRGB32F(0.8095f, 0.8369f, 0.8739f) } },
		{ "Platinum", { ColorRGB32F(0.9602f, 0.9317f, 0.8260f), ColorRGB32F(0.9501f, 0.9461f, 0.9352f) } },
		{ "Titanium", { ColorRGB32F(0.4432f, 0.3993f, 0.3599f), ColorRGB32F(0.8627f, 0.9066f, 0.9481f) } },
		{ "Zinc", { ColorRGB32F(0.8759f, 0.8685f, 0.8542f), ColorRGB32F(0.8769f, 0.9037f, 0.9341f) } },
	};

	int line_count = 0;
	for (int i = 0; i < names_to_f0_f82.size(); i++)
	{
		ColorRGB32F F0 = names_to_f0_f82[i].second.first;
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(F0.r, F0.g, F0.b, 1.0f));
		// Changing text color from black to white depending on luminance for readability
		ImGui::PushStyleColor(ImGuiCol_Text, F0.luminance() > 0.6f ? ImVec4(0.0f, 0.0f, 0.0f, 1.0f) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

		if (ImGui::Button(names_to_f0_f82[i].first.c_str(), /* size */ ImVec2(6.0f * ImGui::GetFontSize(), 1.5f * ImGui::GetFontSize())))
		{
			material_changed = true;

			float original_roughness = material.roughness;

			// Resetting the material
			material = CPUMaterial();

			// Applying preset
			material.roughness = original_roughness;
			material.metallic = 1.0f;
			material.base_color = names_to_f0_f82[i].second.first;
			material.metallic_F82 = names_to_f0_f82[i].second.second;
		}

		ImGui::PopStyleColor();
		ImGui::PopStyleColor();

		line_count++;
		if (line_count == 5)
			line_count = 0;
		else
			ImGui::SameLine();
	}

	ImGui::TreePop();
	ImGui::Dummy(ImVec2(0.0f, 20.0f));

	return material_changed;
}
