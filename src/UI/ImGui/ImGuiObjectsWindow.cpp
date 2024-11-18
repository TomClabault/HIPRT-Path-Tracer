/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ImGui/ImGuiObjectsWindow.h"
#include "UI/RenderWindow.h"

#include "imgui.h"

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
	bool override_coat_anisotropy = false;
	bool override_coat_anisotropy_rotation = false;
	bool override_coat_IOR = false;

	bool override_transmission = false;
	bool override_IOR = false;
	bool override_absorption_distance = false;
	bool override_absorption_color = false;
	bool override_dielectric_priority = false;

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
void apply_material_override(bool override_flag, T RendererMaterial::* property, const T& override_value, std::vector<RendererMaterial>& materials_to_override) 
{
	if (override_flag)
		for (RendererMaterial& renderer_mat : materials_to_override)
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
	static SimplifiedRendererMaterial material_override;
	static ColorRGB32F material_emission = ColorRGB32F(0.0f);
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
					material_override_changed |= draw_material_override_line("Specular", override_state.override_specular, material_override.specular, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("Specular color", override_state.override_specular_color, material_override.specular_color);
					break;

				case 3:
					material_override_changed |= draw_material_override_line("Specular tint strength", override_state.override_specular_tint_strength, material_override.specular_tint, 0.0f, 1.0f);
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

		material_override_changed |= ImGui::ColorEdit3("Sheen color", (float*)&material_override.sheen_color);
		material_override_changed |= ImGui::SliderFloat("Sheen roughness", &material_override.sheen_roughness, 0.0f, 1.0f);

		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::TreePop();
	}

	if (ImGui::CollapsingHeader("Coat Layer"))
	{
		ImGui::TreePush("Coat layer material tree");

		if (ImGui::BeginTable("Table coat layer", 2, ImGuiTableFlags_SizingFixedFit))
		{
			for (int row = 0; row < 9; row++)
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
					material_override_changed |= draw_material_override_line("Coat anisotropy", override_state.override_coat_anisotropy, material_override.coat_anisotropy, 0.0f, 1.0f);
					break;

				case 7:
					material_override_changed |= draw_material_override_line("Coat anisotropy rotation", override_state.override_anisotropy_rotation, material_override.anisotropy_rotation, 0.0f, 1.0f);
					break;

				case 8:
					material_override_changed |= draw_material_override_line("Coat IOR", override_state.override_coat_IOR, material_override.coat_ior, 0.0f, 1.0f);
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
					material_override_changed |= draw_material_override_line("Transmission", override_state.override_transmission, material_override.specular_transmission, 0.0f, 1.0f);
					break;

				case 2:
					material_override_changed |= draw_material_override_line("IOR", override_state.override_IOR, material_override.ior, 1.0f, 3.0f);
					break;

				case 3:
					material_override_changed |= draw_material_override_line("Absorption distance", override_state.override_absorption_distance, material_override.absorption_at_distance, 0.0f, 20.0f);
					break;

				case 4:
					material_override_changed |= draw_material_override_line("Absorption color", override_state.override_absorption_color, material_override.absorption_color);
					break;

				case 5:
					ImGui::BeginDisabled(kernel_options->get_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITIES);
					material_override_changed |= draw_material_override_line("Dielectric priority", override_state.override_dielectric_priority, material_override.dielectric_priority, 1, StackPriorityEntry::PRIORITY_MAXIMUM);
					if (kernel_options->get_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITIES)
						ImGuiRenderer::show_help_marker("Disabled because not using nested dielectrics with priorities.");
					ImGui::EndDisabled();

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
				{
					if (material_override.emission_strength > 0.0f)
						material_emission = material_override.get_emission() / material_override.emission_strength;

					if (draw_material_override_line("Emission", override_state.override_emission, material_emission))
					{
						material_override.set_emission(material_emission / material_override.emission_strength);
						material_override_changed = true;
					}
					break;
				}

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
			for (int row = 0; row < 2; row++)
			{
				ImGui::TableNextRow();

				switch (row)
				{
				case 0:
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Override");
					break;

				case 1:
				{
					material_override_changed |= draw_material_override_line("Opacity", override_state.override_opacity, material_override.alpha_opacity, 0.0f, 1.0f);
					break;
				}
				}
			}

			ImGui::EndTable();
		}

		ImGui::TreePop();
	}

	ImGui::PopItemWidth();


	if (material_override_changed)
	{
		std::vector<RendererMaterial> overriden_materials = m_renderer->get_original_materials();

		apply_material_override(override_state.override_base_color, &RendererMaterial::base_color, material_override.base_color, overriden_materials);
		apply_material_override(override_state.override_roughness, &RendererMaterial::roughness, material_override.roughness, overriden_materials);
		apply_material_override(override_state.override_anisotropy, &RendererMaterial::anisotropy, material_override.anisotropy, overriden_materials);
		apply_material_override(override_state.override_anisotropy_rotation, &RendererMaterial::anisotropy_rotation, material_override.anisotropy_rotation, overriden_materials);

		apply_material_override(override_state.override_specular, &RendererMaterial::specular, material_override.specular, overriden_materials);
		apply_material_override(override_state.override_specular_color, &RendererMaterial::specular_color, material_override.specular_color, overriden_materials);
		apply_material_override(override_state.override_specular_tint_strength, &RendererMaterial::specular_tint, material_override.specular_tint, overriden_materials);

		apply_material_override(override_state.override_metallic, &RendererMaterial::metallic, material_override.metallic, overriden_materials);
		apply_material_override(override_state.override_F82_reflectivity, &RendererMaterial::metallic_F82, material_override.metallic_F82, overriden_materials);
		apply_material_override(override_state.override_F90_reflectivity, &RendererMaterial::metallic_F90, material_override.metallic_F90, overriden_materials);
		apply_material_override(override_state.override_F90_falloff_exponent, &RendererMaterial::metallic_F90_falloff_exponent, material_override.metallic_F90_falloff_exponent, overriden_materials);
		apply_material_override(override_state.override_second_roughness_weight, &RendererMaterial::second_roughness_weight, material_override.second_roughness_weight, overriden_materials);
		apply_material_override(override_state.override_second_roughness, &RendererMaterial::second_roughness, material_override.second_roughness, overriden_materials);

		apply_material_override(override_state.override_sheen_strength, &RendererMaterial::sheen, material_override.sheen, overriden_materials);
		apply_material_override(override_state.override_sheen_color, &RendererMaterial::sheen_color, material_override.sheen_color, overriden_materials);
		apply_material_override(override_state.override_sheen_roughness, &RendererMaterial::sheen_roughness, material_override.sheen_roughness, overriden_materials);

		apply_material_override(override_state.override_coat_strength, &RendererMaterial::coat, material_override.coat, overriden_materials);
		apply_material_override(override_state.override_coat_medium_absorption, &RendererMaterial::coat_medium_absorption, material_override.coat_medium_absorption, overriden_materials);
		apply_material_override(override_state.override_coat_medium_thickness, &RendererMaterial::coat_medium_thickness, material_override.coat_medium_thickness, overriden_materials);
		apply_material_override(override_state.override_coat_roughness, &RendererMaterial::coat_roughness, material_override.coat_roughness, overriden_materials);
		apply_material_override(override_state.override_coat_roughening, &RendererMaterial::coat_roughening, material_override.coat_roughening, overriden_materials);
		apply_material_override(override_state.override_coat_anisotropy, &RendererMaterial::coat_anisotropy, material_override.coat_anisotropy, overriden_materials);
		apply_material_override(override_state.override_coat_anisotropy_rotation, &RendererMaterial::coat_anisotropy_rotation, material_override.coat_anisotropy_rotation, overriden_materials);
		apply_material_override(override_state.override_coat_IOR, &RendererMaterial::coat_ior, material_override.coat_ior, overriden_materials);

		apply_material_override(override_state.override_transmission, &RendererMaterial::specular_transmission, material_override.specular_transmission, overriden_materials);
		apply_material_override(override_state.override_IOR, &RendererMaterial::ior, material_override.ior, overriden_materials);
		apply_material_override(override_state.override_absorption_distance, &RendererMaterial::absorption_at_distance, material_override.absorption_at_distance, overriden_materials);
		apply_material_override(override_state.override_absorption_color, &RendererMaterial::absorption_color, material_override.absorption_color, overriden_materials);
		apply_material_override(override_state.override_dielectric_priority, &RendererMaterial::dielectric_priority, material_override.dielectric_priority, overriden_materials);

		apply_material_override(override_state.override_thin_film, &RendererMaterial::thin_film, material_override.thin_film, overriden_materials);
		apply_material_override(override_state.override_thin_film_thickness, &RendererMaterial::thin_film_thickness, material_override.thin_film_thickness, overriden_materials);
		apply_material_override(override_state.override_thin_film_ior, &RendererMaterial::thin_film_ior, material_override.thin_film_ior, overriden_materials);

		// Special case for the emission since it's a private member
		if (override_state.override_emission)
			for (RendererMaterial& renderer_mat : overriden_materials)
				renderer_mat.set_emission(material_emission);
		apply_material_override(override_state.override_emission_strength, &RendererMaterial::emission_strength, material_override.emission_strength, overriden_materials);

		apply_material_override(override_state.override_opacity, &RendererMaterial::alpha_opacity, material_override.alpha_opacity, overriden_materials);

		m_renderer->update_materials(overriden_materials);
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

	// Keeping a backup of the materials_to_override. Useful when modifying the global emissive factor
	// of objects in the scene because we want the global factor to affect the original
	// emission of the materials_to_override, not the emission that has already been multiplied by
	// a previous factor: this would lead to a buggy exponential growth of the emission
	static std::vector<RendererMaterial> original_materials = m_renderer->get_materials();

	std::vector<RendererMaterial> materials = m_renderer->get_materials();
	const std::vector<std::string>& material_names = m_renderer->get_material_names();
	const std::vector<std::string>& mesh_names = m_renderer->get_mesh_names();

	bool material_changed = false;
	static int currently_selected_material = 0;

	if (ImGui::CollapsingHeader("All objects"))
	{
		ImGui::TreePush("All objects tree");

		if (ImGui::BeginListBox("##all_objects", ImVec2(-FLT_MIN, 7 * ImGui::GetTextLineHeightWithSpacing())))
		{
			for (int n = 0; n < materials.size(); n++)
			{
				const bool is_selected = (currently_selected_material == n);
				std::string text = mesh_names[n] + " (" + material_names[n] + ")";
				if (ImGui::Selectable(text.c_str(), is_selected))
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

		ImGui::PushItemWidth(16 * ImGui::GetFontSize());

		ImGui::Text("- "); ImGui::SameLine();
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.9f, 0.0f, 1.0f));
		ImGui::Text("Selected object"); ImGui::SameLine();
		ImGui::PopStyleColor();
		ImGui::Text(": "); ImGui::SameLine();
		ImGui::Text("%s", material_names[currently_selected_material].c_str());

		if (ImGui::CollapsingHeader("Base Layer"))
		{
			ImGui::TreePush("Base layer material tree");

			material_changed |= ImGui::ColorEdit3("Base color", (float*)&material.base_color);
			material_changed |= ImGui::SliderFloat("Roughness", &material.roughness, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Anisotropy", &material.anisotropy, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Anisotropy rotation", &material.anisotropy_rotation, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("IOR", &material.ior, 1.0f, 3.0f);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Specular layer"))
		{
			ImGui::TreePush("Specular layer material tree");

			material_changed |= ImGui::SliderFloat("Specular", &material.specular, 0.0f, 1.0f);
			material_changed |= ImGui::ColorEdit3("Specular color", (float*)&material.specular_color);
			material_changed |= ImGui::SliderFloat("Specular tint strength", &material.specular_tint, 0.0f, 1.0f);

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
			material_changed |= ImGui::SliderFloat("Coat anisotropy", &material.coat_anisotropy, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Coat anisotropy Rotation", &material.coat_anisotropy_rotation, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("Coat IOR", &material.coat_ior, 1.0f, 3.0f);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Transmission Layer"))
		{
			ImGui::TreePush("Transmission layer material tree");

			material_changed |= ImGui::SliderFloat("Transmission", &material.specular_transmission, 0.0f, 1.0f);
			material_changed |= ImGui::SliderFloat("IOR", &material.ior, 1.0f, 3.0f);
			material_changed |= ImGui::SliderFloat("Absorption distance", &material.absorption_at_distance, 0.0f, 20.0f);
			material_changed |= ImGui::ColorEdit3("Absorption color", (float*)&material.absorption_color);
			ImGui::BeginDisabled(kernel_options->get_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITIES);
			material_changed |= ImGui::SliderInt("Dielectric priority", &material.dielectric_priority, 1, StackPriorityEntry::PRIORITY_MAXIMUM);
			if (kernel_options->get_macro_value(GPUKernelCompilerOptions::INTERIOR_STACK_STRATEGY) != ISS_WITH_PRIORITIES)
				ImGuiRenderer::show_help_marker("Disabled because not using nested dielectrics with priorities.");
			ImGui::EndDisabled();

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

			ImGui::BeginDisabled(material.emission_texture_index > 0);
			// Displaying original emission in ImGui
			ColorRGB32F material_emission = ColorRGB32F(0.0f);
			if (material.emission_strength > 0.0f)
				material_emission = material.get_emission() / material.emission_strength;
			if (ImGui::ColorEdit3("Emission", (float*)&material_emission, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float))
			{
				material.set_emission(material_emission / material.emission_strength);

				material_changed = true;
			}
			ImGui::EndDisabled();
			if (material.emission_texture_index > 0)
				ImGuiRenderer::show_help_marker("Disabled because the emission of this material is controlled by a texture");

			material_changed |= ImGui::SliderFloat("Emission Strength", &material.emission_strength, 0.0f, 10.0f);

			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImGui::TreePop();
		}

		if (ImGui::CollapsingHeader("Other properties"))
		{
			ImGui::TreePush("Other properties material tree");

			material_changed |= ImGui::SliderFloat("Opacity", &material.alpha_opacity, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);

			ImGui::TreePop();
		}

		ImGui::PopItemWidth();
		ImGui::Dummy(ImVec2(0.0f, 20.0f));
		ImGui::Separator();

		material_changed |= draw_material_presets(material);

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

bool ImGuiObjectsWindow::draw_material_presets(RendererMaterial& material)
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
		{ "Silver", { ColorRGB32F(0.9868, 0.9830, 0.9667), ColorRGB32F(0.9929, 0.9961, 1.0000) } },
		{ "Aluminum", { ColorRGB32F(0.9157, 0.9226, 0.9236), ColorRGB32F(0.9090, 0.9365, 0.9596) } },
		{ "Gold", { ColorRGB32F(1.0000, 0.7099, 0.3148), ColorRGB32F(0.9408, 0.9636, 0.9099) } },
		{ "Chromium", { ColorRGB32F(0.5496, 0.5561, 0.5531), ColorRGB32F(0.7372, 0.7511, 0.8170) } },
		{ "Copper", { ColorRGB32F(1.0000, 0.6504, 0.5274), ColorRGB32F(0.9755, 0.9349, 0.9301) } },
		{ "Iron", { ColorRGB32F(0.8951, 0.8755, 0.8154), ColorRGB32F(0.8551, 0.8800, 0.8966) } },
		{ "Mercury", { ColorRGB32F(0.7815, 0.7795, 0.7783), ColorRGB32F(0.8103, 0.8532, 0.9046) } },
		{ "Magnesium", { ColorRGB32F(0.8918, 0.8821, 0.8948), ColorRGB32F(0.8949, 0.9147, 0.9504) } },
		{ "Nickel", { ColorRGB32F(0.7014, 0.6382, 0.5593), ColorRGB32F(0.8134, 0.8352, 0.8725) } },
		{ "Lead", { ColorRGB32F(0.7363, 0.7023, 0.6602), ColorRGB32F(0.8095, 0.8369, 0.8739) } },
		{ "Platinum", { ColorRGB32F(0.9602, 0.9317, 0.8260), ColorRGB32F(0.9501, 0.9461, 0.9352) } },
		{ "Titanium", { ColorRGB32F(0.4432, 0.3993, 0.3599), ColorRGB32F(0.8627, 0.9066, 0.9481) } },
		{ "Zinc", { ColorRGB32F(0.8759, 0.8685, 0.8542), ColorRGB32F(0.8769, 0.9037, 0.9341) } },
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
			material = RendererMaterial();

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
