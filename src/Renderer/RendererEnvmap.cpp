/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RendererEnvmap.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

void RendererEnvmap::init_from_image(const Image32Bit& image, const std::string& envmap_filepath)
{
	m_envmap_data.pack_from(image);
	m_envmap_filepath = envmap_filepath;

	m_width = image.width;
	m_height = image.height;
}

void RendererEnvmap::update(GPURenderer* renderer, float delta_time)
{
	do_animation(renderer, delta_time);

	// Updates the data/pointers in WorldSettings that the shaders will use
	update_renderer(renderer);
}

void RendererEnvmap::recompute_sampling_data_structure(GPURenderer* renderer, const Image32Bit* image)
{
	if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_NO_SAMPLING)
	{
		if (m_cdf.size() > 0)
			m_cdf.free();

		if (m_alias_table_alias.size() > 0)
			m_alias_table_alias.free();

		if (m_alias_table_probas.size() > 0)
			m_alias_table_probas.free();
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_BINARY_SEARCH)
	{
		if (m_alias_table_alias.size() > 0)
			m_alias_table_alias.free();

		if (m_alias_table_probas.size() > 0)
			m_alias_table_probas.free();

		recompute_CDF(image);
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_ALIAS_TABLE)
	{
		if (m_cdf.size() > 0)
			m_cdf.free();

		recompute_alias_table(image);
	}
}

void RendererEnvmap::recompute_CDF(const Image32Bit* image)
{
	std::vector<float> cdf_data;
	if (image != nullptr)
		cdf_data = image->compute_cdf();
	else
	{
		if (m_envmap_filepath.ends_with(".exr"))
			cdf_data = Image32Bit::read_image_exr(m_envmap_filepath, true).compute_cdf();
		else
			cdf_data = Image32Bit::read_image_hdr(m_envmap_filepath, 4, true).compute_cdf();
	}

	m_cdf.resize(cdf_data.size());
	m_cdf.upload_data(cdf_data);
	m_luminance_total_sum = cdf_data.back();
}

void RendererEnvmap::recompute_alias_table(const Image32Bit* image)
{
	std::vector<float> probas;
	std::vector<int> alias;
	if (image != nullptr)
		image->compute_alias_table(probas, alias, &m_luminance_total_sum);
	else
	{
		if (m_envmap_filepath.ends_with(".exr"))
			Image32Bit::read_image_exr(m_envmap_filepath, true).compute_alias_table(probas, alias, &m_luminance_total_sum);
		else
			Image32Bit::read_image_hdr(m_envmap_filepath, 4, true).compute_alias_table(probas, alias, &m_luminance_total_sum);
	}

	m_alias_table_probas.resize(probas.size());
	m_alias_table_probas.upload_data(probas);
	m_alias_table_alias.resize(alias.size());
	m_alias_table_alias.upload_data(alias);
}

RGBE9995Packed* RendererEnvmap::get_packed_data_pointer()
{
	return m_envmap_data.get_data_pointer();
}

void RendererEnvmap::get_alias_table_device_pointers(float*& out_probas_pointer, int*& out_alias_pointer)
{
	out_probas_pointer = m_alias_table_probas.get_device_pointer();
	out_alias_pointer = m_alias_table_alias.get_device_pointer();
}

float* RendererEnvmap::get_cdf_device_pointer()
{
	return m_cdf.get_device_pointer();
}

unsigned int RendererEnvmap::get_width() { return m_width; }
unsigned int RendererEnvmap::get_height() { return m_height; }

float RendererEnvmap::get_sampling_structure_VRAM_usage() const
{
	// Just return the sum of everything (both the CDF and alias table) because only one can be
	// used at a given time so one of the two will be 0 bytes anyways
	return (m_cdf.get_byte_size() + m_alias_table_probas.get_byte_size() + m_alias_table_alias.get_byte_size()) / 1000000.0f;
}

void RendererEnvmap::do_animation(GPURenderer* renderer, float delta_time)
{
	// We can step the animation either if we're not accumulating or
	// if we're accumulating and we're allowed to step the animations
	bool can_step_animation = false;
	can_step_animation |= renderer->get_render_settings().accumulate && renderer->get_animation_state().can_step_animation;
	can_step_animation |= !renderer->get_render_settings().accumulate;

	if (animate && renderer->get_animation_state().do_animations && can_step_animation)
	{
		rotation_X += animation_speed_X / 360.0f / (1000.0f / delta_time);
		rotation_Y += animation_speed_Y / 360.0f / (1000.0f / delta_time);
		rotation_Z += animation_speed_Z / 360.0f / (1000.0f / delta_time);

		rotation_X = rotation_X - static_cast<int>(rotation_X);
		rotation_Y = rotation_Y - static_cast<int>(rotation_Y);
		rotation_Z = rotation_Z - static_cast<int>(rotation_Z);
	}

	if (rotation_X != prev_rotation_X || rotation_Y != prev_rotation_Y || rotation_Z != prev_rotation_Z)
	{
		glm::mat3x3 rotation_matrix, rotation_matrix_inv;

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
		rotation_matrix = glm::orientate3(glm::vec3(rotation_X * M_TWO_PI, rotation_Z * M_TWO_PI, rotation_Y * M_TWO_PI));
		rotation_matrix_inv = glm::inverse(rotation_matrix);

		envmap_to_world_matrix = *reinterpret_cast<float3x3*>(&rotation_matrix);
		world_to_envmap_matrix = *reinterpret_cast<float3x3*>(&rotation_matrix_inv);

		prev_rotation_X = rotation_X;
		prev_rotation_Y = rotation_Y;
		prev_rotation_Z = rotation_Z;
	}
}

void RendererEnvmap::update_renderer(GPURenderer* renderer)
{
	WorldSettings& world_settings = renderer->get_world_settings();

	world_settings.envmap_to_world_matrix = envmap_to_world_matrix;
	world_settings.world_to_envmap_matrix = world_to_envmap_matrix;

	if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_NO_SAMPLING)
	{
		world_settings.envmap_cdf = nullptr;

		world_settings.envmap_alias_table.alias_table_probas = nullptr;
		world_settings.envmap_alias_table.alias_table_alias = nullptr;
		world_settings.envmap_alias_table.size = 0;
		world_settings.envmap_alias_table.sum_elements = 0;
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_BINARY_SEARCH)
	{
		world_settings.envmap_cdf = m_cdf.get_device_pointer();
		world_settings.envmap_total_sum = m_luminance_total_sum;

		world_settings.envmap_alias_table.alias_table_probas = nullptr;
		world_settings.envmap_alias_table.alias_table_alias = nullptr;
		world_settings.envmap_alias_table.size = 0;
		world_settings.envmap_alias_table.sum_elements = 0;
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_ALIAS_TABLE)
	{
		world_settings.envmap_cdf = nullptr;
		world_settings.envmap_total_sum = m_luminance_total_sum;

		world_settings.envmap_alias_table.alias_table_probas = m_alias_table_probas.get_device_pointer();
		world_settings.envmap_alias_table.alias_table_alias = m_alias_table_alias.get_device_pointer();
		world_settings.envmap_alias_table.size = m_alias_table_alias.size();
		world_settings.envmap_alias_table.sum_elements = m_luminance_total_sum;
	}
}
