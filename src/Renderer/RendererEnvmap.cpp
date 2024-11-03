/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Image/Image.h"
#include "Renderer/GPURenderer.h"
#include "Renderer/RendererEnvmap.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

void RendererEnvmap::init_from_image(const Image32Bit& image, const std::string& envmap_filepath)
{
	m_orochi_envmap.init_from_image(image);
	m_envmap_filepath = envmap_filepath;
}

void RendererEnvmap::update(GPURenderer* renderer)
{
	do_animation(renderer);

	// Updates the data/pointers in WorldSettings that the shaders will use
	update_renderer(renderer);
}

void RendererEnvmap::recompute_sampling_data_structure(GPURenderer* renderer, const Image32Bit* image)
{
	if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_NO_SAMPLING)
	{
		m_orochi_envmap.free_cdf();
		m_orochi_envmap.free_alias_table();
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_BINARY_SEARCH)
	{
		if (image != nullptr)
			m_orochi_envmap.compute_cdf(*image);
		else
			m_orochi_envmap.compute_cdf(Image32Bit::read_image_hdr(m_envmap_filepath, 4, true));

		m_orochi_envmap.free_alias_table();
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_ALIAS_TABLE)
	{
		if (image != nullptr)
			m_orochi_envmap.compute_alias_table(*image);
		else
			m_orochi_envmap.compute_alias_table(Image32Bit::read_image_hdr(m_envmap_filepath, 4, true));

		m_orochi_envmap.free_cdf();
	}
}

void RendererEnvmap::do_animation(GPURenderer* renderer)
{
	// We can step the animation either if we're not accumulating or
	// if we're accumulating and we're allowed to step the animations
	bool can_step_animation = false;
	can_step_animation |= renderer->get_render_settings().accumulate && renderer->get_animation_state().can_step_animation;
	can_step_animation |= !renderer->get_render_settings().accumulate;

	if (animate && renderer->get_animation_state().do_animations && can_step_animation)
	{
		float renderer_delta_time = renderer->get_last_frame_time();

		rotation_X += animation_speed_X / 360.0f / (1000.0f / renderer_delta_time);
		rotation_Y += animation_speed_Y / 360.0f / (1000.0f / renderer_delta_time);
		rotation_Z += animation_speed_Z / 360.0f / (1000.0f / renderer_delta_time);

		rotation_X = rotation_X - static_cast<int>(rotation_X);
		rotation_Y = rotation_Y - static_cast<int>(rotation_Y);
		rotation_Z = rotation_Z - static_cast<int>(rotation_Z);
	}

	if (rotation_X != prev_rotation_X || rotation_Y != prev_rotation_Y || rotation_Z != prev_rotation_Z)
	{
		glm::mat4x4 rotation_matrix, rotation_matrix_inv;

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
		rotation_matrix = glm::orientate3(glm::vec3(rotation_X * 2.0f * M_PI, rotation_Z * 2.0f * M_PI, rotation_Y * 2.0f * M_PI));
		rotation_matrix_inv = glm::inverse(rotation_matrix);

		envmap_to_world_matrix = *reinterpret_cast<float4x4*>(&rotation_matrix);
		world_to_envmap_matrix = *reinterpret_cast<float4x4*>(&rotation_matrix_inv);

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

		world_settings.alias_table_probas = nullptr;
		world_settings.alias_table_alias = nullptr;
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_BINARY_SEARCH)
	{
		world_settings.envmap_cdf = m_orochi_envmap.get_cdf_device_pointer();
		world_settings.envmap_total_sum = m_orochi_envmap.get_luminance_total_sum();

		world_settings.alias_table_probas = nullptr;
		world_settings.alias_table_alias = nullptr;
	}
	else if (renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::ENVMAP_SAMPLING_STRATEGY) == ESS_ALIAS_TABLE)
	{
		world_settings.envmap_cdf = nullptr;
		world_settings.envmap_total_sum = m_orochi_envmap.get_luminance_total_sum();

		m_orochi_envmap.get_alias_table_device_pointers(world_settings.alias_table_probas, world_settings.alias_table_alias);
	}
}

OrochiEnvmap& RendererEnvmap::get_orochi_envmap()
{
	return m_orochi_envmap;
}
