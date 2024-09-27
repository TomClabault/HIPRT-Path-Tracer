/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RendererEnvmap.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/euler_angles.hpp"

void RendererEnvmap::update(GPURenderer* renderer)
{
	if (animate)
	{
		float renderer_delta_time = renderer->get_render_pass_time(GPURenderer::FULL_FRAME_TIME_KEY);

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

		update_renderer(renderer);
	}

	prev_rotation_X = rotation_X;
	prev_rotation_Y = rotation_Y;
	prev_rotation_Z = rotation_Z;

}

void RendererEnvmap::update_renderer(GPURenderer* renderer)
{
	renderer->get_world_settings().envmap_to_world_matrix = envmap_to_world_matrix;
	renderer->get_world_settings().world_to_envmap_matrix = world_to_envmap_matrix;
}

OrochiEnvmap& RendererEnvmap::get_orochi_envmap()
{
	return m_orochi_envmap;
}
